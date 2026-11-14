# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Musculoskeletal MJCF / ``MjSpec`` builders backed by ``myo_sim``.

All geometry, meshes, textures, tendons, and Hill-type muscles are resolved
directly from the installed ``myo_sim`` package so ``myosuite-reaf`` ships zero
duplicate binary STL/PNG assets while remaining self-contained on Python 3.12+
and MuJoCo 4.x.
"""

from __future__ import annotations

import hashlib
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import mujoco
import myo_sim
import numpy as np

_PATCH_DIR = Path(tempfile.gettempdir()) / "myosuite_reaf_assets"

_HAND_TIP_SITES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("THtip_r", (0.8, 0.0, 0.0, 0.8)),
    ("IFtip_r", (0.0, 0.8, 0.0, 0.8)),
    ("MFtip_r", (0.0, 0.0, 0.8, 0.8)),
    ("RFtip_r", (0.8, 0.8, 0.0, 0.8)),
    ("LFtip_r", (0.8, 0.0, 0.8, 0.8)),
)

_REACH_WORKSPACE_CACHE: dict[tuple[object, ...], np.ndarray] = {}


def _write_cached_xml(content: str, stem: str) -> Path:
    """Write *content* atomically into a content-hashed temp XML file."""
    _PATCH_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    dest = _PATCH_DIR / f"{stem}_{digest}.xml"
    if not dest.exists():
        tmp = dest.with_suffix(f".{digest}.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(dest)
    return dest


def _absolutize_scene_xml(scene_path: Path) -> Path:
    """Return a cached copy of *scene_path* with absolute compiler/include/mesh paths."""
    tree = ET.parse(scene_path)
    root = tree.getroot()
    base = scene_path.parent
    meshes_pool = myo_sim.MODELS_DIR.resolve() / "meshes"
    for include in root.findall("include"):
        file_attr = include.get("file")
        if file_attr and not Path(file_attr).is_absolute():
            include.set("file", (base / file_attr).resolve().as_posix())
    for compiler in root.findall("compiler"):
        for attr in ("meshdir", "texturedir"):
            val = compiler.get(attr)
            if val and not Path(val).is_absolute():
                compiler.set(attr, (base / val).resolve().as_posix())
    for elem in root.iter():
        if elem.tag in ("mesh", "texture"):
            raw = elem.get("file")
            if raw and not Path(raw).is_absolute():
                local = (base / raw).resolve()
                if local.exists():
                    elem.set("file", local.as_posix())
                else:
                    pooled = (meshes_pool / Path(raw).name).resolve()
                    if pooled.exists():
                        elem.set("file", pooled.as_posix())
    xml_bytes = ET.tostring(root, encoding="unicode")
    return _write_cached_xml(xml_bytes, scene_path.stem)


def peak_isometric_force(model: mujoco.MjModel) -> np.ndarray:
    """Resolve effective per-actuator peak force ``gainprm[:, 2]``.

    MuJoCo muscles declared with ``force="-1"`` compute peak force at runtime
    as ``gainprm[3] / max(actuator_acc0, mjMINVAL)``. Resolving those before
    scaling prevents sarcopenia from multiplying ``-1`` into ``-0.5``.
    """
    force = np.array(model.actuator_gainprm[:, 2], dtype=np.float64)
    auto = (force < 0.0) & (model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
    if np.any(auto):
        force[auto] = model.actuator_gainprm[auto, 3] / np.maximum(
            model.actuator_acc0[auto], mujoco.mjMINVAL
        )
    return force


def apply_sarcopenia_to_spec(spec: mujoco.MjSpec, force_scale: float = 0.5) -> mujoco.MjSpec:
    """Scale peak isometric force of every actuator in *spec* in-place."""
    resolved = peak_isometric_force(spec.compile())
    for actuator, force in zip(spec.actuators, resolved, strict=True):
        actuator.gainprm[2] = float(force) * force_scale
    return spec


def apply_sarcopenia_to_model(model: mujoco.MjModel, force_scale: float = 0.5) -> None:
    """Scale peak isometric force on a compiled ``MjModel`` in-place."""
    model.actuator_gainprm[:, 2] = peak_isometric_force(model) * force_scale


def build_elbow_spec(*, exo: bool = False) -> mujoco.MjSpec:
    """Build 1-DoF 6-muscle elbow (or hybrid 6-muscle + 1-motor exo) ``MjSpec``."""
    filename = "myoelbow_1dof6muscles_1dofexo.xml" if exo else "myoelbow_1dof6muscles.xml"
    xml_path = myo_sim.MODELS_DIR / "legacy" / "elbow" / filename
    if not xml_path.exists():
        xml_path = myo_sim.MODELS_DIR / "elbow" / filename
    spec = mujoco.MjSpec.from_file(str(xml_path))
    if spec.site("wrist_target") is None:
        site = spec.worldbody.add_site()
        site.name = "wrist_target"
        site.pos = [0.007, -0.261, 0.065]
        site.size = [0.005, 0.005, 0.005]
    return spec


def build_finger_spec(*, motor: bool = False) -> mujoco.MjSpec:
    """Build 4-DoF 5-actuator tendon finger (muscle or motor) ``MjSpec``."""
    filename = "motorfinger_v0.xml" if motor else "myofinger_v0.xml"
    xml_path = myo_sim.MODELS_DIR / "legacy" / "finger" / filename
    if not xml_path.exists():
        xml_path = myo_sim.MODELS_DIR / "finger" / filename
    return mujoco.MjSpec.from_file(str(xml_path))


def build_hand_pose_spec() -> mujoco.MjSpec:
    """Build 23-DoF 39-muscle right hand ``MjSpec`` with 5 fingertip target sites."""
    hand_xml = myo_sim.MODELS_DIR / "hand" / "myohand_r.xml"
    if hand_xml.exists():
        spec = mujoco.MjSpec.from_file(str(hand_xml))
    else:
        from myo_sim.build.compose import build_right_hand_from_arm_spec

        spec = build_right_hand_from_arm_spec()
        root = spec.body("myoarm_r_root")
        root.pos = [-0.02525155, 0.07146933, 1.39249298]
        root.quat = [0.52167746, 0.46981366, -0.47875126, -0.52718591]

    for tip_name, rgba in _HAND_TIP_SITES:
        target_name = f"{tip_name}_target"
        if spec.site(target_name) is None:
            site = spec.worldbody.add_site()
            site.name = target_name
            site.pos = [0.0, 0.0, 0.002]
            site.size = [0.005, 0.005, 0.005]
            site.rgba = list(rgba)
    return spec


_ARM_REACH_CACHED_SPEC: mujoco.MjSpec | None = None


def _recursive_immobilize(
    spec: mujoco.MjSpec,
    temp_model: mujoco.MjModel,
    parent: mujoco.MjsBody,
    *,
    remove_sites: bool = False,
) -> list[int]:
    """Delete joints in *parent* and its subtree (matching ``myosuite.utils.spec_processing``)."""
    removed_qpos_adrs: list[int] = []
    if remove_sites:
        for site in list(parent.sites):
            spec.delete(site)
    for jnt in list(parent.joints):
        removed_qpos_adrs.append(int(temp_model.joint(jnt.name).qposadr[0]))
        spec.delete(jnt)
    for child in list(parent.bodies):
        removed_qpos_adrs.extend(
            _recursive_immobilize(spec, temp_model, child, remove_sites=remove_sites)
        )
    return removed_qpos_adrs


def build_arm_reach_spec() -> mujoco.MjSpec:
    """Build 18-DoF 50-muscle right arm ``MjSpec`` matching ``myoArmReach*-v0``.

    Reproduces MyoSuite's ``full_arm`` + ``edit_fn_arm_reaching`` composition:
    calibrates ``myo_sim.load_spec("myoarm_r")`` into the legacy world frame,
    immobilizes the 5 finger digit chains, prunes zero-moment finger tendons
    and actuators, and adds the ``IFtip`` fingertip site and ``IFtip_target``.
    """
    global _ARM_REACH_CACHED_SPEC
    if _ARM_REACH_CACHED_SPEC is not None:
        return _ARM_REACH_CACHED_SPEC.copy()

    arm_spec = myo_sim.load_spec("myoarm_r")
    root = arm_spec.body("Full Body")
    root.pos = [0.00685025, -0.00150371, 0.91886071]
    root.quat = [0.99999984, -0.00039816, 0.00000016, 0.00039816]

    spec = mujoco.MjSpec()
    frame = spec.worldbody.add_frame()
    spec.attach(arm_spec, prefix="", suffix="", frame=frame)
    spec.compile()

    spec.stat.extent = 1.5
    spec.stat.center = [-0.17, -0.3, 1.2]
    for digit_root in (
        "firstmc_r",
        "secondmc_r",
        "thirdmc_r",
        "fourthmc_r",
        "fifthmc_r",
    ):
        _recursive_immobilize(
            spec,
            spec.copy().compile(),
            spec.body(digit_root),
            remove_sites=False,
        )

    temp_m = spec.compile()
    temp_d = mujoco.MjData(temp_m)
    mujoco.mj_step(temp_m, temp_d, nstep=100)

    j_tendon = np.empty((temp_m.ntendon, temp_m.nv))
    mujoco.mju_sparse2dense(
        j_tendon,
        temp_d.ten_J,
        temp_m.ten_J_rownnz,
        temp_m.ten_J_rowadr,
        temp_m.ten_J_colind,
    )
    for ten in list(spec.tendons):
        if np.sum(np.abs(j_tendon[temp_m.tendon(ten.name).id, :])) < 1e-10:
            for act in list(spec.actuators):
                if act.target == ten.name:
                    spec.delete(act)
            spec.delete(ten)

    og_site = spec.site("IFtip_r")
    spec.body("distph2_r").add_site(
        name="IFtip",
        size=og_site.size,
        pos=og_site.pos,
        rgba=og_site.rgba,
        type=og_site.type,
    )
    spec.worldbody.add_site(
        name="IFtip_target",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0.02, 0.02],
        pos=[-0.2, -0.2, 1.2],
        rgba=[0.0, 0.0, 1.0, 0.3],
    )
    _ARM_REACH_CACHED_SPEC = spec
    return spec.copy()


def build_torso_spec(*, exo: bool = False) -> mujoco.MjSpec:
    """Build articulated 210-muscle MyoTorso (or passive-exosuit variant) ``MjSpec``."""
    models_dir = myo_sim.MODELS_DIR.resolve()
    meshdir = models_dir.as_posix().rstrip("/") + "/"
    scene_path = _absolutize_scene_xml(models_dir / "scene" / "myosuite_scene.xml").as_posix()
    torso_assets = (models_dir / "torso" / "assets").as_posix()
    head_assets = (models_dir / "head" / "assets" / "myohead_simple_assets.xml").as_posix()

    if not exo:
        xml_text = f"""<mujoco model="MyoTorso_v1.0">
  <include file="{torso_assets}/myotorso_assets.xml"/>
  <include file="{torso_assets}/myotorso_muscle.xml"/>
  <include file="{torso_assets}/myotorso_tendon.xml"/>
  <include file="{torso_assets}/myotorso_abdomen_muscle.xml"/>
  <include file="{torso_assets}/myotorso_abdomen_tendon.xml"/>
  <include file="{head_assets}"/>
  <include file="{scene_path}"/>
  <compiler meshdir="{meshdir}" texturedir="{meshdir}"/>
  <worldbody>
    <body name="Torso" pos="-0.025 0.1 0.9">
      <include file="{torso_assets}/myotorso_chain.xml"/>
    </body>
  </worldbody>
</mujoco>
"""
        return mujoco.MjSpec.from_file(str(_write_cached_xml(xml_text, "myotorso")))

    meshes_dir = (models_dir / "meshes").as_posix()
    xml_text = f"""<mujoco model="MyoExoTorso">
  <include file="{scene_path}"/>
  <include file="{torso_assets}/myotorso_assets.xml"/>
  <include file="{torso_assets}/myotorso_muscle.xml"/>
  <include file="{torso_assets}/myotorso_tendon.xml"/>
  <include file="{torso_assets}/myotorso_abdomen_muscle.xml"/>
  <include file="{torso_assets}/myotorso_abdomen_tendon.xml"/>
  <include file="{head_assets}"/>
  <compiler meshdir="{meshdir}" texturedir="{meshdir}"/>
  <default class="main">
    <default class="myotorsoexo">
      <geom margin="0.001" material="mat_myotorso" rgba="0.35 0.3 0.3 1" conaffinity="0" contype="0"/>
      <site size="0.001 0.005 0.005"/>
      <tendon limited="false" width="0.006" group="1" rgba="0.38 0.33 0.33 1"/>
    </default>
  </default>
  <asset>
    <mesh name="lower_exo_belt_geom_1" file="{meshes_dir}/lower_exo_belt.stl" scale="0.0015 0.0015 0.0013"/>
    <mesh name="lower_exo_legs_geom_1" file="{meshes_dir}/lower_exo_legs.stl" scale="0.0013 0.0013 0.0013"/>
    <mesh name="upper_exo_geom_1" file="{meshes_dir}/upper_exo.stl" scale="0.001 0.0011 0.001"/>
  </asset>
  <worldbody>
    <body name="full_body" pos="0.005 0.1 0.93">
      <include file="{torso_assets}/myotorso_chain.xml"/>
    </body>
    <body name="exo_pelvis" pos="0.005 0.13 0.979" quat="0.499949 0.500051 -0.500051 -0.499949" childclass="myotorsoexo">
      <geom name="lower_exo_legs_geom" type="mesh" euler="1.57 3.14 -1.57" pos="-0.08 0.1 0" mesh="lower_exo_legs_geom_1"/>
      <site name="Lower_Exo_Left_Leg" pos="-0.08 -0.15 -0.17" size="0.005" rgba="0.4 0.4 0.4 1"/>
      <site name="Lower_Exo_Right_Leg" pos="-0.08 -0.15 0.17" size="0.005" rgba="0.4 0.4 0.4 1"/>
      <site name="Exo_LeftLeg" pos="-0.2 0 -0.04" size="0.0005" rgba="0 1 0 1"/>
      <site name="Exo_RightLeg" pos="-0.2 0 0.04" size="0.0005" rgba="0 1 0 1"/>
    </body>
    <body name="exo_lumbar4" pos="-0.004 0.2001 1.05" quat="0.499949 0.500051 -0.500051 -0.499949" childclass="myotorsoexo">
      <freejoint/>
      <site name="Lower_Exo_Left_Belt" pos="0.02 0.02 -0.158" size="0.005" rgba="0.4 0.4 0.4 1"/>
      <site name="Lower_Exo_Right_Belt" pos="0.02 0.02 0.158" size="0.005" rgba="0.4 0.4 0.4 1"/>
      <geom name="lower_exo_belt_geom" type="mesh" euler="1.57 3.14 -1.57" pos="0.03 0.03 0" mesh="lower_exo_belt_geom_1" mass="0.094"/>
    </body>
    <body name="exo_torso" pos="0.0045 0.216 1.06" quat="0.499949 0.500051 -0.500051 -0.499949" childclass="myotorsoexo">
      <freejoint/>
      <geom name="upper_exo_geom" type="mesh" euler="1.57 3.14 1.57" pos="0.01 0.165 0" mesh="upper_exo_geom_1"/>
      <geom name="Exo_cylinder" size="0.105 0.13" pos="0.025 0.03 0" class="myotorso_back_wrap"/>
      <site name="Exo_LeftShoulder" pos="-0.11 0.28 -0.07" size="0.0005" rgba="0 1 0 1"/>
      <site name="Exo_RightShoulder" pos="-0.11 0.28 0.07" size="0.0005" rgba="0 1 0 1"/>
      <site name="Exo_sidesite" pos="-0.14 0.05 0" class="myotorso_sidesite"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="torso" body2="exo_torso" anchor="-0.1 0.2 0"/>
    <weld body1="lumbar4" body2="exo_lumbar4"/>
  </equality>
  <tendon>
    <spatial name="Exo_LS_RL" stiffness="515.5998876" springlength="0 0.37" class="myotorsoexo">
      <site site="Exo_RightShoulder"/>
      <geom geom="Exo_cylinder" sidesite="Exo_sidesite"/>
      <site site="Exo_LeftLeg"/>
    </spatial>
    <spatial name="Exo_RS_LL" stiffness="515.5998876" springlength="0 0.37" class="myotorsoexo">
      <site site="Exo_LeftShoulder"/>
      <geom geom="Exo_cylinder" sidesite="Exo_sidesite"/>
      <site site="Exo_RightLeg"/>
    </spatial>
    <spatial name="Lower_Exo_Left" stiffness="0" class="myotorsoexo">
      <site site="Lower_Exo_Left_Belt"/>
      <site site="Lower_Exo_Left_Leg"/>
    </spatial>
    <spatial name="Lower_Exo_Right" stiffness="0" class="myotorsoexo">
      <site site="Lower_Exo_Right_Belt"/>
      <site site="Lower_Exo_Right_Leg"/>
    </spatial>
  </tendon>
</mujoco>
"""
    return mujoco.MjSpec.from_file(str(_write_cached_xml(xml_text, "myotorso_exo")))


def build_leg_walk_spec() -> mujoco.MjSpec:
    """Build 80-muscle bipedal MyoLeg + rigid torso ``MjSpec`` matching ``myolegs_with_torso.xml``."""
    models_dir = myo_sim.MODELS_DIR.resolve()
    meshdir = models_dir.as_posix().rstrip("/") + "/"
    scene_path = _absolutize_scene_xml(models_dir / "scene" / "myosuite_scene.xml").as_posix()
    rigid_torso_dir = (models_dir / "legacy" / "torso" / "assets").as_posix()
    rigid_torso_assets_xml = _absolutize_scene_xml(
        models_dir / "legacy" / "torso" / "assets" / "myotorso_rigid_assets.xml"
    ).as_posix()
    leg_assets_dir = (models_dir / "leg" / "assets").as_posix()

    # Note: myo_sim==0.2.5's legacy myotorso_rigid_chain.xml still carries a stale
    # euler="0 0 -1.57" on <body name="torso">; wrapping it in <frame euler="0 0 1.57">
    # restores identity orientation inside root, matching MyoSuite's myolegs_with_torso.xml.
    xml_text = f"""<mujoco model="MyoSuite's MyoLeg">
  <include file="{rigid_torso_assets_xml}"/>
  <include file="{leg_assets_dir}/myolegs_tendon.xml"/>
  <include file="{leg_assets_dir}/myolegs_muscle.xml"/>
  <include file="{leg_assets_dir}/myolegs_assets.xml"/>
  <include file="{scene_path}"/>
  <compiler meshdir="{meshdir}" texturedir="{meshdir}"/>

  <asset>
    <hfield name="terrain" size="6 6 1 0.001" nrow="100" ncol="100"/>
  </asset>

  <worldbody>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 -10.0" material="matfloor" conaffinity="1" contype="1" rgba="0.9 0.9 0.9 0.0"/>
    <site name="pelvis_target" size="0.02" pos="0 0 0.92" group="4"/>
    <body name="root" pos="0 0 1" euler="0 0 -1.57">
      <site name="head" size="0.02" pos="-0.1 0 0.72" group="4"/>
      <frame euler="0 0 1.57">
        <include file="{rigid_torso_dir}/myotorso_rigid_chain.xml"/>
      </frame>
      <frame euler="0 0 1.57">
        <include file="{leg_assets_dir}/myolegs_chain.xml"/>
      </frame>
      <freejoint name="root"/>
    </body>
  </worldbody>

  <keyframe>
    <key qpos="0 0 .92 0.707388 0 0 -0.706825 0.161153 -0.0279385 -0.041886 0.00247908 0.00101098 0.461137 0.0275069 0.136817 0.334 -0.00117055 -0.000125295 -0.0302192 0.0395202 -0.194029 0.161153 -0.0279385 -0.041886 0.00247908 0.00101098 0.461137 0.0275069 0.136817 0.334 -0.00117055 -0.000125295 -0.0302192 0.0395202 -0.194029"/>
    <key qpos="0 0 .9 0.707388 0 0 -0.706825 0.405648 -0.020957 -0.118677 0.0039054 0.00122326 0.7329 0.0102961 0.215496 0.40143 -0.006982 -0.02618 -0.03738 0.0080579 -0.87272 0.405648 -0.020957 -0.118677 0.0039054 0.00122326 0.7329 0.0102961 0.215496 0.40143 -0.006982 -0.02618 -0.03738 0.0080579 -0.87272"/>
    <key qpos="0 0 1.0 0.707388 0 0 -0.706825 -0.2326 -0.0279385 -0.041886 0.00247908 0.00101098 1.227 0.0275069 0.136817 0.1672 -0.00117055 -0.000125295 -0.0302192 0.0395202 -0.194029 -0.1652 -0.0279385 -0.041886 0.00247908 0.0010198 0.0888 0.0275069 0.136817 -0.019 -0.00117055 -0.000125295 -0.0302192 0.0395202 -0.194029" qvel="0 -1.5 0 0 0 0 4.9066 0 0 0 0 -3.597 0 0 0.633 0 0 0 0 0 0.175 0 0 0 0 0.175 0 0 0.988 0 0 0 0 0"/>
    <key qpos="0 0 1.0 0.707388 0 0 -0.706825 -0.1652 -0.0279385 -0.041886 0.00247908 0.00101098 0.0888 0.0275069 0.136817 -0.019 -0.00117055 -0.000125295 -0.0302192 0.0395202 -0.194029 -0.2326 -0.0279385 -0.041886 0.00247908 0.0010198 1.227 0.0275069 0.136817 0.1672 -0.00117055 -0.000125295 -0.0302192 0.0395202 -0.194029" qvel="0 -1.5 0 0 0 0 -0.576 0 0 0 0 0.175 0 0 0.988 0 0 0 0 0 4.9066 0 0 0 0 -3.597 0 0 0.633 0 0 0 0 0"/>
  </keyframe>

  <sensor>
    <touch name="r_foot" site="foot_touch_r"/>
    <touch name="r_toes" site="toes_touch_r"/>
    <touch name="l_foot" site="foot_touch_l"/>
    <touch name="l_toes" site="toes_touch_l"/>
  </sensor>
</mujoco>
"""
    return mujoco.MjSpec.from_file(str(_write_cached_xml(xml_text, "myolegs_torso")))


def compute_init_qpos(model: mujoco.MjModel, *, normalize_act: bool = True) -> np.ndarray:
    """Compute canonical MyoSuite ``_init_qpos`` (mid-range for joint-driven linear joints)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    init_qpos = data.qpos.ravel().copy()
    if normalize_act and model.nu > 0:
        actuated_jnt_ids = model.actuator_trnid[
            model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT, 0
        ]
        linear_mask = np.logical_or(
            model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE,
            model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE,
        )
        linear_act_ids = np.intersect1d(actuated_jnt_ids, np.where(linear_mask)[0])
        if linear_act_ids.size > 0:
            qpos_ids = model.jnt_qposadr[linear_act_ids]
            init_qpos[qpos_ids] = np.mean(model.jnt_range[linear_act_ids], axis=1)
    return init_qpos


def reachable_target_points(
    model: mujoco.MjModel,
    site_ids: Sequence[int],
    low: np.ndarray,
    high: np.ndarray,
    n_samples: int = 100_000,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic table of workspace-reachable site positions inside target boxes."""
    ids = tuple(int(i) for i in site_ids)
    low_arr = np.asarray(low, dtype=float).reshape(len(ids), 3)
    high_arr = np.asarray(high, dtype=float).reshape(len(ids), 3)
    cache_key = (
        model.jnt_range.tobytes(),
        model.qpos0.tobytes(),
        model.body_pos.tobytes(),
        model.body_quat.tobytes(),
        model.site_pos[list(ids)].tobytes(),
        ids,
        low_arr.tobytes(),
        high_arr.tobytes(),
        n_samples,
        seed,
    )
    if cache_key not in _REACH_WORKSPACE_CACHE:
        rng = np.random.default_rng(seed)
        qpos = np.tile(model.qpos0, (n_samples, 1))
        joint_types = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        for j in range(model.njnt):
            if model.jnt_limited[j] and int(model.jnt_type[j]) in joint_types:
                lo_j, hi_j = model.jnt_range[j]
                qpos[:, model.jnt_qposadr[j]] = rng.uniform(lo_j, hi_j, n_samples)
        data = mujoco.MjData(model)
        positions = np.empty((n_samples, len(ids), 3), dtype=np.float64)
        for idx in range(n_samples):
            data.qpos[:] = qpos[idx]
            mujoco.mj_kinematics(model, data)
            positions[idx] = data.site_xpos[list(ids)]
        inside = np.all((positions >= low_arr) & (positions <= high_arr), axis=(1, 2))
        if not inside.any():
            raise ValueError("No joint configuration reaches the target box.")
        _REACH_WORKSPACE_CACHE[cache_key] = positions[inside]
    return _REACH_WORKSPACE_CACHE[cache_key]
