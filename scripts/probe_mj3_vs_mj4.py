# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-version MuJoCo 3 (``myosuite``) vs MuJoCo 4 (``myosuite-reaf``) dynamics probe.

Executes identical initial states and deterministic action rollouts in a separate
MuJoCo 3 Python environment (via subprocess JSON IPC) and compares static model
constants (``actuator_lengthrange``, ``actuator_gainprm``, ``actuator_biasprm``,
``actuator_acc0``) and 20-step trajectories (``qpos``, ``qvel``, ``act``,
``actuator_force``, ``obs``, ``reward``) against MuJoCo 4 CPU (``impl="cpp"``) and
MuJoCo Warp 4 GPU (``impl="warp"``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from myosuite_reaf import EnvConfig, make_env

PROBE_TASKS: tuple[str, ...] = (
    "myoElbowPose1D6MRandom-v0",
    "myoFingerPoseRandom-v0",
    "motorFingerReachFixed-v0",
    "myoHandPoseFixed-v0",
    "myoArmReachFixed-v0",
    "myoLegWalk-v0",
)

_MJ3_WORKER_CODE = r"""
import json
import sys
import mujoco
import numpy as np
import myosuite

payload = json.loads(sys.stdin.read())
env_id = payload["env_id"]
actions = np.asarray(payload["actions"], dtype=np.float32)

env = myosuite.make_env(env_id)
obs0, _ = env.reset(seed=0)
u = env.unwrapped
model = u.model
data = u.data

# Capture exact post-reset initial state so MJ4 uses the exact same state & targets
state = {
    "mj_version": mujoco.__version__,
    "nq": int(model.nq),
    "nv": int(model.nv),
    "nu": int(model.nu),
    "na": int(model.na),
    "lengthrange": model.actuator_lengthrange.tolist(),
    "gainprm": model.actuator_gainprm[:, :4].tolist(),
    "biasprm": model.actuator_biasprm[:, :4].tolist(),
    "acc0": model.actuator_acc0.tolist(),
    "init_qpos": data.qpos.copy().tolist(),
    "init_qvel": data.qvel.copy().tolist(),
    "site_pos": model.site_pos.copy().tolist(),
    "target_jnt_value": getattr(u, "target_jnt_value", np.array([])).tolist()
        if hasattr(u, "target_jnt_value") and getattr(u, "target_jnt_value") is not None else None,
    "obs0": obs0.tolist(),
}

traj_qpos, traj_qvel, traj_act, traj_frc, traj_obs, traj_rwd = [], [], [], [], [], []
for step_idx in range(actions.shape[0]):
    obs, rwd, term, trunc, info = env.step(actions[step_idx])
    traj_qpos.append(data.qpos.copy().tolist())
    traj_qvel.append(data.qvel.copy().tolist())
    traj_act.append(data.act.copy().tolist() if model.na > 0 else [])
    traj_frc.append(data.actuator_force.copy().tolist())
    traj_obs.append(obs.tolist())
    traj_rwd.append(float(rwd))

env.close()
state["traj"] = {
    "qpos": traj_qpos,
    "qvel": traj_qvel,
    "act": traj_act,
    "actuator_force": traj_frc,
    "obs": traj_obs,
    "reward": traj_rwd,
}
print(json.dumps(state))
"""


def _run_mj3_worker(
    mj3_python: Path,
    env_id: str,
    actions: np.ndarray,
    mj3_pythonpath: str | None = None,
) -> dict[str, Any]:
    env_vars = os.environ.copy()
    env_vars["PYTHONDONTWRITEBYTECODE"] = "1"
    if mj3_pythonpath:
        env_vars["PYTHONPATH"] = mj3_pythonpath
    proc = subprocess.run(
        [str(mj3_python), "-B", "-c", _MJ3_WORKER_CODE],
        input=json.dumps({"env_id": env_id, "actions": actions.tolist()}),
        text=True,
        capture_output=True,
        check=True,
        env=env_vars,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    return json.loads(lines[-1])


def _run_mj4_rollout(
    env_id: str,
    *,
    impl: str,
    mj3_ref: dict[str, Any],
    actions: np.ndarray,
) -> dict[str, Any]:
    device = "cuda:0" if (impl == "warp" and torch.cuda.is_available()) else "cpu"
    env = make_env(env_id, config=EnvConfig(impl=impl, num_envs=1, device=device, seed=0))  # type: ignore[arg-type]

    override: dict[str, Any] = {
        "qpos": np.asarray(mj3_ref["init_qpos"], dtype=np.float32),
        "qvel": np.asarray(mj3_ref["init_qvel"], dtype=np.float32),
    }
    if mj3_ref.get("target_jnt_value") is not None:
        override["target_angles"] = np.asarray(mj3_ref["target_jnt_value"], dtype=np.float32)

    # For reach tasks, copy exact target site positions from MJ3
    if "Reach" in env_id:
        mj_model = env.mj_model
        mj3_site_pos = np.asarray(mj3_ref["site_pos"], dtype=np.float32)
        target_sids = [
            s for s in range(mj_model.nsite) if mj_model.site(s).name.endswith("_target")
        ]
        if target_sids:
            override["target_pos"] = mj3_site_pos[target_sids]

    obs0, _ = env.reset(seed=0, options={"reset_override": override})
    sim = env.reaf_env.device_layer.sim
    mj_model = sim.mj_model

    traj_qpos, traj_qvel, traj_act, traj_frc, traj_obs, traj_rwd = [], [], [], [], [], []
    for step_idx in range(actions.shape[0]):
        obs, rwd, _, _, _ = env.step(actions[step_idx])
        traj_qpos.append(sim.data.qpos[0].detach().cpu().numpy().tolist())
        traj_qvel.append(sim.data.qvel[0].detach().cpu().numpy().tolist())
        traj_act.append(sim.data.act[0].detach().cpu().numpy().tolist() if mj_model.na > 0 else [])
        traj_frc.append(sim.data.actuator_force[0].detach().cpu().numpy().tolist())
        traj_obs.append(obs.tolist())
        traj_rwd.append(float(rwd))

    out = {
        "lengthrange": mj_model.actuator_lengthrange.tolist(),
        "gainprm": mj_model.actuator_gainprm[:, :4].tolist(),
        "biasprm": mj_model.actuator_biasprm[:, :4].tolist(),
        "acc0": mj_model.actuator_acc0.tolist(),
        "obs0": obs0.tolist(),
        "traj": {
            "qpos": traj_qpos,
            "qvel": traj_qvel,
            "act": traj_act,
            "actuator_force": traj_frc,
            "obs": traj_obs,
            "reward": traj_rwd,
        },
    }
    env.close()
    return out


def _max_abs(a: Any, b: Any) -> float:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.size == 0 and arr_b.size == 0:
        return 0.0
    return float(np.max(np.abs(arr_a - arr_b)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MuJoCo 3 vs MuJoCo 4 musculoskeletal dynamics."
    )
    parser.add_argument(
        "--mj3-python",
        type=Path,
        required=True,
        help="Path to Python interpreter with myosuite + MuJoCo 3 installed.",
    )
    parser.add_argument(
        "--mj3-pythonpath",
        type=str,
        default=None,
        help="Optional PYTHONPATH to locate myosuite without editable install.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    import mujoco

    rng = np.random.default_rng(12345)
    summary: list[dict[str, Any]] = []

    print(
        f"# Cross-Version Probe: MuJoCo 3 (`myosuite`) vs MuJoCo {mujoco.__version__} (`myosuite-reaf`)\n"
    )
    print(
        "| Task | MJ3->MJ4 `acc0` Δ | MJ3->MJ4 `qpos` Max Δ | MJ3->MJ4 `act` Max Δ | MJ3->MJ4 `obs` Max Δ | MJ3->MJ4 `rwd` Max Δ | MJ4 CPU->Warp `qpos` Max Δ |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")

    for env_id in PROBE_TASKS:
        probe_env = make_env(env_id, seed=0)
        lo = probe_env.action_space.low
        hi = probe_env.action_space.high
        actions = rng.uniform(lo, hi, size=(args.steps, lo.shape[0])).astype(np.float32)
        probe_env.close()

        mj3 = _run_mj3_worker(args.mj3_python, env_id, actions, mj3_pythonpath=args.mj3_pythonpath)
        mj4_cpp = _run_mj4_rollout(env_id, impl="cpp", mj3_ref=mj3, actions=actions)
        mj4_warp = _run_mj4_rollout(env_id, impl="warp", mj3_ref=mj3, actions=actions)

        row = {
            "env_id": env_id,
            "mj3_version": mj3["mj_version"],
            "mj4_version": mujoco.__version__,
            "lengthrange_max_diff": _max_abs(mj3["lengthrange"], mj4_cpp["lengthrange"]),
            "gainprm_max_diff": _max_abs(mj3["gainprm"], mj4_cpp["gainprm"]),
            "acc0_max_diff": _max_abs(mj3["acc0"], mj4_cpp["acc0"]),
            "obs0_max_diff": _max_abs(mj3["obs0"], mj4_cpp["obs0"]),
            "mj3_vs_mj4_cpp": {
                k: _max_abs(mj3["traj"][k], mj4_cpp["traj"][k])
                for k in ("qpos", "qvel", "act", "actuator_force", "obs", "reward")
            },
            "mj4_cpp_vs_warp": {
                k: _max_abs(mj4_cpp["traj"][k], mj4_warp["traj"][k])
                for k in ("qpos", "qvel", "act", "actuator_force", "obs", "reward")
            },
        }
        summary.append(row)
        d34 = row["mj3_vs_mj4_cpp"]
        d4w = row["mj4_cpp_vs_warp"]
        print(
            f"| `{env_id}` | {row['acc0_max_diff']:.2e} | {d34['qpos']:.2e} | "
            f"{d34['act']:.2e} | {d34['obs']:.2e} | {d34['reward']:.2e} | {d4w['qpos']:.2e} |"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
