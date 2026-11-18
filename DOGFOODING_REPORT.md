# Dogfooding Report: `reaf-sim` + `mujoco_playground` (PyTorch / MuJoCo 4) on MyoSuite Musculoskeletal Environments

**Author:** Mark Colley (`M-Colley`)
**Repository:** [`https://github.com/M-Colley/myosuite-reaf`](https://github.com/M-Colley/myosuite-reaf)
**Target Upstream Repositories:**
- [`google-deepmind/reaf-sim`](https://github.com/google-deepmind/reaf-sim) (`reaf-sim==1.0.1`, `reaf==1.1.2`, `gdm-robotics==1.0.2`)
- [`google-deepmind/mujoco_playground`](https://github.com/google-deepmind/mujoco_playground) (`playground==1.0.1`)
- [`google-deepmind/mujoco`](https://github.com/google-deepmind/mujoco) & [`google-deepmind/mujoco_warp`](https://github.com/google-deepmind/mujoco_warp) (`mujoco==4.0.1`, `mujoco-warp==4.0.1`)
- [`MyoHub/myo_sim`](https://github.com/MyoHub/myo_sim) (`myo-sim==0.2.5`)

---

## 1. Executive Summary

To evaluate Google DeepMind's rewritten PyTorch `mujoco_playground` and `reaf-sim` stack on real biomechanical workloads—and to assess its viability as a future MuJoCo 4 backend for [MyoSuite](https://github.com/MyoHub/myosuite)—we built **`myosuite-reaf`**, a standalone suite of **21 base musculoskeletal & motor tasks (59 registered environments including Sarcopenia, 3CC-r Muscle Fatigue, and Tendon-Transfer Reafferentation variants)** spanning:

- **1-DoF 6-muscle & hybrid muscle+exoskeleton elbow pose tracking** (`myoElbowPose1D6M*`, `myoElbowPose1D6MExo*`) with per-world payload mass & geometry domain randomization (`carry_weight`).
- **4-DoF torque-motor and 5-muscle index finger pose & 3D reach** (`motorFinger*`, `myoFinger*`).
- **23-DoF 39-muscle dexterous hand ASL pose & 5-fingertip 3D reach** (`myoHandPose*`, `myoHandReach*`) with polynomial `JointCouplings` and `EIP -> EPL` reafferentation.
- **18-DoF 50-muscle full-arm fingertip reaching** (`myoArmReach*`) with runtime `MjSpec` digit immobilization and zero-moment tendon pruning.
- **18-DoF 216-muscle lumbar/thoracic spine** (`myoTorsoPoseFixed-v0`) and **passive-exosuit spine** (`myoTorsoExoPoseFixed-v0`, containing **2 unnamed `<freejoint/>`s and 2 `<weld>` equality constraints**).
- **34-DoF 80-muscle floating-base bipedal locomotion** (`myoLegWalk-v0`) with cylinder/sphere spatial tendon wrapping and cyclic gait phase rewards.

### High-Level Verdict

1. **`reaf-sim`'s core architecture is a standout fit for musculoskeletal RL.** Writing a task **once** in PyTorch and executing the exact same `DeviceLayer` + `TaskLayer` graph across single-world CPU C++ (`Impl.CPP`) and 4,096-world GPU MuJoCo Warp (`Impl.WARP`) eliminates the entire "dual-codebase twin maintenance" burden currently required by `MyoSuite`'s NumPy CPU + PyTorch `mjlab` split. Even better, `reaf-sim` natively simulates multi-freejoint passive-exosuit models (`myoTorsoExoPoseFixed-v0`) and automatically tracks per-world `SetConstLevel` (`FIXED`, `KIN0`, `INERT0`) updates and masked resets during runtime domain randomization.
2. **Hill-type muscle biomechanics exposed 14 concrete upstream bugs and API gaps** across `reaf-sim`, `mujoco_playground`, `mujoco` 4.0.1, and `myo-sim`. None are architectural blockers—we worked around all of them cleanly in `myosuite-reaf` (~1,800 LoC) and reproduced every single upstream issue with automated regression assertions in [`tests/test_upstream_reaf_findings.py`](tests/test_upstream_reaf_findings.py). Several findings (notably **Findings 4, 5, 7, 9, and 12**) affect general robot learning tasks well beyond biomechanics and make ideal high-impact upstream PRs.
3. **Performance & numerical parity are production-grade.** On an `NVIDIA GeForce RTX 4070 Ti SUPER` (16 GiB), `myosuite-reaf` achieves **`1.9e-7` to `5.5e-5`** CPU-vs-Warp trajectory parity over 100 physics substeps, scales to **609,097 env steps/s (6.09M physics substeps/s, 437x over CPU)** on `myoElbow` and **93,550 env steps/s (935k physics substeps/s, 160x over CPU)** on 80-muscle `myoLegWalk-v0`, and trains working policies end-to-end with `mujoco_playground`'s `RslRlVecEnvWrapper` + `rsl-rl-lib` PPO.

### Test Environment Matrix

| Component | Exact Version Verified |
| :--- | :--- |
| Hardware | `NVIDIA GeForce RTX 4070 Ti SUPER` (16 GiB VRAM, `sm_89`), Windows 11 x86_64 |
| Python & Package Manager | `Python 3.13.7`, `uv 0.13.8` |
| Simulation & Warp | `mujoco==4.0.1`, `mujoco-warp==4.0.1`, `warp-lang==1.18.0` |
| DeepMind Authoring Stack | `reaf-sim==1.0.1`, `reaf==1.1.2`, `playground==1.0.1`, `gdm-robotics==1.0.2` |
| RL & Tensor Stack | `torch==2.13.0+cu132`, `rsl-rl-lib==5.4.2`, `tensordict==0.15.6`, `gymnasium==1.2.3` |
| Biomechanics Assets | `myo-sim==0.2.5` (+ isolated `mujoco==3.17.0` venv for cross-version MJ3 vs MJ4 probe) |

---

## 2. What Worked Exceptionally Well in `reaf-sim` + `mujoco_playground`

Before cataloging rough edges, four design choices in `reaf-sim` solved longstanding pain points we hit in `mjlab` and `MJX`:

1. **True Single-Codebase CPU (`Impl.CPP`) + GPU (`Impl.WARP`) Execution (`TorchWrapper`):**
   - In `MyoSuite` today, every task requires two implementations: a NumPy `MyoGymnasiumEnv` on CPU and a PyTorch `ManagerBasedRlEnvCfg` twin in `mjlab`, guarded by parity tests.
   - In `reaf-sim`, `Simulator` wraps C++ `MjData` on CPU (`nworld=1`, zero-copy `torch.from_numpy`) and `mjw.Data` on GPU (`nworld=N`) behind one PyTorch interface. Every `PoseTaskSpec`, `ReachTaskSpec`, `WalkTaskSpec`, and `ExcitationPipelineProcessor` in `myosuite-reaf` runs unchanged on both backends.
2. **Native Multi-Freejoint + Equality Weld Support (`myoTorsoExoPoseFixed-v0`):**
   - `mjlab` assumes at most a single floating base per entity, forcing passive exosuit parts with `<freejoint/>`s to be rewritten into three orthogonal slide joints + three hinge joints (`6-DoF` demotion).
   - `reaf-sim`'s `Simulator` handles arbitrary `<freejoint/>` topologies (`myoTorsoExoPoseFixed-v0` has 18 spine joints + 2 unnamed exosuit `<freejoint/>`s + 2 `<weld>` constraints, `nq=32, nv=30, nu=216`) natively on both C++ and MuJoCo Warp with **`4.40e-7` rad** CPU-vs-GPU `qpos` parity.
3. **Automatic `SetConstLevel` Tracking & Masked Per-World Model Resets:**
   - In MuJoCo >= 3.8 / 4.0 and MuJoCo Warp, mutating `body_mass` or `geom_size` at reset time without recomputing derived model constants (`mjw.set_const_0`, `mjw.set_const_fixed`) silently leaves inertia and collision bounding spheres stale.
   - `reaf-sim`'s `TorchWrapper` intercepts writes to `sim.model.body_mass` and `sim.model.geom_size`, automatically marks the exact dirty constant group (`INERT0`, `FIXED`, `KIN0`), flushes them before stepping, and restores untouched worlds via `sim.model.restore_defaults(env_mask)` (verified in `test_positive_torch_wrapper_per_world_dr_isolation`).
4. **Clean `DeviceLayer` vs `TaskLayer` Isolation for Physiological Pipelines:**
   - Routing raw neural actions through `ExcitationPipelineProcessor` (sigmoid -> van Beers motor noise -> stateful 3CC-r `TorchFatigueState` -> `EIP -> EPL` reafferentation -> custom `ExcitationStage`) inside `TaskLayer` while reading muscle activations (`act`) from `MuscleActuatorDevice` kept physiological state cleanly encapsulated per episode and per world mask.

---

## 3. Detailed Upstream Findings & Proposed Fixes

Every finding below is backed by an executable test in [`tests/test_upstream_reaf_findings.py`](tests/test_upstream_reaf_findings.py) or [`tests/test_cpu_warp_parity.py`](tests/test_cpu_warp_parity.py).

---

### Finding 1 — `mujoco_playground`: Unconditional POSIX `import fcntl` Crashes Import on Windows
- **Component:** `google-deepmind/mujoco_playground` — `mujoco_playground/_src/sim/asset_loader.py:18`
- **Severity:** High (blocks `import mujoco_playground` completely on Windows)
- **Root Cause:** `asset_loader.py` unconditionally executes `import fcntl` at module import time to lock file cache writes (`fcntl.flock(lock_file, fcntl.LOCK_EX)`). Because `fcntl` is POSIX-only, importing any `mujoco_playground` module raises `ModuleNotFoundError: No module named 'fcntl'` on Windows—even when all simulation assets come from local `MjSpec` objects and `asset_loader` is never invoked.
- **Reproducer:** `tests/test_upstream_reaf_findings.py::test_finding_1_playground_windows_fcntl_shim`
- **Workaround in `myosuite-reaf`:** [`_compat.py`](src/myosuite_reaf/_compat.py) installs a `msvcrt.locking`-backed shim on `sys.modules["fcntl"]` before importing `mujoco_playground`.
- **Upstream PR:** [`google-deepmind/mujoco_playground#391`](https://github.com/google-deepmind/mujoco_playground/pull/391) — replaces the unconditional `fcntl.flock` call with a cross-platform `_file_lock` context manager (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows).

---

### Finding 2 — `myo-sim` + `mujoco` 4: Packaging Cap, Empty-Prefix `attach` Warning, and Legacy OSL Torso Conventions
- **Component:** `MyoHub/myo_sim` (`myo-sim==0.2.5`) & `mujoco==4.0.1`
- **Severity:** Medium (packaging + custom composition ergonomics)
- **Root Causes:**
  1. **Intentional `pyproject.toml` upper bound (`MyoHub/myo_sim@57743dc`):** `myo-sim==0.2.5` intentionally pins `mujoco>=3.4,<4` until spatial-tendon side-site semantics are audited against MuJoCo 4.0 (aligning directly with Finding 14 below). Until `myo-sim` lifts `<4`, installing alongside `reaf-sim>=1.0.1` (`mujoco[sysid]>=4.0.0`) requires `[tool.uv] override-dependencies = ["mujoco[sysid]>=4.0.0"]`.
  2. **MuJoCo 4 `MjSpec.attach` empty-prefix warning:** `myo_sim.load_spec("myoarm_r")` attaches child hand/arm sub-specs with `prefix=""` so muscle and joint names remain unprefixed, triggering MuJoCo 4.0.1 `WARNING: Attaching a spec with an empty prefix may prevent the model from being saved correctly`.
  3. **Legacy OSL torso meshdir & yaw conventions (`MyoHub/myo_sim#131`):** `models/legacy/torso/assets/myotorso_rigid_assets.xml` and `myotorso_rigid_chain.xml` are authored relative to `myolegs_osl.xml` (`meshdir="../../meshes/"` and a compensating `euler="0 0 -1.57"` canceled by a wrapper body in `myolegs_osl.xml`), so standalone `MjSpec` composers outside `myolegs_osl.xml` must set `meshdir` to `models/meshes/` and neutralize the `-1.57` rad yaw (`<frame euler="0 0 1.57">` in [`assets.py`](src/myosuite_reaf/assets.py)).

---

### Finding 3 — `reaf-sim`: `ActuatorDevice` Lacks Muscle States (`act`, `actuator_length`, `actuator_velocity`, `set_act`) & Leaks `float64` on `Impl.CPP`
- **Component:** `google-deepmind/reaf-sim` — `reaf/simulation/torch/_src/common/actuator_device.py`
- **Severity:** High (blocks stateful muscle actuators & observations out of the box)
- **Root Cause:**
  1. `ActuatorDevice` exposes `ctrl`, `actuator_force`, `tendon_length`, `tendon_velocity`, `joint_pos`, `joint_vel`, and `joint_range`, but omits:
     - `act` (`data.act` — muscle activation state, mandatory in every MyoSuite observation vector and needed on reset for zero-initial-jerk seeding),
     - `actuator_length` (`data.actuator_length` — gear-scaled muscle-tendon unit length, distinct from unweighted `tendon_length`),
     - `actuator_velocity` (`data.actuator_velocity`), and
     - a puppet `set_act(env_mask, values)` reset helper analogous to `ArticulatedDevice.set_joint_positions`.
  2. On `Impl.CPP`, `TorchWrapper` wraps C++ `MjData` (`float64`) via zero-copy `torch.from_numpy`, whereas `Impl.WARP` uses `float32`. `substep_measurements_spec()` declares `dtype=np.float32`, yet `get_measurements()` returns raw `float64` views on CPU, causing downstream `FeatureProducer`s to emit `float64` on CPU vs `float32` on GPU unless explicitly `.to(torch.float32)`-cast.
- **Reproducer:** `tests/test_upstream_reaf_findings.py::test_finding_3_and_4_actuator_device_missing_act_and_bound_accessor_bug`
- **Workaround in `myosuite-reaf`:** Implemented [`MuscleActuatorDevice`](src/myosuite_reaf/devices/muscle_actuator_device.py) supporting mixed stateful (`DYNTYPE_MUSCLE`) and stateless (`DYNTYPE_NONE` exo motor) actuators via `mj_model.actuator_actadr` / `actuator_actnum` and casting measurements to `torch.float32`.
- **Upstream PR:** [`google-deepmind/reaf-sim#14`](https://github.com/google-deepmind/reaf-sim/pull/14) — exposes `"act"`, `"actuator_length"`, `"actuator_velocity"`, and `set_act(env_mask, act)` on `ActuatorDevice`.

---

### Finding 4 — `reaf-sim`: `BoundDataAccessor.act` Raises `AttributeError: 'Data' object has no attribute 'actuator_act'`
- **Component:** `google-deepmind/reaf-sim` — `reaf/simulation/torch/_src/core/torch_wrapper.py:845-895`
- **Severity:** Medium (bug in `sim.data.bind(...)`)
- **Root Cause:** When binding `mjsActuator` specs via `bound = sim.data.bind(sim.model, actuator_specs)`, `BoundDataAccessor.__getattr__("act")` prepends the element prefix `"actuator_"` and looks up `data.actuator_act`. In MuJoCo (`MjData` and `mjw.Data`), activations live in `data.act` (`(nworld, na)`), NOT `data.actuator_act`, and `na` need not equal `nu` (e.g., `myoElbowPose1D6MExoRandom-v0` has `nu=7, na=6`; pneumatic/filter actuators can also have `actnum > 1`).
- **Reproducer:** `tests/test_upstream_reaf_findings.py::test_finding_3_and_4_actuator_device_missing_act_and_bound_accessor_bug`
- **Upstream PR:** [`google-deepmind/reaf-sim#13`](https://github.com/google-deepmind/reaf-sim/pull/13) — resolves `act` and `act_dot` spans on `BoundDataAccessor` via `actuator_actadr` and `actuator_actnum`.

---

### Finding 5 — `reaf-sim`: `DeviceLayer._substep()` Omits Post-Step `sim.forward()`, Leaving Derived Quantities Stale by 1 Substep
- **Component:** `google-deepmind/reaf-sim` — `reaf/simulation/torch/_src/core/device_layer.py:357-395`
- **Severity:** High (affects **all** `reaf-sim` tasks reading `site_xpos`, `xpos`, `xmat`, `cvel`, `subtree_com`, `actuator_force`, `tendon_length`, or sensors after `step()`)
- **Root Cause:** In MuJoCo, `mj_step(model, data)` runs `mj_step1` (computing position/velocity-dependent kinematics, `site_xpos`, `cvel`, `actuator_force`, contacts, sensors at $t$) followed by `mj_step2` (integrating `qpos`, `qvel`, and `act` to $t + \Delta t$). It intentionally does **not** recompute position/velocity-dependent fields at $t + \Delta t$ until the *next* step's `mj_step1` or an explicit `mj_forward`.
  - `DeviceLayer.reset()` calls `self._sim.forward()` before collecting measurements, so `reset()` observations reflect $t = 0$ accurately.
  - `DeviceLayer._substep()` calls `self._sim.step()` and immediately calls `device.get_measurements(self._sim)` **without** calling `self._sim.forward()` on `is_last_substep=True`. Consequently, after `env.step()`, `qpos`, `qvel`, and `act` are at $t + N\Delta t$, while `site_xpos`, `cvel`, `subtree_com`, and `actuator_force` are still from $t + (N-1)\Delta t$!
  - Empirically on `myoLegWalk-v0` (`sim_dt=0.002s`), omitting post-step `forward()` leaves `actuator_force` stale by **72.87 N**, `cvel` stale by **0.752 rad/s**, and `site_xpos` stale by **1.27 mm** on both `Impl.CPP` and `Impl.WARP`.
- **Reproducer:** `tests/test_upstream_reaf_findings.py::test_finding_5_substep_derived_kinematics_lag_and_sync_forward_fix`
- **Workaround in `myosuite-reaf`:** Added [`SyncForwardDevice`](src/myosuite_reaf/devices/sync_forward_device.py) as the first device in `DeviceLayer`; its `get_measurements()` calls `sim.forward()` once when `_substep_idx == _n_substeps - 1` (driven by `on_substep_post_step`), reducing post-step `site_xpos` lag from `> 1e-3 m` to `0.0` (`<= 1e-7` float32 ULP).
- **Upstream PR:** [`google-deepmind/reaf-sim#11`](https://github.com/google-deepmind/reaf-sim/pull/11) — calls `self.sim.forward()` on `is_last_substep` right after `self.sim.step()` in `DeviceLayer._substep()`.

---

### Finding 6 — `reaf-sim`: `builders.config.Sim` Defaults `simulation_timestep = 0.005`, Silently Overwriting `0.002s` Muscle MJCFs
- **Component:** `google-deepmind/reaf-sim` — `reaf/simulation/torch/_src/builders/config.py:31, 50`
- **Severity:** Medium (silent numerical instability trap for stiff/muscle MJCFs)
- **Root Cause:** `builders.config.Sim` defines `simulation_timestep: float = 0.005`, and `apply_sim_config(mj_spec, sim_cfg)` unconditionally executes `mj_spec.option.timestep = sim_cfg.simulation_timestep`. Hill-type muscle models (`myo_sim`) have `10 ms` activation time constants (`tau_act = 0.01s`) and stiff nonlinear tendon force-length curves tuned for `timestep="0.002"`; silently coarsening `sim_dt` by `2.5x` to `0.005s` causes Euler integration overshoot unless the author remembers to inspect `mj_spec.option.timestep` beforehand and pass it back into `Sim(simulation_timestep=...)`.
- **Reproducer:** `tests/test_upstream_reaf_findings.py::test_finding_6_sim_config_overwrites_mjcf_muscle_timestep`
- **Proposed Upstream Fix:** Change `simulation_timestep: float | None = None` on `builders.config.Sim` and only overwrite `mj_spec.option.timestep` when `simulation_timestep is not None`.

---

### Finding 7 — `reaf-sim`: `ArticulatedDevice` Drops Unnamed `<freejoint/>`s and Mismatches `joint_range` Shapes on Floating-Base Models
- **Component:** `google-deepmind/reaf-sim` — `reaf/simulation/torch/_src/common/articulated_device.py:95, 139-164`
- **Severity:** High (breaks floating-base models using `joint_range` and any MJCF with unnamed `<freejoint/>`)
- **Root Causes:**
  1. **Unnamed `<freejoint/>`s dropped (`line 95`):** `configure_spec` filters joints via `[j for j in mj_spec.joints if j.name]`. In MuJoCo XML, `<freejoint/>` without a `name=` attribute is idiomatic and legal (`j.name == ""`). On `myoTorsoExoPoseFixed-v0` (`njnt=20, nq=32`: 18 named spine joints + 2 unnamed exosuit `<freejoint/>`s), `ArticulatedDevice._joints` silently retains only `18` joints and builds `home_qpos` of shape `(18,)` instead of `(32,)`.
  2. **`joint_range` spec vs runtime tensor shape mismatch (`lines 139-164`):** When a named `<freejoint name="root"/>` *is* present (e.g. `myoLegWalk-v0`, `njnt=29, nq=35`), `substep_measurements_spec()` declares `joint_range` with shape `(self._nq, 2)` == `(35, 2)`, whereas `get_measurements()` returns `self._bound_joints.range`, which has shape `(nworld, njnt, 2)` == `(nworld, 29, 2)`!
- **Reproducer:** `tests/test_upstream_reaf_findings.py::test_finding_7_articulated_device_unnamed_freejoint_and_joint_range_bugs`
- **Workaround in `myosuite-reaf`:** [`KinematicsDevice`](src/myosuite_reaf/devices/kinematics_device.py) indexes compiled `sim.mj_model` directly (`nq` for `qpos`, `nv` for `qvel`, `njnt` for `jnt_range`), handling named and unnamed freejoints identically.
- **Upstream PR:** [`google-deepmind/reaf-sim#15`](https://github.com/google-deepmind/reaf-sim/pull/15) — retains unnamed joints (`list(mj_spec.joints)`) in `ArticulatedDevice.initialize()` and fixes `joint_range` spec/measurement shapes to `(len(self._joints), 2)` / `(nworld, len(self._joints), 2)`.

---

### Finding 8 — `reaf-sim` & `mujoco_playground`: Float32 Clock Drift in `TimeoutTerminationChecker` & `int()` Truncation in `RslRlVecEnvWrapper`
- **Component:**
  - `reaf-sim`: `reaf/simulation/torch/_src/common/timeout_termination_checker.py:49`
  - `mujoco_playground`: `mujoco_playground/_src/rl/rsl_rl/vecenv_wrapper.py:48`
- **Severity:** Low-Medium (off-by-one episode horizon jitter)
- **Root Causes:**
  1. `TimeoutTerminationChecker` evaluates `time > self._timeout` strictly in `float32`. Because `N * sim_dt` accumulated in `float32` can land `-1 ULP` below `timeout` (e.g. `1.9999999` vs `2.0`), an episode configured for `100` steps (`2.0s` at `dt=0.02s`) can terminate on step `100` or step `101` depending on `sim_dt` binary representation.
  2. `vecenv_wrapper.py:48` computes `max_episode_length = int(episode_length_secs / ctrl_dt)` instead of `round(...)`. For `episode_length_secs=0.58, ctrl_dt=0.02`, IEEE-754 `0.58 / 0.02 == 28.999999999999996`, so `int(...)` truncates to `28` instead of `29`.
- **Reproducer:** `tests/test_upstream_reaf_findings.py::test_finding_8_timeout_float32_boundary_and_vecenv_int_truncation`
- **Upstream PR / Status:** [`google-deepmind/mujoco_playground#390`](https://github.com/google-deepmind/mujoco_playground/pull/390) uses `round(checker.timeout / control_dt)` in `_derive_max_episode_length`; float32 clock accumulation drift in `ClockDevice` / `TimeoutTerminationChecker` was fixed upstream on `reaf-sim` `main` in `b76e81d`.

---

### Finding 9 — `mujoco_playground`: `RslRlVecEnvWrapper.step()` Multi-Key `action_spec` Silent Slice/Shape Bug
- **Component:** `google-deepmind/mujoco_playground` — `mujoco_playground/_src/rl/rsl_rl/vecenv_wrapper.py:87-89, 187-190`
- **Severity:** Medium-High (breaks hybrid multi-device action spaces in RSL-RL)
- **Root Cause:** In `RslRlVecEnvWrapper.__init__`, `self.num_actions` sums `math.prod(spec.shape)` across **all** keys in `env.action_spec()` (e.g., `"muscles"` of size 6 + `"exo_motor"` of size 1 = `7`). However, `RslRlVecEnvWrapper.step(actions)` executes:
  ```python
  action_key = list(self._env.action_spec().keys())[0]
  env_actions = {action_key: clipped_actions}
  ```
  passing the entire `(nworld, 7)` tensor into the first key alone and omitting all remaining action keys!
- **Workaround in `myosuite-reaf`:** Unified hybrid muscle + exoskeleton actuation (`myoElbowPose1D6MExo*`) under a single `"actions"` key in `ExcitationPipelineProcessor` + `MuscleActuatorDevice`, selectively applying muscle sigmoid/noise/fatigue only to the muscle indices (`actuator_Mask`).
- **Upstream PR:** [`google-deepmind/mujoco_playground#390`](https://github.com/google-deepmind/mujoco_playground/pull/390) — splits flat `(nworld, num_actions)` tensors across all `action_spec` keys in both `RslRlVecEnvWrapper.step()` and `play._RslRlPolicy.step()`.

---

### Finding 10 — `reaf-sim`: `FunctionalRewardProvider` Multiplies Rewards by `dt` Unconditionally
- **Component:** `google-deepmind/reaf-sim` — `reaf/simulation/torch/_src/common/functional_reward.py:108-111`
- **Severity:** Low (API ergonomics when porting existing Gymnasium / MyoSuite benchmarks)
- **Root Cause:** `FunctionalRewardProvider.compute_reward()` returns `(weights @ terms) * self._dt` with no `scale_rewards_by_dt: bool = True` configuration flag (whereas `mjlab` provides `scale_rewards_by_dt=False`). While `* dt` makes continuous-time integral returns invariant to control frequency, existing benchmarks (MyoSuite, Gymnasium MuJoCo, Manipulation) define per-step reward magnitudes and published return thresholds without `* dt`.
- **Reproducer:** `tests/test_upstream_reaf_findings.py::test_finding_10_functional_reward_provider_scales_by_dt`
- **Workaround in `myosuite-reaf`:** Pre-scaled task reward weights by `1.0 / ctrl_dt` when constructing `flat_scene.Reward`, preserving exact MyoSuite per-step reward parity (`atol <= 1.9e-6`).
- **Proposed Upstream Fix:** Add `scale_by_dt: bool = True` to `FunctionalRewardProvider` and `flat_scene.Environment`.

---

### Finding 11 — `reaf-sim` & `mujoco_playground`: Windows Packaging Gates, `cu128` Torch Cap, and `/tmp` Log Paths
- **Component:** `reaf-sim/pyproject.toml`, `mujoco_playground/pyproject.toml`, `mujoco_playground/scripts/train.py`
- **Severity:** Low-Medium (cross-platform developer friction)
- **Root Causes:**
  1. Upstream `uv` CUDA groups gate wheels on `sys_platform == 'linux'` and pin `cu128` (`torch<=2.11.0`), whereas MuJoCo 4.0.1 + MuJoCo Warp 4.0.1 + `reaf-sim` + `rsl-rl-lib` run cleanly on Windows 11 x86_64 with `torch==2.13.0+cu132` (`RTX 4070 Ti SUPER`).
  2. `mujoco_playground` training scripts hardcode POSIX `/tmp/...` log roots instead of `Path(tempfile.gettempdir())`.
- **Proposed Upstream Fix:** Use `tempfile.gettempdir()` in `scripts/train.py` and widen Windows CUDA index/platform markers once `Finding 1` (`fcntl`) is patched.

---

### Finding 12 — `reaf-sim` + `rsl-rl-lib`: `Environment._reset()` Crashes with `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`
- **Component:** `google-deepmind/reaf-sim` — `reaf/simulation/torch/_src/core/environment.py:380, 430`
- **Severity:** High (crashes evaluation / post-training `env.reset()` after `rsl-rl` `OnPolicyRunner.learn()`)
- **Root Cause:**
  - `reaf.simulation.torch.Environment.step()` and `reset_with_options()` are decorated with `@torch.no_grad()`.
  - `rsl-rl-lib`'s `OnPolicyRunner.learn()` wraps rollout collection inside `with torch.inference_mode():`. Because `inference_mode` is stronger than `no_grad`, tensors created and stored inside `self._last_timestep.observation[k]` during `learn()` become **PyTorch inference tensors**.
  - When training finishes (`exiting inference_mode`) and user/evaluation code calls `env.reset()` or `vec_env.reset()`, `Environment._reset()` executes:
    ```python
    observations_w_reset[k][:] = torch.where(...)  # environment.py:430
    ```
    where `observations_w_reset` aliases `self._last_timestep.observation` (an inference tensor!), immediately crashing with:
    `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`.
- **Reproducer:** `tests/test_upstream_reaf_findings.py::test_finding_12_inference_mode_reset_inplace_crash`
- **Upstream PR:** [`google-deepmind/reaf-sim#12`](https://github.com/google-deepmind/reaf-sim/pull/12) — guards `TorchTensor.__init__`, `Environment`, `DeviceLayer`, `TaskLayer`, `Simulator`, `Clock`, and `flat_scene.build_*` against `torch.inference_mode()` tensor tainting and replaces in-place `TimeStep` mutations with out-of-place `torch.where`.

---

### Finding 13 — `mujoco-warp` 4.0.1: Inter-World Float32 Non-Associativity Across Identically Seeded Worlds (`~1e-7` to `9.6e-6` rad)
- **Component:** `google-deepmind/mujoco_warp` (`mujoco-warp==4.0.1`)
- **Severity:** Informational (important for parity test tolerances)
- **Observation:** When `nworld=8` GPU worlds are initialized to bit-identical `(qpos, qvel, act, ctrl)` states and stepped for 100 substeps, worlds `0..7` exhibit `0.0` spread on contact-free exo elbow (`myoElbowPose1D6MExoRandom-v0`), `1.19e-7` to `2.74e-6` rad spread on hand/torso/leg models, and up to `9.62e-6` rad spread on 50-muscle `myoArmReachFixed-v0` due to parallel atomic contact buffer allocation (`nacon`) and tiled Cholesky reductions. All 8 GPU worlds remain within `4.23e-6` rad of the float64 C++ CPU reference.
- **Implication for Parity Suites:** GPU inter-world determinism tests on contact-bearing musculoskeletal models should set `atol=2e-5` rather than `atol=0.0` or `1e-7`.

---

### Finding 14 — `mujoco` 3.17.0 -> 4.0.1: Spatial Tendon Cylinder/Sphere Wrapping `actuator_acc0` Shift on `MyoLeg` (`recfem_r` / `recfem_l`)
- **Component:** `google-deepmind/mujoco` (`3.17.0` -> `4.0.1`) + `MyoLeg` (`myolegs_v0.56(mj237).mjb` / XML)
- **Severity:** Critical Migration Insight for MyoSuite's MuJoCo 4 Upgrade
- **Empirical Discovery (`scripts/probe_mj3_vs_mj4.py`):**
  Compiling the **exact same XML** in `mujoco==3.17.0` vs `mujoco==4.0.1` and comparing compiled `MjModel` muscle constants (`actuator_lengthrange`, `actuator_gainprm`, `actuator_biasprm`, `actuator_acc0`) across all task families revealed:
  - `myoElbow` (`6` muscles), `myoFinger` (`5` muscles), `myoHand` (`39` muscles), and `myoArm` (`50` muscles): `lengthrange` and `gainprm` match to `0.0`, and `actuator_acc0` matches within **`0.0` to `8.14e-12`**! Over 200 physics substeps on elbow/hand, MJ 3.17.0 vs MJ 4.0.1 `qpos` matches to **`3.5e-8` – `4.2e-8` rad**.
  - `myoLegWalk-v0` (`80` muscles): `lengthrange` and `gainprm` match to `0.0` across all 80 muscles, and **78 of 80 muscles have bit-identical `actuator_acc0`** to 15 decimal places (e.g. `vasmed_r = 23.256963206312385` in both MJ 3.17.0 and MJ 4.0.1). **However, exactly two muscles—`recfem_r` (idx 28) and `recfem_l` (idx 68), the biarticular rectus femoris muscles routing over hip/knee cylinder/sphere wrap geoms—drop in `actuator_acc0` from `44.9624` (MJ 3.17.0) to `15.6284` (MJ 4.0.1), a `Δ = 29.3335` change!**
- **Why It Matters:** MuJoCo uses `actuator_acc0` to scale muscle force-velocity damping/stiffness regularizers during `mj_setConst`. Meanwhile, within MuJoCo 4.0.1 itself, C++ (`Impl.CPP`) and GPU Warp (`Impl.WARP`) agree on `myoLegWalk-v0` to **`4.06e-5` rad** over 200 substeps! When MyoSuite migrates from MuJoCo 3 to MuJoCo 4, `MyoLeg` parity baselines and published locomotion checkpoints on Hugging Face will need a clean MuJoCo 4 baseline epoch.

---

## 4. Empirical Benchmark Tables

### 4.1 Cross-Version Probe: MuJoCo 3.17.0 (`CPU`) vs MuJoCo 4.0.1 (`reaf-sim CPP` & `reaf-sim WARP`, 20 Steps = 200 Substeps)

Generated via [`scripts/probe_mj3_vs_mj4.py`](scripts/probe_mj3_vs_mj4.py):

| Environment | `acc0 max \|Δ\|` (MJ3 vs MJ4) | `obs0 max \|Δ\|` | MJ3 vs MJ4 C++ `max \|Δqpos\|` | MJ3 vs MJ4 C++ `max \|Δact\|` | MJ3 vs MJ4 C++ `max \|Δrwd\|` | MJ4 C++ vs MJ4 Warp `max \|Δqpos\|` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `myoElbowPose1D6MRandom-v0` | `3.33e-16` | `0.00e+00` | `4.19e-08` | `4.75e-08` | `9.51e-08` | `1.46e-06` |
| `motorFingerReachFixed-v0` | `0.00e+00` | `7.45e-09` | `1.24e-08` | `0.00e+00` | `2.18e-08` | `2.03e-06` |
| `myoHandPoseFixed-v0` | `4.97e-13` | `0.00e+00` | `3.48e-08` | `6.88e-08` | `3.82e-07` | `2.04e-06` |
| `myoFingerPoseRandom-v0` | `0.00e+00` | `0.00e+00` | `1.62e-05` | `9.75e-08` | `1.24e-05` | `4.34e-02` *(1)* |
| `myoArmReachFixed-v0` | `8.14e-12` | `0.00e+00` | `7.33e-03` | `8.83e-08` | `4.84e-04` | `3.56e-06` |
| `myoLegWalk-v0` | **`2.93e+01`** *(2)* | `1.79e-07` | `1.54e-01` | `0.00e+00` | `9.92e-01` | `4.06e-05` |

*(1) `myoFingerPoseRandom-v0` hits hard tendon/joint limit contacts around step 15 under open-loop sinusoidal excitation where float32 vs float64 contact penetration amplifies velocity sensitivity; over the standard 10-step (100-substep) parity window (`tests/test_cpu_warp_parity.py`), `MJ4 C++ vs Warp Δqpos == 6.59e-06`.*
*(2) Driven exclusively by `recfem_r` / `recfem_l` spatial tendon wrapping in MuJoCo 4.0.1 (see Finding 14).*

### 4.2 End-to-End Throughput (`NVIDIA GeForce RTX 4070 Ti SUPER`, `substeps=10`, `SyncForwardDevice=True`)

Generated via [`scripts/benchmark_throughput.py`](scripts/benchmark_throughput.py):

| Environment | `nq / nv / nu` | CPU (`1` env) | Warp (`1`) | Warp (`64`) | Warp (`512`) | Warp (`2048`) | Warp (`4096`) | Substeps/s (`4096`) | Speedup |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `myoElbowPose1D6MRandom-v0` | `1 / 1 / 6` | 1,394.1 | 247.6 | 16,404.2 | 121,216.0 | 404,164.8 | **609,096.5** | **6,090,965** | **436.9x** |
| `myoHandReachRandom-v0` | `23 / 23 / 39` | 914.4 | 199.6 | 10,541.8 | 53,795.8 | 127,534.1 | **163,270.6** | **1,632,706** | **178.6x** |
| `myoArmReachFixed-v0` | `18 / 18 / 50` | 517.0 | 145.9 | 7,532.2 | 33,090.1 | 67,503.3 | **80,906.2** | **809,062** | **156.5x** |
| `myoLegWalk-v0` | `35 / 34 / 80` | 586.5 | 155.7 | 7,966.8 | 35,791.6 | 77,838.5 | **93,549.9** | **935,499** | **159.5x** |

---

## 5. Architectural Comparison: `reaf-sim` vs `mjlab` for MyoSuite (MuJoCo 4 Horizon)

| Dimension | `MyoSuite` Today (`MyoGymnasiumEnv` + `mjlab`) | `myosuite-reaf` (`reaf-sim` + `mujoco_playground`) |
| :--- | :--- | :--- |
| **MuJoCo Version** | MuJoCo 3.x (`<4`) | MuJoCo 4.x (`>=4.0.1`) |
| **Task Authoring Count** | **2 implementations per task** (NumPy CPU class + PyTorch `mjlab` config) | **1 implementation per task** (`TorchWrapper` runs both `Impl.CPP` and `Impl.WARP`) |
| **Multi-Freejoint Exosuits** | Requires rewriting `<freejoint/>` into 6-DoF slide+hinge chains | **Native** (`myoTorsoExoPoseFixed-v0` works unmodified) |
| **Runtime Model Edits / DR** | Manual `refresh_geom_derived_fields` + `mjw.set_const` management | **Automatic** dirty-flag tracking (`FIXED`, `KIN0`, `INERT0`) + `restore_defaults(env_mask)` |
| **Muscle `data.act` Support** | Manual `entity.data.data.act` escape hatch in `mjlab` | Via [`MuscleActuatorDevice`](src/myosuite_reaf/devices/muscle_actuator_device.py) (clean upstream candidate for `ActuatorDevice`) |
| **Post-Step `mj_forward` Sync** | Manual `mdp.sync_forward` in first termination term | Via [`SyncForwardDevice`](src/myosuite_reaf/devices/sync_forward_device.py) (clean upstream candidate for `DeviceLayer._substep`) |
| **Sim-to-Real / Hardware Bus** | Simulation-only | Built-in `DeviceLayer` abstraction designed for zero-shot exoskeleton / prosthetic hardware swap |

---

## 6. Submitted Upstream PRs (`google-deepmind/reaf-sim`, `google-deepmind/mujoco_playground`, & `google-deepmind/tree`)

Ten focused, single-concern upstream PRs have been opened:

| PR | Finding(s) / Scope | Summary |
| :--- | :--- | :--- |
| [`reaf-sim#11`](https://github.com/google-deepmind/reaf-sim/pull/11) | Finding 5 | Refresh derived simulator quantities (`self.sim.forward()`) after the final physics substep in `DeviceLayer._substep()`. |
| [`reaf-sim#12`](https://github.com/google-deepmind/reaf-sim/pull/12) | Finding 12 | Prevent `torch.inference_mode()` tensor tainting across `Environment`, `DeviceLayer`, `TaskLayer`, `Simulator`, `Clock`, and `TorchTensor`, and avoid in-place `TimeStep` mutation. |
| [`reaf-sim#13`](https://github.com/google-deepmind/reaf-sim/pull/13) | Finding 4 | Support zero-host-sync `act` and `act_dot` spans on `BoundDataAccessor` (`sim.data.bind(sim.model, actuator_specs)`) via `actuator_actadr` / `actuator_actnum`. |
| [`reaf-sim#14`](https://github.com/google-deepmind/reaf-sim/pull/14) | Finding 3 | Expose `"act"`, `"actuator_length"`, `"actuator_velocity"` measurements and `set_act(env_mask, act)` on `ActuatorDevice`. |
| [`reaf-sim#15`](https://github.com/google-deepmind/reaf-sim/pull/15) | Finding 7 | Retain unnamed joints (`<freejoint/>`) in `ArticulatedDevice.initialize()` and fix `joint_range` spec/measurement shapes (`(num_joints, 2)` / `(nworld, num_joints, 2)`). |
| [`reaf-sim#17`](https://github.com/google-deepmind/reaf-sim/pull/17) | Packaging (Py 3.15) | Widen `requires-python = ">=3.12,<3.16"` and raise the PyTorch upper bound to `torch<2.16` for Python 3.15 (`cp315`) wheels. |
| [`mujoco_playground#390`](https://github.com/google-deepmind/mujoco_playground/pull/390) | Findings 8 & 9 | Split flat RSL-RL actions across multi-key `action_spec` dicts in `RslRlVecEnvWrapper` and `play._RslRlPolicy`, and round `checker.timeout / control_dt`. |
| [`mujoco_playground#391`](https://github.com/google-deepmind/mujoco_playground/pull/391) | Finding 1 | Replace unconditional POSIX `fcntl.flock` in `asset_loader.py` with a cross-platform `_file_lock` context manager (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows). |
| [`mujoco_playground#393`](https://github.com/google-deepmind/mujoco_playground/pull/393) | Packaging (Py 3.14–3.15) | Widen `requires-python = ">=3.12,<3.16"` for simulation and `rl` extras while gating `<3.14` real-robot dependencies and enabling `torch<2.16` on Python 3.15. |
| [`tree#162`](https://github.com/google-deepmind/tree/pull/162) | Packaging (`dm-tree` Py 3.15) | Build and ship `cp315` and free-threaded `cp315t` wheels for `dm-tree` across Linux (`x86_64`, `aarch64`), macOS (`universal2`), and Windows (`amd64`). |
