# Learning Cortex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a predictive-coding hierarchy that learns only with local rules. Show it as twin mirrors (Newborn and Adult), and measure the phenomena that emerge through three phases.

**Architecture:** A new package `cortical_mirror/cortex/` contains the following parts.
- A fixed retina.
- Levels that settle by error-driven inference and learn by local Hebbian, Kolen-Pollack and delta rules.
- A temporal predictor per level.
- A display adapter.
- A `PredictiveEngine` wrapper.
- A phenomena library.

It plugs into the existing server through an `extra_views` hook.

**Tech Stack:** Python 3.10+, PyTorch tensor ops with autograd never used, OpenCV, and the existing stdlib HTTP server.

**Spec:** `docs/superpowers/specs/2026-10-27-learning-cortex-design.md`

## Global Constraints

- No autograd in `cortical_mirror/cortex/`. All tensors use `requires_grad=False`, and all updates are explicit.
- Feedback weights B never copy or read W. They start mismatched and receive the same local update.
- No pretrained weights or models anywhere in the cortex track.
- Retina is 128×96, luminance only. ON/OFF has 2 channels; with lag (phase 2+) it has 4.
- V1 has 32 cells, kernel 8, stride 2, padding 3.
- The inference step is 0.9 / λ_max, estimated by a 20-iteration power iteration every 100 frames.
- Homeostasis: θ ← θ·exp(0.002·clip(rate/target − 1, −1, 1)), with target 0.05 and θ ∈ [λ/10, 10λ]. Activities are clamped to [0, 1].
- Training uses flank GPUs 0, 1 and 3 only, never GPU 2.
- Commits are small and on side branches, merged by PR. No AI attribution anywhere.
- Every protocol is written in `docs/protocols/` before its first measurement.

## Review Focus

- Camera frames of any size or aspect ratio are resized to 128×96, and frames that are not 3-channel are converted. Test: `test_retina_accepts_any_frame_size`.
- A uniform or completely dark scene gives no NaN and no divide-by-zero. Test: `test_retina_uniform_frame_is_silent`.
- Hours of continuous input keep rates bounded and weights finite. Test: `test_hierarchy_long_run_stays_bounded` (20,000 frames, synthetic).
- Loading a checkpoint from a different architecture (phase 1 into a phase 2 network) fails with a clear message, not a shape error. Test: `test_checkpoint_architecture_mismatch_is_reported`.
- Resetting the newborn mid-stream leaves the adult untouched and the server running. Test: `test_twin_reset_affects_newborn_only`.

---

## Phase 1

### Task 1: Retina

**Files:**
- Create: `cortical_mirror/cortex/__init__.py`, `cortical_mirror/cortex/retina.py`
- Test: `tests/test_cortex_retina.py`

**Interfaces:**
- Produces: `class Retina(width=128, height=96, lagged=False, device="cpu")` with:
  - `reset()`
  - `encode(frames: Tensor[B,1,H,W] in [0,1]) -> Tensor[B,C,H,W]`, where C is 2, or 4 if lagged; channels are ON, OFF, then lagged ON, lagged OFF
  - `channels: int`
- Produces: `frame_to_gray(frame_bgr: np.ndarray, width, height, device) -> Tensor[1,1,H,W]`

- [ ] Tests:
  - `test_retina_uniform_frame_is_silent`: output is all < 1e-6 and finite.
  - `test_bright_spot_drives_on_centre`: ON at the spot > 0.5 and OFF = 0.
  - `test_dark_spot_drives_off_centre`
  - `test_lagged_channels_repeat_previous_frame`
  - `test_retina_accepts_any_frame_size`: 640×480 and 1920×1080 BGR, and grayscale 2-D input.
- [ ] Implement with a DoG (σ 1 and 2, 9×9) and normalisation by a 15×15 mean of |DoG| + 0.02, clipped at ±3 then divided by 3.
- [ ] Commit `feat(cortex): add the retina front end`.

### Task 2: Local-learning hierarchy (V1 with temporal prediction)

**Files:**
- Create: `cortical_mirror/cortex/hierarchy.py`
- Test: `tests/test_cortex_hierarchy.py`

**Interfaces:**
- Produces: `@dataclass LevelConfig(cells=32, kernel=8, stride=2, padding=3, sparsity=0.05, learning_rate=10.0, target_rate=0.05, temporal_kernel=9, temporal_rate=0.05)`
- Produces: `class Hierarchy(in_channels: int, levels: list[LevelConfig], device, seed=0)` with:
  - `step(code: Tensor[B,C,H,W], learn: bool = True) -> StepResult`. It settles, learns, and returns the activities, errors and the next-code prediction.
  - `predict_next_code() -> Tensor[B,C,H,W]`
  - `reset_state()` for activities only; `reset_learning(seed)` for weights back to the unlearned state.
  - `age_frames: int`
  - `state_dict() -> dict` and `load_state_dict(d)`. The dict holds weights, thresholds, age and an `architecture` dict.
- Produces: `@dataclass StepResult(activities: list[Tensor], errors: list[Tensor], code_error: float, persistence_error: float, predicted_next: Tensor)`. `code_error` is the mean squared error of the previous frame's prediction for this frame. `persistence_error` is the mean squared difference between this code and the previous code.

- [ ] Tests:
  - `test_cortex_never_uses_autograd`: monkeypatch `torch.Tensor.backward` to raise, run 50 steps, and assert no state tensor has `requires_grad` or a `grad_fn`.
  - `test_feedback_weights_converge_without_copying`: B starts mismatched; after 2,000 frames, ‖B−W‖/‖W‖ < 0.1.
  - `test_hierarchy_long_run_stays_bounded`: 20,000 frames of 1/f texture; every cell's rate is in [0.005, 0.5] and everything is finite.
  - `test_temporal_predictor_beats_persistence_on_a_pan`: a 1/f texture panning 2 px per frame for 3,000 frames; code_error < 0.8 × persistence_error over the last 200.
  - `test_checkpoint_architecture_mismatch_is_reported`
- [ ] Implement exactly the spike-proven updates from the spec. The temporal predictor V is a conv `cells→cells` (kernel 9, padding 4), predicting the next activity as clamp(V*r, 0, 1). The delta rule is ΔV = temporal_rate · corr(r_next − r̂_next, r_prev), averaged over positions. The next code is W rendering of the predicted V1 activity.
- [ ] Commit `feat(cortex): add the local-learning hierarchy`.

### Task 3: Engine and display adapter

**Files:**
- Create: `cortical_mirror/cortex/engine.py`
- Test: `tests/test_cortex_engine.py`

**Interfaces:**
- Consumes: `Retina`, `Hierarchy`.
- Produces: `class CortexEngine(device="cpu", learn=True, width=256, height=192, phase: int = 1)`, implementing `PredictiveEngine`:
  - `predict(frame_bgr) -> EnginePrediction`. `observed_error` is in code space.
  - `reset()` for state only; `reset_learning()` makes it a newborn again.
  - `age_frames`
  - `last_code_error`, `last_persistence_error`
  - `receptive_fields() -> np.ndarray` (BGR grid image)
  - `save(path)`; `@classmethod load(path, device, learn)`
- Produces: `render_prediction(code_next: Tensor, frame_bgr, width, height) -> np.ndarray`. This is the display adapter: the low-passed luminance of the current frame (Gaussian σ 3) plus the predicted ON−OFF contrast × 0.25, as a gray BGR image.

- [ ] Tests:
  - `test_engine_predicts_a_frame_of_the_mirror_size`
  - `test_learning_engine_ages_and_frozen_engine_does_not`
  - `test_reset_learning_restores_newborn_weights`
  - `test_save_and_load_round_trip` (in `tmp_path`)
- [ ] Commit `feat(cortex): wrap the hierarchy as a mirror engine`.

### Task 4: Phenomena library and phase 1 protocols

**Files:**
- Create: `docs/protocols/01-orientation-tuning.md`, `docs/protocols/02-oddball-adaptation.md`, `cortical_mirror/cortex/phenomena.py`, `tools/phenomena.py`
- Test: `tests/test_cortex_phenomena.py`

**Interfaces:**
- Produces: `orientation_tuning(hierarchy, period=8.0, orientations=12, phases=8) -> dict` with keys `osi` (list), `median_osi`, `fraction_selective` (OSI > 0.3), `curves`, and `responsive_cells`. The network is frozen during measurement, and the state is restored after.
- Produces: `oddball_adaptation(hierarchy, standard_deg=0, deviant_deg=90, trials=200, p_deviant=0.1, frames_per_stimulus=4) -> dict` with keys:
  - `deviant_response`, `control_response`, `standard_response` (mean V1 error energy at stimulus onset);
  - `ssa_index = (deviant − control) / (deviant + control)`.

  The control is the many-standards sequence: 10 orientations, equiprobable. Temporal learning stays on during the sequence, because adaptation is learning; weights are restored after.
- Produces CLI: `tools/phenomena.py --checkpoint PATH|--untrained --phase N --output reports/cortex/<name>.json`

- [ ] Write both protocols first: stimulus, measure, controls, and the pre-declared pass criteria.
  - Orientation: the median OSI of the learned network exceeds the untrained one by ≥ 0.2, and the fraction selective is ≥ 0.5.
  - SSA: `ssa_index` > 0, and higher than in the untrained network.
- [ ] Tests:
  - `test_oriented_bar_cells_have_high_osi`: a hand-built test fixture whose W kernels are oriented bars gives median OSI > 0.5.
  - `test_random_cells_have_low_osi` (< 0.4)
  - `test_ssa_index_math`
- [ ] Commit `feat(cortex): add phase 1 phenomena and protocols`.

### Task 5: Adult training tool

**Files:**
- Create: `tools/train_cortex.py`
- Test: `tests/test_cortex_training.py`

**Interfaces:**
- Consumes: `split_videos`, `Retina`, `Hierarchy`, `CortexEngine.save`.
- Produces CLI: `tools/train_cortex.py --data UCF --phase N --streams 16 --hours H --output checkpoints/cortex-adult-pN.pt --log reports/cortex/train-pN.jsonl`. It streams consecutive frames from training-split videos, one video per stream, and moves to a new random video at the end. Every 10,000 frames it logs the reconstruction error, rates and median OSI.

- [ ] Test `test_training_smoke_writes_loadable_checkpoint`: two synthetic AVIs in `tmp_path`, 200 frames.
- [ ] Commit `feat(cortex): train an adult with the same local rules`.

### Task 6: Twin mirrors

**Files:**
- Create: `cortical_mirror/cortex/twin.py`
- Modify: `cortical_mirror/server.py` (generic `extra_views()` / `extra_metrics()` hooks), `cortical_mirror/cli.py` (a `twins` command), `cortical_mirror/static/index.html` (twins view)
- Test: `tests/test_cortex_twin.py`

**Interfaces:**
- Produces: `class TwinEngine(newborn: CortexEngine, adult: CortexEngine)`, implementing `PredictiveEngine`. Its prediction is the newborn's. It adds `extra_views() -> {"newborn", "adult", "newborn_cells", "adult_cells"}`, `extra_metrics() -> {"newborn": {...}, "adult": {...}}` (age, code error and persistence error, smoothed), and `reset_newborn()`.
- The server's `control({"reset_newborn": true})` calls `reset_newborn()`.
- CLI: `cortical-mirror twins --adult PATH --source 0`
- Page: `?view=twins`. It's the default when `metrics.mode == "twins"`. Two mirrors, each with its cells grid, its age, and a surprise curve vs persistence. Key N resets the newborn.

- [ ] Tests:
  - `test_twin_reset_affects_newborn_only`
  - `test_server_publishes_extra_views`
- [ ] Commit in two parts (engine with hooks, then page).

### Task 7: Phase 1 measurements, docs, PR

- [ ] Train the phase 1 adult on flank GPU 0 for 3 h.
- [ ] Run the protocols on: untrained; newborn after the sample clip looped to 2 min; newborn after 2 min of a held-out UCF test scene; and the adult. Write `reports/cortex/phase1-*.json`.
- [ ] README section "A cortex that learns to see": rules, results against the pre-declared criteria (including failures), and how to run the twins.
- [ ] Open a PR and merge it.

## Phase 2

### Task 8: Lagged channels and direction selectivity

**Files:**
- Modify: `retina.py` (lagged already exists; enable it for phase 2), `engine.py` (`phase=2` → lagged retina, 4 channels)
- Create: `docs/protocols/03-direction-selectivity.md`
- Modify: `phenomena.py` with `direction_selectivity(hierarchy, period=8, speed=1, directions=12, frames=16) -> dict` (`dsi`, `median_dsi`, `fraction_selective` with DSI > 0.3). DSI = (R_pref − R_opposite) / (R_pref + R_opposite), using the mean response over drift.
- Test: `test_dsi_math`, and `test_mirror_symmetric_cells_have_low_dsi` (fixture).
- Pass criterion (declared in the protocol): the learned network's fraction selective is ≥ 0.2, and above the untrained network's.

### Task 9: V2 and top-down prediction

**Files:**
- Modify: `hierarchy.py`. A second `LevelConfig(cells=64, kernel=5, stride=2, padding=2)` predicts V1 activity. V1 inference adds the prior term −(r1 − Ŵ2·r2) with weight 0.5 (Rao & Ballard).
- Create: `docs/protocols/04-surround-suppression.md`, `docs/protocols/05-end-stopping.md`
- Modify: `phenomena.py` with:
  - `size_tuning(hierarchy, diameters=(4,6,8,12,16,24,32,48), use_feedback=True) -> dict` (`suppression_index` per cell and median), on the cells' preferred orientation;
  - `end_stopping(hierarchy, lengths=(2,4,6,8,12,16,24)) -> dict` (`end_stopping_index`).
- Tests:
  - `test_feedback_term_reduces_v1_activity_for_predictable_input`
  - the index-math tests
- Pass criterion: the median suppression index with feedback exceeds without feedback by ≥ 0.1. End-stopping is reported likewise.

### Task 10: Phase 2 adult, measurements, docs, PR

- [ ] Train the phase 2 adult for 3 h, then run protocols 1–5 on untrained, newborn and adult. Update the README and open a PR.

## Phase 3

### Task 11: Eye (fovea, saccades, efference copy)

**Files:**
- Create: `cortical_mirror/cortex/eye.py`
- Test: `tests/test_cortex_eye.py`

**Interfaces:**
- Produces: `class Eye(world_width=256, world_height=192, fovea=(128, 96), saccade_px=8, period_frames=10, inhibition=0.7)` with:
  - `look(world_gray: Tensor[1,1,Hw,Ww], surprise_map: Tensor[1,1,h,w] | None) -> tuple[Tensor fovea, int command]`
  - `gaze: tuple[int,int]`
  - `history`

  The command takes values 0–8: 0 means stay, 1–8 are compass directions. Saccades go toward the highest inhibition-weighted surprise, one fixed step at a time. The surprise map comes from the periphery level (Task 12).
- Modify: `Hierarchy`. Its temporal predictor becomes 9 kernels V[m], one per motor command, selected by the efference copy. The `step(code, learn, command=0)` signature keeps command=0 as the default, which is backwards compatible.
- Tests:
  - `test_eye_stays_inside_the_world`
  - `test_saccade_goes_toward_surprise`
  - `test_inhibition_of_return_avoids_last_location`
  - `test_command_selects_its_own_temporal_kernel`

### Task 12: Periphery pathway

**Files:**
- Modify: `engine.py` (`phase=3`). It runs the whole frame at 64×48 through its own small level (`LevelConfig(cells=16)`) with the same rules. Its code-space error map is the surprise map for the eye.
- Test: `test_periphery_error_map_marks_moving_region`

### Task 13: Phase 3 protocols and measurements

- [ ] Protocols:
  - `docs/protocols/06-predictive-remapping.md`: prediction error after a self-generated saccade vs the same image shift with command 0, as `remapping_ratio`. The pass criterion is a ratio < 0.8 in the adult.
  - `docs/protocols/07-surprise-driven-gaze.md`: the fraction of fixations on above-median surprise vs a random-gaze baseline. The pass criterion is a gain of ≥ 0.1 over random.
- [ ] Functions `predictive_remapping(engine)` and `gaze_statistics(engine, video)` in `phenomena.py`, with math tests.
- [ ] Train the phase 3 adult (3 h), measure, and write the reports.

### Task 14: Phase 3 demo, docs, PR

- [ ] The twins view shows each mirror's fovea and gaze path over the camera. CLI `twins --phase 3`. Update the README, then open a PR and merge it.
