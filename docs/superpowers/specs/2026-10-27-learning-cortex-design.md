# A visual cortex that learns to see: design

## Intent

The goal is a system that learns to see the way a real visual cortex does, judged by which cortical phenomena emerge on their own. The current mirror (v0.2) is a good next-frame predictor trained by backpropagation on 8,000 videos: an engineering result, and in the user's words "nice play" rather than biology. This track builds the alternative. It is a predictive-coding hierarchy that learns only with local rules, starts blank, and is judged by phenomena a real visual cortex is known to show, measured with neuroscience protocols. Its predictions are never fitted to brain recordings.

The CIMC hackathon (Friday 2026-10-30) is a place to share the idea and show a first working stage; the real work continues afterwards.

## Goals

1. A cortex-like hierarchy whose inference and learning are entirely local, with no backpropagation and no weight transport.
2. Two instances of the same network side by side as twin mirrors. **Newborn** starts blank in front of the camera. **Adult** has watched hours of video with exactly the same rules.
3. A growing, pre-registered checklist of emergent phenomena, each with a written protocol, a measurement tool and a report:
   - **Phase 1:** orientation-tuned cells (V1); stimulus-specific adaptation to repetition, and the response to a deviant (oddball).
   - **Phase 2:** surround suppression and end-stopping from top-down prediction (Rao & Ballard, 1999); direction-selective cells.
   - **Phase 3:** an eye that moves. Saccades are driven by surprise, and efference copies let it predict the consequences of its own movements (predictive remapping).

## Non-goals

- Beating v0.2's next-frame accuracy. The cortex mirror is expected to look blurrier.
- Spiking neurons, membrane dynamics, or any claim about consciousness.
- Fitting to, or scoring against, neural recordings.

## Rules (anti-goals)

These were decided by Claude and accepted by the user ("idk, what do you think?"). Breaking any of them makes a result invalid, however good it looks.

1. **No backpropagation, no weight transport.** Every update uses only the activity at the two ends of a connection and the error the cell itself carries. Feedback weights are separate and learn by the same local rule (Kolen-Pollack), and they start mismatched. The cortex code never uses autograd: tensors never require gradients, which a test enforces.
2. **Innate structure only where biology has it at birth.** Allowed: a retina-like front end (luminance, centre-surround ON/OFF cells, contrast normalisation, lagged cells), local receptive fields, a coarse-to-fine layout, and homeostatic set points. Not allowed: hand-designed edge filters or any built-in orientation, direction or saccade target.
3. **The newborn learns only from its own stream**, in order and in real time. The adult learns from video with the identical rules. No pretrained models anywhere in this track.
4. **Phenomena are measured, not eyeballed,** with the stimuli neuroscience used (gratings, bars, oddball sequences). Protocols are written before the measurement, and failures are reported.
5. **Edge cells are a first check, not the result.** Many learning methods produce them; the prediction-specific phenomena carry the weight.

### Declared simplification

Weights are shared across positions (convolution). This is equivalent to every location learning from the same statistics. Biology learns each location separately, so this is a computational shortcut, and the README says so.

## Architecture (Claude's calls)

- **Retina** (`cortex/retina.py`, fixed): luminance at 128×96, a difference-of-Gaussians centre-surround filter, divisive local contrast normalisation, and ON/OFF rectification into 2 channels. From phase 2 there are also 2 *lagged* channels, the previous frame's ON/OFF, modelling lagged LGN cells.
- **Levels** (`cortex/hierarchy.py`): V1 has 32 cell types, 8×8 receptive fields and stride 2; V2 is added in phase 2. Each level has:
  - generative weights W that predict the level below;
  - feedforward weights B that learn by the same local Hebbian product as W (Kolen-Pollack);
  - per-cell thresholds with slow homeostasis toward a target firing rate;
  - activities bounded between 0 and 1;
  - a temporal predictor V. It predicts the level's next activity from its current one, and learns by the local delta rule (prediction error times presynaptic activity).
- **Inference:** a few iterations per frame of error-driven updates (sparse coding / ISTA-like). The step size is 0.9 divided by the operator's largest eigenvalue, estimated by power iteration every 100 steps. The previous frame's state is the warm start.
- **Learning:** after inference, every frame. ΔW = ΔB = η · (error ⊗ activity), averaged over positions, then per-cell normalisation (synaptic scaling). ΔV follows the delta rule.
- **Prediction for the mirror:** the temporal predictor gives V1's next activity, and W renders it as the next retinal code. A *display adapter*, which is not part of the cortex, turns that retinal code into a picture: the current frame's low-pass luminance plus the predicted contrast. Prediction is **scored in retinal-code space** against "the next retinal code equals this one" (persistence).
- **Phase 3 eye:** the cortex sees a 128×96 foveal window of a larger frame. A coarse peripheral pathway (a small local-learning level over the whole frame) produces a surprise map. Saccades of a fixed size in 8 directions go toward peripheral surprise, with inhibition of return, at about 3 per second. The temporal predictor is conditioned on the motor command: a separate learned V per saccade direction, selected by the efference copy.

## Phenomena and protocols

Each protocol lives in `docs/protocols/` and is written before its first run. Measurements are made on a frozen network (no learning during the test), on the unlearned network as control, on the newborn after N frames, and on the adult.

| Phase | Phenomenon | Stimulus | Measure |
|---|---|---|---|
| 1 | Orientation tuning (V1 simple cells) | Static gratings, 12 orientations × 8 phases | Orientation selectivity index (OSI); fraction of cells with OSI > 0.3 |
| 1 | Stimulus-specific adaptation | Oddball: grating A at 90%, B at 10%, plus a many-standards control | Error response to B as deviant vs B in the control |
| 2 | Surround suppression | Grating patches of growing diameter on a cell's receptive field | Suppression index (peak minus largest-size response, divided by peak), with vs without V2 feedback |
| 2 | End-stopping | Bars of growing length | End-stopping index |
| 2 | Direction selectivity | Drifting gratings in both directions | Direction selectivity index (DSI); fraction > 0.3 |
| 3 | Predictive remapping | The same image shift, self-generated (saccade with efference copy) vs external | Prediction error ratio |
| 3 | Surprise-driven gaze | Free viewing of video | Fixations on above-median surprise vs a random-gaze baseline; inhibition of return |

## Demo

The demo is twin mirrors (layout A, chosen by the user): Newborn and Adult side by side, each with a live grid of its cells' receptive fields and an "age" (frames of experience). Each mirror's surprise curve is plotted against persistence. A key resets the newborn. In phase 3 the view also shows each eye's gaze path.

## Constraints

- GPUs on flank: 0, 1 and 3 only, never GPU 2.
- No AI attribution in commits, PRs or documents.
- Small, incremental commits on side branches, merged by pull request.
- The v0.2 code and release stay intact; this is a new package, `cortical_mirror/cortex/`.

## Left to the builder

Exact sizes and rates, V2's shape, protocol thresholds (decided in the protocols before measuring), and visual details of the twin view.
