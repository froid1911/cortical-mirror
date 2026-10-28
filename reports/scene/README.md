# Scene-level reports

`v0.4a.json`: protocols 8 and 10 on held-out sprite-world scenes (`tools/scene_report.py`).

| Protocol | Criterion | Result | Verdict |
|---|---|---|---|
| 8 Object discovery | mean IoU ≥ 0.6 | 0.79 | pass |
| | ≥ 0.15 above the frozen graph | +0.03 (frozen 0.76) | **fail** |
| | ≥ 0.15 above the motion baseline | +0.25 (baseline 0.54) | pass |
| | count accuracy ≥ 0.7 | 0.87 (frozen 0.42) | pass |
| 10 Manipulation | delete: graph ≤ 0.75 × bitmap | 0.23 × (wins 95% of edits) | pass |
| | move: graph < bitmap | 0.53 × (wins 90%) | pass |
| | ≥ 80% of objects matched | 93% | pass |

Protocol 8 fails as written: masks born from the moving residual already overlap the objects well, so learning adds little IoU. What learning adds is the right number of nodes (0.87 vs 0.42) and the memory that protocol 10 needs.

`v0.4b-permanence.json`: protocol 9, first run, as pre-registered (seeds 3000–3019).

| Criterion | Result | Verdict |
|---|---|---|
| Newborn kept ≤ 0.2 | 0.0 | pass |
| Newborn violation ratio < 1 | 0.0 (surprised only by reappearance) | pass |
| Magic-world τ < half the adult's | 4.3 vs 126.6 | pass |
| Adult kept ≥ 0.8 | 0.4 | **fail** |
| Adult median position error ≤ 4 px | 3.6 | pass |
| Adult violation ratio ≥ 2 | 0.24 | **fail** |

Persistence is learned from the world: it rises from 1 to 127 frames where hidden things come back, and stays near 4 where they vanish. The adult still fails, and so does a graph with permanence built in (τ = 60), with identical numbers. The cause is tracking, not belief. Each test scene shows the screen for the first time, so the graph does not yet know it stands in front. As the object slides behind it, the search pulls the node back, and the extrapolated node lags the object.

`v0.4c-self-world.json`: protocol 12, first run. **Every criterion passes.** The motor map learned is exactly 32 px per saccade. The adult takes its own saccades for a still world (fraction 0.0) and the same shift without a command for the world moving (1.0). Before landing, it predicts the coming view from memory at 0.02 × the error of "no change". The newborn takes every shift for the world moving (commanded 1.0), and its prediction ratio is 0.99.

## Second run, final code (`v0.4.json`)

Tracking behind screens was then changed:
- visibility is measured from the pixels a node explains;
- shape is learned only from complete views;
- a node keeps to its path through a partial view (a continuity prior).

These changes were tuned on development seeds 3100–3119 and the discovery seeds 0–19, never on the test seeds. Every protocol was then run once more on its pre-registered held-out seeds. The first-run files above are kept unchanged.

| Protocol | First run | Second run |
|---|---|---|
| 8 Discovery: IoU (frozen, baseline) | 0.79 (0.76, 0.54): **fail** | 0.81 (0.31, 0.54), count 0.92: pass |
| 9 Permanence (adult): kept, error, violation ratio | 0.40, 3.6 px, 0.24: **fail** | 0.50, 4.4 px, 0.44: **fail** |
| 9 Newborn, and τ in normal vs magic world | pass; 127 vs 4.3 | pass; 116 vs 4.0 |
| 10 Manipulation: delete, move (× bitmap) | 0.23, 0.53: pass | 0.30, 0.48: pass |
| 12 Self vs world | pass | pass (unchanged) |

Protocol 8 passes in the second run partly because the frozen control got worse: tracking now relies on learning, since templates are refined and shape comes only from complete views, so nodes that cannot learn fall apart. Its margin over the motion baseline (+0.27) does not depend on that.

Protocol 9 still fails for the adult. Half the hidden objects are kept, and in those trials the vanish trials are 3–5 × as surprising as the reappear trials. Overall, though, the ratio stays below 1. The other half are lost on the way behind the screen, mostly low-contrast objects that the tracker already follows poorly in plain view. A graph with permanence built in (τ = 60) scores the same, so the open problem is tracking, not belief.

## Real video (`v0.4-sample-video.json`)

The release's sample clip, 750 frames, with the scene adult (`tools/scene_video.py`). There is no ground truth here, so this is a description, not a protocol. The graph explains each frame with a mean squared error of 0.0054. Its expectation of the next frame is **1.9 × worse than repeating the last frame**, and it holds 1.3 objects in view on average (8 at most). The scene level works on the sprite world. It does not yet work on real video, where the camera moves, people deform and the light changes.

## Protocol 11: learned attention (`v0.4-attention.json`)

The adult's gaze value was trained on seeds 500–539 and tested on 5000–5019, as pre-registered.

| Condition | Noise capture | World error | Time to novelty |
|---|---|---|---|
| Newborn (random) | 0.146 | 0.0220 | 72 frames |
| Surprise reflex (v0.3's rule) | 0.627 | 0.0643 | 272 frames |
| Adult (learned) | 0.009 | 0.0264 | 145 frames |

| Criterion | Result | Verdict |
|---|---|---|
| Setup: surprise reflex captured ≥ 1.5 × newborn | 4.3 × | pass |
| Adult noise capture ≤ 0.5 × newborn | 0.06 × | pass |
| Adult world error ≤ 0.9 × newborn | 1.2 × | **fail** |
| Adult time to novelty ≤ 0.5 × newborn | 2.0 × | **fail** |

The noise is a real trap: the surprise reflex spends 63% of its fixations on it, and so ends up knowing the world worst. The learned eye escapes the trap, but in the crude way. Its weights are negative for "unknown" (−0.30) and for "familiar" (−0.11), positive only for the bias (+0.16), and near zero for error (+0.05) and novelty (+0.02). It has learned to avoid unknown places rather than to seek the ones that teach, so it explores less evenly than random gaze and finds a newcomer more slowly. A linear value over these features cannot tell an unexplored place from noise until the place has been looked at. The design's hope, curiosity directed at learning progress, did not emerge.

## Known deviations and limits (found in review, after measuring)

- **Protocol 11, time to novelty:** the code counts the newcomer as found on any frame its centre is inside the fovea. The protocol says at a fixation. So a newcomer drifting into a still fovea counts too. This favours every condition alike, and the measured numbers are left as they are.
- **Protocol 10, bitmap baseline:** the code inpaints the mask dilated by one pixel, where the protocol says the mask itself. That gives the baseline a slightly larger hole. It probably costs the baseline a little, and it does not change which wins.
- **Speed on real video:** recognising forgotten objects searches widely and can slow the live page well below 25 fps on busy footage.
- **The eye graph** keeps its gaze inside its canvas only because the protocol drivers keep it there; it is not part of the live page.

## v0.5

Protocols 13–15 were written before measuring; 9 and 11 were re-run on fresh test seeds. The design and its amendments (all made before any test measurement) are in `docs/superpowers/specs/2026-10-28-v0.5-design.md`.

### Protocol 13: parts and wholes (`v0.5-parts.json`)

| Criterion | Result | Verdict |
|---|---|---|
| Grouping recall ≥ 0.7 | 0.62 | **fail** |
| Grouping specificity ≥ 0.9 | 0.95 | pass |
| Creature delete ≤ 0.75 × bitmap | 0.31 × (9 creatures) | pass |

Proximity alone (reported, no criterion) groups almost every same-creature pair (recall 0.98), but also wrongly groups 11% of pairs of different creatures (specificity 0.89). Binding by common fate is cleaner, but it misses about 4 pairs in 10. On the development seeds it reached recall 0.72, so the test seeds were harder.

### Protocol 14: objects under changing light (`v0.5-lighting.json`)

| Code | Changing light: IoU, counts right | Constant light: IoU, counts right |
|---|---|---|
| Normalised (v0.5) | 0.87, 0.54 | 0.86, 0.69 |
| Brightness (v0.4) | 0.24, 0.00 | 0.81, 0.90 |
| V1's code (reported) | 0.27, 0.55 | 0.27, 0.38 |

| Criterion | Result | Verdict |
|---|---|---|
| Changing light: normalised IoU ≥ 0.6 | 0.87 | pass |
| … and ≥ brightness + 0.15 | +0.63 | pass |
| Constant light: not more than 0.05 below brightness | +0.06 | pass |

Passes as written. The pre-registered criteria are on IoU. Count accuracy is lower than on the development seeds (0.54 and 0.69 against 0.99 and 0.90), and lower than brightness under constant light: the normalised code makes more spurious nodes. V1's learned code finds only outlines.

### Protocol 9 re-run (`v0.5-permanence_v05.json`, seeds 3020–3039)

Newborn: kept 0.0, violation ratio 0.0, pass. τ: 118 frames in the normal world, 4.0 in the magic world, pass. Adult: kept 0.55, **fail**; position error 1.7 px, pass; violation ratio 0.76, **fail**. A built-in τ = 60 scores the same. Permanence still fails on tracking behind a screen the graph has not seen before. The two remedies tried in development both made it worse and are not in v0.5.

### Protocol 15: real video (`v0.5-real-video.json`, 40 UCF101 test clips)

| Condition | Expectation ÷ last frame | Clips better than the last frame |
|---|---|---|
| v0.4 graph | 4.59 | 0% |
| v0.5, registration, brightness | 1.04 | 35% |
| v0.5, registration, normalised | 1.04 | 40% |

**Fails** (needs below 0.95). Registering the camera's motion and predicting only what it explains takes the graph from 4.6 × worse than doing nothing to about even with it. On development, the last frame shifted by the camera alone reached 0.99. In this footage, what changes between frames is mostly non-rigid motion and noise, which a graph of rigid objects does not predict.

### On a GPU

`SceneGraph(device="cuda")` (or `SCENE_DEVICE=cuda`) runs the search, 78% of the time, on a torch device. The placements are the same as on the CPU (`tests/test_scene_device.py`). On flank: 40 ms per frame on the CPU arrays, 12.9 ms on one GPU. Protocol seeds run in parallel processes (`SCENE_WORKERS`).

### v0.4's protocols on v0.5's code (`v0.5-carryover.json`, same seeds, a third run)

| Protocol | Result | Verdict |
|---|---|---|
| 8 Discovery | IoU 0.81 (frozen 0.30, motion baseline 0.54), count 0.91 | pass |
| 10 Manipulation | delete 0.22 ×, move 0.44 × the bitmap; 94 of 100 matched | pass |
| 12 Self vs world | commanded 0.0, uncommanded 1.0, prediction 0.02; newborn 1.0 | pass |

Run on flank's GPU 3 with `SCENE_DEVICE=cuda`.

### Protocol 11 re-run (`v0.5-attention_v05.json`, seeds 5020–5039)

| Condition | Noise capture | World error | Time to novelty |
|---|---|---|---|
| Newborn (random) | 0.175 | 0.0217 | 152 frames |
| Surprise reflex | 0.633 | 0.0636 | 258.5 frames |
| Adult (learned, with place values) | 0.109 | 0.0202 | 134 frames |

| Criterion | Result | Verdict |
|---|---|---|
| Setup: surprise reflex ≥ 1.5 × newborn | 3.6 × | pass |
| Adult noise capture ≤ 0.5 × newborn | 0.62 × | **fail** |
| Adult world error ≤ 0.9 × newborn | 0.93 × | **fail** |
| Adult time to novelty ≤ 0.5 × newborn | 0.88 × | **fail** |

All three fail. Unlike v0.4, the adult is now better than random gaze on every measure. The learned feature weights are still negative for "unknown" (−0.21), "error" (−0.09) and "novelty" (−0.06). What pulls the gaze is mostly each place's learned value.

**Fixed after this run:** the eye graph's error, visit and progress maps were not shifted when registration found the world itself moving. With the true motor map used here, that shift is essentially zero, so the numbers above stand.

## After v0.5: Kalman tracking (development seeds only, no protocol run)

`SceneGraph(tracking="kalman")` fuses each search result with a constant-velocity prediction, weighted by how much of the node was seen (noise ∝ 1/visibility⁴), and coasts while the node is hidden, widening its search with its uncertainty. It was compared with v0.5's direct tracking on development seeds only: permanence 3100–3119, grouping 600–619, discovery 0–19.

| Tracker | Permanence: kept, violation ratio | Grouping: recall, specificity | Discovery: IoU, counts right |
|---|---|---|---|
| Direct (v0.5, default) | 0.65, 0.45 | 0.72, 0.97 | 0.81, 0.91 |
| Kalman, acceleration 0.02, power 4 | 0.70, 0.60 | 0.73, 0.95 | 0.77, 0.79 |
| Kalman, 0.01, power 5 | 0.70, 0.72 | 0.52, 0.98 | 0.77, 0.79 |
| Kalman, 0.02, power 6 | 0.65, 0.72 | 0.56, 0.97 | 0.78, 0.82 |

A single filtered estimate gains a little permanence and loses discovery, so it is not the fix, and it stays an option, not the default. No pre-registered protocol was run. Getting past a screen likely needs several hypotheses per hidden object (multi-hypothesis or particle tracking), not one estimate weighted more or less.

## After v0.5: headroom of motion models on real video (`dev-headroom.json`)

`tools/video_headroom.py` on 20 UCF101 validation clips (development, not a protocol). Each model of how frame t becomes t+1 is fitted to dense optical flow. The numbers are the mean squared error of the predicted next frame divided by that of repeating frame t.

| Model | Oracle (fitted to t+1) | Last motion carried forward | × 0.25 | × 0.5 | × 0.75 | Last 3 averaged, × 0.5 |
|---|---|---|---|---|---|---|
| Global shift | 0.65 | 1.09 | 0.97 | 0.99 | 1.03 | 0.92 |
| Translation per 32 px block | 0.36 | 1.06 | 0.93 | 0.94 | 0.98 | 0.86 |
| Affine per block | 0.29 | 1.07 | 0.92 | 0.92 | 0.98 | 0.85 |
| Dense flow | 0.23 | 1.06 | 0.91 | 0.92 | 0.97 | 0.84 |

- **Large headroom:** knowing the motion into the next frame would cut the error by up to 77%.
- **Raw motion does not continue:** carrying the last frame's motion forward is worse than repeating the frame, for every model.
- **Smoothed motion does:** averaged over three frames and carried forward at half strength, it beats repeating by about 15%.
- **Affine per block is enough:** it gets 0.85, dense flow 0.84. For predicting real video, parts with affine motion and smoothed, damped prediction should suffice; fully deformable objects are not needed. That is the path for v0.6's real-video work.

## After v0.5: particle tracking (development seeds only, not merged)

32 hypotheses per node, weighed by the search's normalised evidence, surviving while hidden. On the development seeds it was far worse than direct tracking:

| Measure | Direct | Particles |
|---|---|---|
| Permanence kept | 0.65 | 0.25 |
| Grouping recall | 0.72 | 0.39 |
| Discovery IoU / counts right | 0.81 / 0.91 | 0.56 / 0.28 |

The node's position, the mean of its hypotheses, jitters from frame to frame. Shape learning then fails and nodes fragment. The code stays on branch `feat/v0.6-particles` and is not merged.

## v0.6

The design and its amendment (made before any test measurement) are in `docs/superpowers/specs/2026-10-28-v0.6-design.md`. Two changes:

- **Part motion:** the next frame is predicted from the smoothed affine motion of the graph's parts (object nodes, and background blocks).
- **Velocity from complete views:** an established node's velocity, like its shape, is learned only from complete views.

### Protocol 15: real video (`v0.6-real-video.json`, 40 fresh UCF101 test clips, seed 1, excluding v0.5's)

| Condition | Expectation ÷ last frame | Clips better than the last frame |
|---|---|---|
| v0.4 graph | 5.18 | 0% |
| v0.5, registration, brightness | 0.91 | 45% |
| v0.5, registration, normalised | 0.905 | 45% |
| **v0.6, part motion** | **0.765** | **95%** |

**Passes** (below 0.95). These clips are easier than v0.5's: v0.5's own prediction scores 0.905 here, against 1.04 on its clips. The fair comparison is on the same clips: part motion 0.765 against 0.905, about 15% less error, and better than the last frame on 95% of clips against 45%.

### Protocol 9: permanence (`v0.6-permanence.json`, seeds 3040–3059, adult raised afresh)

| Criterion | Result | Verdict |
|---|---|---|
| Adult kept ≥ 0.8 | 0.80 | pass |
| Adult median position error ≤ 4 px | 3.7 | pass |
| Adult violation ratio ≥ 2 | 1.19 | **fail** |
| Newborn kept ≤ 0.2, violation ratio < 1 | 0.0, 0.0 | pass |
| Magic-world τ < half the adult's | 3.9 vs 112.7 | pass |

The adult now keeps hidden objects and knows where they are: v0.5 kept 0.55, v0.4 0.40–0.50. It is surprised when one fails to reappear, but not twice as surprised as when it comes back.

### Protocol 13: parts and wholes (`v0.6-parts.json`, seeds 6020–6039)

| Criterion | Result | Verdict |
|---|---|---|
| Grouping recall ≥ 0.7 | 0.60 | **fail** |
| Grouping specificity ≥ 0.9 | 0.94 | pass |
| Creature delete ≤ 0.75 × bitmap | 1.09 × (3 creatures) | **fail** |

Grouping did not improve. Deleting a creature is now worse than the bitmap, but only 3 creatures were fully matched (v0.5: 9), so that number rests on very few cases.

### v0.4's protocols on v0.6's code (`v0.6-carryover.json`, same seeds as before)

| Protocol | Result | Verdict |
|---|---|---|
| 8 Discovery | IoU 0.82 (frozen 0.30, motion baseline 0.54), counts right 0.94 | pass |
| 10 Manipulation | delete 0.21 ×, move 0.43 × the bitmap; 96 of 100 matched | pass |
| 12 Self vs world | commanded 0.0, uncommanded 1.0, prediction 0.02; newborn 1.0 | pass |

Run on flank's GPU 3. The first attempt crashed at its last step (the checkpoint folder was missing in a fresh worktree) and was re-run in full.

## v0.7: learned motion, used (`v0.7-learned-motion.json`, protocol 17)

40 fresh UCF101 test clips (seed 2, excluding the clips of seeds 0 and 1).

| Condition | Expectation ÷ last frame | Clips better than the last frame |
|---|---|---|
| **Learned motion (MT adult)** | **0.878** | 87.5% |
| MT silenced | 1.000 | 0% |
| Optical flow (v0.6, yardstick) | 0.822 | 87.5% |

**Passes both criteria:** learned motion below 1.0, silenced at or above 0.99. Real video is now predicted from motion the cortex learned. Silencing MT takes the prediction back to exactly the last frame, so nothing engineered senses motion in that path. Learned motion is 7% behind engineered flow in error, and equal in the fraction of clips it wins. Phase correlation still aligns the graph's canvas, as labelled scaffolding outside the prediction.
