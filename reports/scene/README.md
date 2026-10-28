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
