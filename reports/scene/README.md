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
