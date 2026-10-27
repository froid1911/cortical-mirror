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
