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
