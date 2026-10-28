# Cortex reports

The JSON files are the protocol measurements of v0.3's learning cortex (phases 1–3), written by `tools/phenomena.py`. The README's research-track section summarises them.

## Why direction selectivity failed (diagnosis, 2026-10-28)

Protocol 3 (direction selectivity) failed in phase 2. Learning made it worse:

| Network | Median DSI | Fraction direction-selective (DSI > 0.3) |
|---|---|---|
| Untrained (random weights) | 0.12 | 0.19 |
| Newborn after experience | 0.004 | 0.00 |
| Adult (77 h of UCF101) | 0.008 | 0.00 |

The phase 2 adult's V1 weights explain why. Every cell's weights on the lagged channels (the previous frame) are practically identical to its weights on the current frame: cosine similarity 0.997–1.000 across all 32 cells, with half the weight energy on each.

Consecutive frames of natural video are nearly the same. Sparse coding on [current, lagged] therefore learns the same pattern for both: a field that is separable in space and time, and blind to direction by construction. Learning removed the accidental direction preference that random starting weights had.

For v0.7: static structure must stop dominating the temporal input. Candidates are a transient (change) channel, several lags with temporal decorrelation, or an objective that rewards predicting change.
