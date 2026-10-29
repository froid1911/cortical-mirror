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

## v0.7: motion cells (`v0.7-motion-cells.json`, protocol 16)

MT was raised for 2 hours on flank's GPU 1 (about 5.5 million frames of UCF101 training video) over the frozen phase 1 V1 adult (`train-mt.jsonl`, log on flank).

| Criterion (amended before the adult measurement; see the protocol) | Untrained | Adult | Verdict |
|---|---|---|---|
| Compass sectors covered by learned displacements ≥ 6 | 0 | 8 | pass |
| Median decoded direction error on textures ≤ 30° | 67° | **1.4°** | pass |
| Tuning agreement (preferred grating direction within 45° of the kernel's) ≥ 0.5 | none judged | 0.25 | **fail** |
| Original: fraction of types with responsibility DSI > 0.5 (untrained ≤ 0.1) | 0.94 | 0.94 | untrained condition fails, by construction |

**Why the tuning agreement fails.** It is not a sign error. Each type answers gratings about 60° off its own direction, on both sides:
- type 0 points right and answers gratings at 60° and 300°;
- type 3 points left and answers at 120° and 240°;
- type 5 points up and answers at 210° and 330°.

This is the aperture problem. A grating shows motion only along its stripes' normal, and the test gratings move slowly (1–2 px per frame), while the types learned fast motion (about 4 px per frame). A type moving 4 px along its direction explains a 2 px-per-frame grating whose direction lies 60° away (4 × cos 60° = 2). The cells are tuned to direction and speed together, which slow gratings cannot reveal. Translating textures are not ambiguous, and their direction is decoded to within 1.4°. Tested with gratings matched to each type's speed, the agreement would likely hold, but that test was not pre-registered, so the fail stands.

## v0.9: faster motion (`v0.9-motion-cells.json`, protocol 16)

MT with 32 types per scale was raised for 4 hours on flank's GPU 1 (about 9.9 million stream frames). Half the streams skipped every other frame (`train-mt-v09.jsonl`).

| Criterion | Untrained | Adult v0.9 | Verdict |
|---|---|---|---|
| Compass sectors covered ≥ 6 | 0 | 8 | pass |
| **Fast sectors** (displacement ≥ 1.5 positions) **≥ 6** (v0.7 adult: 4) | 0 | **7** | **pass** |
| Median decoded direction error ≤ 30° | 67° | 1.6° | pass |
| Tuning agreement ≥ 0.5 | none judged | 0.05 | **fail** (the aperture problem, as in v0.7) |

The untrained comparison used 16 types (the default), not 32. It covers no sectors at any size, so this does not change a verdict.
