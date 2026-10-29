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

## v0.10 step 1: grouping learned from common fate (`v0.10-grouping.json`, protocol 19)

An association field between V1's 32 types at offsets up to 4 positions learned for 60 training seeds × 300 frames of the sprite world (about 12 minutes on flank's GPU 1), gated by the v0.9 MT adult's velocities. The rule was amended before any test measurement (see the protocol). The test grouped 200 still frames (seeds 9000–9019) from features alone.

| Condition | Mean object IoU | ARI | Groups per frame |
|---|---|---|---|
| **Learned field** | **0.021** | **−0.017** | 9.4 |
| Newborn (zero) | 0 | 0 | 0 |
| Similarity baseline (identity) | 0.007 | −0.012 | 2.5 |

| Criterion | Result | Verdict |
|---|---|---|
| Learned ARI ≥ similarity ARI + 0.05 | −0.005 | **fail** |
| Learned mean object IoU ≥ 0.5 | 0.021 | **fail** |

The engineered scene graph reaches about 0.8 object IoU on comparable scenes (protocol 8).

**Why it fails (diagnosis on training seeds 900–909 only, after the test).** The links did learn something; the read-out cannot use it.
- Mean affinity between places in the same object is 2.3 times that across an object's border with the learned field, against 1.55 times with the similarity baseline. Common fate taught the field which neighbours belong together better than feature similarity does.
- But objects cover only 3.3% of V1's grid, and the background is a texture with nearly the same V1 statistics (the cosine between the mean V1 profiles of objects and background is 0.885). Of the frame's strongest 20% of affinities, 92% lie inside the background (91% for the baseline), about its share of all pairs. The global threshold and connected components therefore carve the background into texture patches, and the objects are rarely found.
- So a still frame in this world does not show its objects in local features. Objects there are defined by their borders and by moving, and a read-out by threshold has no notion of a figure against a ground.

**What this suggests for the next step** (not tested here): a read-out that lets groups compete for places (EM over slots) instead of a global threshold; border ownership, so that a border belongs to the figure inside it; and grouping during motion, where common fate is directly available, with the learned links carrying groups through still moments.

## v0.10 step 2: objects as slots found by learned motion (`v0.10-slots.json`, protocol 20)

Slots (a background slot and up to 6 object slots, assigned by EM) read the frozen v0.9 MT adult's velocity, V1's code and protocol 19's learned links. MT compares V1 now with V1 four frames ago (amended before the test; see the protocol). Test seeds 9000–9019: 200 frames scored while the objects move, and 20 still moments.

| Condition | Moving IoU | Moving ARI | Slots per frame | Still moment: IoU before → after |
|---|---|---|---|---|
| **Slots** (motion and links) | **0.304** | **0.505** | 5.6 | 0.313 → 0.123 |
| Silenced MT | 0 | 0 | 0 | 0 → 0 |
| Without links | 0.331 | 0.538 | 5.6 | 0.330 → **0.000** |

| Criterion | Result | Verdict |
|---|---|---|
| 1. Moving IoU ≥ 0.5 | 0.304 | **fail** |
| 2. Silenced MT: IoU ≤ 0.1 | 0.000 | **pass** |
| 3. Still moment: IoU after ≥ 0.8 × before (with links) | 0.39 × | **fail** |

For comparison: the engineered scene graph reaches about 0.8 IoU (protocol 8). Grouping still frames by learned links alone (protocol 19) reached ARI −0.017, and the slots reach 0.505.

**What it shows.**
- Learned motion finds objects: the ARI rises from chance (protocol 19) to 0.5. Silencing MT removes every object, so motion is what finds them.
- The learned links are what keep an object when it stops: without them, every slot is lost in the 30-frame pause (IoU 0.000); with them, 39% of the IoU remains. While objects move, the links cost a little (0.304 against 0.331).

**Why it falls short (development, training seeds only).**
- MT's velocity is too coarse for small, slow sprites. Frame to frame, its median error on the sprites is 0.58 retinal px per frame, about the sprites' own speed. Comparing frames four apart halves it.
- MT's motion spills onto the background next to a moving object: about 1.5 background places read as moving for every object place. As an object moves, the background it uncovers and covers changes too, and MT reads that change as motion.
- Even when each object's true velocity is given, choosing places by MT's velocity reaches an IoU of only 0.26. The slots (0.30) already do better than that through their memory and links. The bottleneck is MT's spatial precision at object borders, not the slot machinery.
- Objects split into fragments: 5.6 slots per frame for 3 objects, and the one-to-one IoU credits only one fragment per object.

**Next.** Make MT precise at borders: motion cells on V1's own grid with less pooling, and a learned account of covering and uncovering (what appears behind a moving edge belongs to the background), the same problem as border ownership. Then rerun protocol 20 unchanged.
