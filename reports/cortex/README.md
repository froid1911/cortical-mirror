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

### Checks after protocol 20: where MT's spill comes from (development, training seeds 900–903)

The report above blamed covering and uncovering at moving borders for MT's spill. Measured, that is the smaller part:

| Where the spilled "moving" background places lie | Share |
|---|---|
| In the area the objects swept during the 4-frame gap (covered or uncovered) | 16% |
| A ring 1–2 V1 positions around each object, outside the swept area | 61% |
| 3–8 positions away | 18% |
| Farther (noise) | 20% |

(The distance bands overlap the swept area, so the shares add to more than 100%.)

Objects have a median radius of only 3.1 positions on V1's grid, so a ring of 1–2 positions nearly doubles them. The ring comes from V1's receptive fields and MT's pooling straddling the object's edge, a matter of resolution.

| Check (no retraining) | Spill per object place | Place-wise ceiling IoU |
|---|---|---|
| As in protocol 20 (128 × 96, smoothing 3, gap 4) | 1.78 | 0.256 |
| A "novelty" competitor in MT (a place no type predicts well counts as unknown), κ = 1 | 1.17 | 0.165 |
| Full resolution (256 × 192), smoothing 3 | 1.08 | 0.202 |
| Full resolution, smoothing 5 | 1.37 | 0.214 |

- **Novelty does not single out occlusion.** The untrained MT predicts the sprites' own motion about as badly as uncovered background: at κ = 1 novelty takes 49% of swept background places and 43% of object places.
- **Looking closer shrinks the ring.** Spill falls from 1.78 to 1.08 background places per object place. But MT's place-by-place velocity does not get more precise (median error 0.7–0.9 world px per frame).
- **What this means for protocol 20.** A perfect "is it moving" map, with MT's coverage and spill, bounds the object IoU near 0.8 / 2.8 ≈ 0.29 at 128 × 96, which is what the slots reached (0.30). At full resolution the bound is near 0.37. Neither reaches 0.5.

## v0.10 step 3: links relearned from integrated motion (`v0.10-links.json`, protocol 21)

The learned links were retrained by protocol 19's rule from the motion the slots use (MT comparing V1 four frames apart), on training seeds 900–949. Everything else is protocol 20's. Test: fresh seeds 9100–9119.

| Condition | Moving IoU | Moving ARI | Still moment: IoU before → after |
|---|---|---|---|
| **Slots, relearned links** | **0.295** | **0.520** | 0.267 → 0.139 |
| Silenced MT | 0 | 0 | 0 → 0 |
| Without links | 0.332 | 0.551 | 0.280 → 0.012 |

| Criterion | Result | Verdict |
|---|---|---|
| 1. Moving IoU ≥ 0.5 | 0.295 | **fail** (expected: MT's map bounds it near 0.29) |
| 2. Silenced MT: IoU ≤ 0.1 | 0.000 | **pass** |
| 3. Still moment keeps ≥ 80% of its IoU | 52% | **fail** |

- The relearned links keep more of each object through a pause than protocol 20's links (52% against 39%; without links, 4%).
- Development on 4 held-out training seeds had shown 88%. Four seeds were too few, and the estimate was optimistic.
- While objects move, the links still cost a little IoU (0.295 against 0.332 without them).

**Where v0.10 stands.** Learned motion finds objects (ARI about 0.5, up from chance in protocol 19), and learned links hold them through pauses partly but not well enough. Object IoU is capped near 0.3 by how sharply MT sees motion at an object's edge, about half of what the protocol asks for.

### Where spatial precision is lost (development, training seeds 900–903 and 950–953)

After protocol 21, a stronger test: assign each place to the layer that predicts it best. Each object's layer moves the code from 4 frames ago by the object's **true** velocity, and the background layer keeps it still. This needs no MT at all, so it bounds what any motion read-out of a representation can reach.

| Representation the layers predict | Bound on object IoU |
|---|---|
| Raw pixels, 256 × 192 | 0.60 |
| Raw pixels, 128 × 96 | 0.52 |
| Retina (centre–surround contrast), 128 × 96 | 0.42; its contrast normalisation makes no difference (0.41 without) |
| V1's feedforward response (the learned filters, rectified) | 0.38 at stride 2 or 1 alike |
| V1's settled sparse code (what MT reads) | 0.30 at 128 × 96 and 0.29 at 256 × 192 input; 0.27 after 60 settling steps |
| V1's settled code pooled 3 × 3 (as complex cells would) | 0.26 |

- Every stage loses edge sharpness, and 0.5 sits near the pixel level. With frame pairs, no read-out of this cortex reaches protocol 20's criterion.
- The largest single loss is V1's settling. Competition between cells makes the code jump under small shifts, and longer settling makes it worse. Pooling into complex cells blurs edges further.
- In cortex, MT's fast (magnocellular) input arrives with the first feedforward sweep, before recurrent processing (Lamme & Roelfsema 2000). With `MotionCortex(fast=True)`, MT reads V1's feedforward response instead, with no retraining.
  - MT's spill falls from 1.78 to 1.36 background places per object place, and its error from 0.36 to 0.32 px per frame.
  - On held-out training seeds 950–953 the slots reach moving IoU 0.36 (0.295 on the settled code), ARI 0.53, and keep 77% of their IoU through the still moment with protocol 21's links.
- The engineered scene graph reaches 0.8 because it builds pixel templates from many complete views. Sharp objects need memory of shape across time, not better frame-pair evidence.

## v0.10 step 4: manipulating a scene of learned objects (`v0.10-manipulation.json`, protocol 22)

A scene of learned objects: the motion slots (MT's fast path, protocol 21's links) and a background memory, a running average at every pixel the background slot owns. Protocol 10's edits on fresh seeds 2100–2139 (100 objects): delete each matched object, and move it 30 px. Nothing was tuned; three development seeds (2200–2202) only checked that the code runs.

| Criterion | Learned scene | Verdict |
|---|---|---|
| 1. Delete: MSE ≤ 0.75 × the bitmap baseline's | **0.24 ×**, better than the bitmap in 98% of edits | **pass** |
| 2. Move: MSE < the bitmap baseline's | **0.59 ×**, better in 98% | **pass** |
| 3. Objects matched (a slot covers ≥ half the object) ≥ 80% | **88%** | **pass** |

Yardstick, the engineered scene graph on the same scenes (protocol 10's matching):

| | Delete MSE (× its bitmap) | Move MSE (× its bitmap) | Matched |
|---|---|---|---|
| Learned scene | 0.0015 (0.24 ×) | 0.0032 (0.59 ×) | 88% |
| Engineered graph | 0.0004 (0.16 ×) | 0.0009 (0.44 ×) | 98% |
| Leaving the frame unchanged | 0.020 | 0.021 | |

- **What passes.** Objects found by learned motion, plus a memory of the background, explain the scene well enough to show what it would look like after a change, far better than a bitmap with the same segmentation. That was Joscha Bach's request, now met by learned objects instead of templates.
- **What is still behind.** The engineered graph's absolute error is about a quarter (delete) and a third (move) of the learned scene's. Its masks are sharp, built from pixel templates over many complete views. The learned masks are blocky: 4 × 4 px cells, larger than the objects. A moved object therefore drags some background along, and that dominates the remaining move error.
- **Engineered parts:** rendering by pasting, the inpainting fallback, and the slots' EM.

## v0.11: object files (`v0.11-files/`, protocol 23)

Each slot keeps an object file. In the object's own frame, every place keeps a running mean and variance (a local rule that forgets slowly). The object's shape is where the variance stays low, because the object's texture moves with it while the background slides past. The file follows its slot's velocity and corrects its position within ±3 px by its own learned appearance. Settings were chosen on training seeds 940–943 and 950–953 (see the protocol). Test: moving scenes on seeds 9200–9239 (400 scored frames), manipulation on seeds 2300–2339.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Pixel object IoU with files ≥ 0.5 | **0.505** (without files 0.379) | **pass**, narrowly |
| 2. MT silenced: IoU ≤ 0.1 | 0.000 | **pass** |
| 3. Move: MSE with files < without | 0.0020 against 0.0036 | **pass as written; see below** |

Reported without a criterion:

| Manipulation (seeds 2300–2339) | With files | Without files |
|---|---|---|
| Objects matched (a mask covers ≥ half the object) | **67%** | 85% |
| Delete, × the bitmap with the same mask | **0.56 ×** | 0.26 × |
| Move, × the bitmap with the same mask | 0.71 × | 0.64 × |

- **What the files do.** Masks become sharp: for the first time, learned objects reach protocol 20's bar of 0.5 IoU, from 0.30 on V1's grid. Motion still finds them: silenced, nothing is found.
- **Criterion 3 compares different sets.** With files, fewer objects are matched (67% against 85%), so the two move means are over different objects. The criterion passes as written, but it does not show that the files make moving better.
- **Sharp is not yet reliable.** A file's mask sometimes misses part of its object, or slips onto a neighbour, so fewer objects are covered by half. Deleting then leaves remnants: 0.56 × the bitmap's error against 0.26 × with the coarse masks. A coarse mask that covers too much is safer to delete with than a sharp one that covers too little.
- **Next:** files that know how sure they are. Use the sharp mask where the file is confident, the coarse one where it is not, and let a file that loses its object hand back to its slot.

## v0.12: object files that know how sure they are (`v0.12-confident-files/`, protocol 24)

On training seeds, 10 of protocol 23's 17 mature files had lost their objects. Now a file measures its fit: its prediction error relative to the error of the rival explanation, the remembered background (or an unrelated template where no background is known).
- A file that drifted onto background explains it well, but no better than the background does, so it is not trusted.
- A mature file that cannot explain what it sees does not learn from it, and looks for its object again near its slot.
- The scene uses a file's sharp mask only while the file is sure.

Development showed that measuring the fit against the file's own variance was unstable; the background as rival made it work. Test: moving scenes on seeds 9300–9339; manipulation on seeds 2400–2439, each scene run with and without files and compared on the 76 objects both matched.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Pixel object IoU with files ≥ 0.5 | **0.564** (without files 0.383) | **pass** |
| 2. Objects matched with files ≥ 80% | **84%** (protocol 23: 67%; without files 88%) | **pass** |
| 3. Delete: ≤ 0.75 × the bitmap, and below the scene without files (paired) | 0.52 × the bitmap; MSE 0.0017 against 0.0011 without files | **fail** |
| 4. Move: below the scene without files (paired) | MSE 0.0015 against 0.0029; files better on 72% of objects | **pass** |
| 5. MT silenced: IoU ≤ 0.1 | 0.000 | **pass** |

- **Sharp and mostly reliable.** Learned objects now reach 0.56 pixel IoU, and moving a learned object errs about half as much with its object file as without.
- **Why delete fails.** Files make deleting better on 61% of objects, but not on average. When a sure file's shape misses part of its object, that part stays behind after the delete, and a few such remnants outweigh the many small gains. For deleting, a coarse mask that covers too much is the safer answer; for moving, the sharp one is.
- Development seeds (2200–2209) had shown delete 0.12 ×; the test shows 0.52 ×. Ten development scenes were again too few.
- **Next:** let the edit ask what it needs. Delete everything that could belong to the object (the sharp shape and the slot's region), and move only what surely does (the sharp shape).

## v0.13: remove what could belong, paste what surely does (`v0.13-cautious-edits/`, protocol 25)

An edit now removes all that could belong to an object, a sure file's shape *and* the slot's coarse region. When moving, it pastes only what surely belongs, the sure shape. Nothing else changed from protocol 24. Test: fresh seeds 2500–2539, each scene with and without object files, compared on the 70 objects both matched.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Objects matched with files ≥ 80% | **77%** (without files 87%; protocol 24: 84%) | **fail**, narrowly |
| 2. Delete: ≤ 0.75 × the bitmap, and below the scene without files | **0.15 ×**; MSE 0.00046 against 0.00127; files better on 73% of objects | **pass** |
| 3. Move: below the scene without files | MSE 0.00097 against 0.00302; files better on 77% | **pass** |

- **Deleting is fixed.** Removing all that could belong to an object leaves no remnants. With object files, deleting errs about a third as much as with coarse masks alone, and moving about a third.
- **Close to the engineered graph.** Its errors in protocol 22 (other seeds) were 0.0004 for delete and 0.0009 for move. The learned scene now reaches 0.00046 and 0.00097. These are different seeds, so this is a comparison, not a criterion.
- **Matching is not yet stable.** Unchanged code matched 84% of objects in protocol 24 and 77% here. A sharp mask must cover half of its object to count, and some sure files cover less. Matching sits around the 80% bar, not safely above it.

The live page can now show these learned objects: `cortical-mirror scene --objects learned` (for a still camera, since the background is remembered pixel by pixel).

### After protocol 25: matching, and the first real video (development; no protocol)

**Matching.** On development seeds 2200–2214, 33 of 38 objects were matched (87%). The 5 misses had no common cause:
- twice the coarse slot already missed the object;
- twice the file sat partly off it;
- once its variance was too high.

With 84% (protocol 24) and 77% (protocol 25), matching lies near 82%, about ±4 points for 100 objects. Nothing was changed for it.

**The first real video** (`docs/images/learned-video.gif`, `tools/record_learned_video_gif.py`): the sample clip, a person signing in front of a still camera (median global shift 0.16 px per frame). Qualitative only; there is no ground truth.
- The learned scene finds the moving hands and arms as objects, unaided.
- The torso hardly moves, so, as for an infant, it is not an object but background. Deleting every object leaves a ghost of the person: the background memory has averaged the body and the arms that passed.
- Hands split into several slots, and an unsure slot's coarse region can spread over the body.
- Two faults showed here first, and are fixed, with tests:
  1. A file on a place never seen as background could only be compared with an unrelated template, and could win that comparison by chance. A file is now sure only against a background the scene knows.
  2. On a plain, still region, file and background both explain the image almost perfectly, and the ratio of two tiny errors is noise. A file is now sure only where the background fails to explain what it sees (mean squared error ≥ 0.002).
- Neither change alters the sprite results on development seeds: IoU 0.58, 88% matched, delete 0.00034 and move 0.0015 against 0.0015 and 0.0043 without files.
