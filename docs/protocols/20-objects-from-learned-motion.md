# Protocol 20: Objects from learned motion

Written before the first measurement.

## Claim tested

A cortex can find objects as what moves together, using only motion it has learned, and keep them through a still moment with the links it learned from common fate (protocol 19).

## Stimulus

- The sprite world: 256 × 192, 3 objects, no occluders, sent to the cortex at 128 × 96, as in protocol 19.
- **Test seeds 9000–9019.** Slots run on every frame from the first.
- **Moving scenes:** the world runs 300 frames. Scored from frame 100 on, every 20th frame (10 frames per seed).
- **Still moment:** a separate run per seed. The world runs 150 frames, then stops for 30 frames: the same picture is shown again and nothing moves. Scored at frame 149 (before) and at frame 179 (end of the pause).
- Development and tuning use training seeds 900–959 only.

## Measures

As in protocol 19, against the true visible objects on V1's grid:
- **mean object IoU**, objects matched one-to-one to object slots, greedily, best first (an unmatched object scores 0);
- **ARI** over all V1 positions, with the background slot and the background as label 0.

## Conditions

1. **Slots:** learned motion and learned links.
2. **Silenced MT:** the velocity is zero everywhere.
3. **Without links** (β = 0): for the still moment.

## Pass criteria (declared in advance)

1. Moving scenes: the slots' mean object IoU ≥ 0.5.
2. Silenced MT: mean object IoU ≤ 0.1. Motion, not something else, finds the objects.
3. Still moment, with links: mean IoU at the end of the pause ≥ 0.8 × the mean IoU before it.

Reported without a criterion:
- the still moment without links;
- the moving scenes' ARI, next to protocol 19's (learned field −0.017, similarity baseline −0.012);
- for reference, the engineered scene graph's object IoU on comparable scenes, about 0.8 (protocol 8).

Failures are reported as failures.

## Amendment (2026-10-29, before any test measurement)

Development on training seeds 900–905 only:
- **MT integrates over time.** MT's velocity on the sprites, frame to frame, has a median error of 0.58 retinal px per frame, about as large as the sprites' speeds. It sees motion on only 44% of object places. Comparing V1 now with V1 four frames ago, divided by four, halves the error (0.35) and sees 81% of object places as moving, with MT's evidence pooled over 3 instead of 5 positions. These are settings of the frozen MT adult's read-out, not new learning.
- **Slot settings:** σ = 0.25 retinal px per frame, β = 2, blur 0.2, 3 EM rounds, merge tolerance = σ.
- **What development showed, so that the test is read in its light:**
  - On training seeds 900–903 the slots reach a moving IoU of 0.27 (ARI 0.48) and keep 37% of their IoU through the still moment.
  - Even given each object's true velocity, choosing places by MT's velocity reaches IoU 0.26 at best: MT's motion spills onto 1.5 background places per object place.
  - Criteria 1 and 3 are therefore expected to fail. The criteria are not changed.
