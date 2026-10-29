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
