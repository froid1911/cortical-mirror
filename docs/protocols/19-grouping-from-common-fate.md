# Protocol 19: Grouping learned from common fate

Written before the first measurement.

## Claim tested

What moves together belongs together (Wertheimer's common fate; Spelke's infants). A cortex can learn which features belong together from how they move, and then group by features alone. Prodöhl, Würtz & von der Malsburg (2003) learned collinearity links this way from video.

Here, an association field between V1 feature types at small offsets learns by a Hebbian rule gated by common fate: whether two places move together according to the learned MT cells. The test asks whether the learned links group objects, and whether they still do so on still frames, where motion gives no help.

## Stimulus

- The sprite world: 256 × 192, 3 objects, no occluders, sent to the cortex at 128 × 96.
- **Training:** seeds 900–959, 300 frames each.
- **Test:** seeds 9000–9019. From frame 100 on, every 20th frame (10 frames per seed).

## Procedure

1. V1 (the frozen phase 1 adult) codes each frame. On the training frames, MT (the frozen v0.9 adult) gives each place's velocity.
2. The association field learns only during training, and is frozen during the test.
3. On each test frame, the affinity between active places is computed from the learned field and **the current frame's V1 code only**: motion is not used. Groups are the connected components of affinities above the frame's 80th percentile of positive affinities, at least 6 V1 positions large.
4. Groups are compared with the true visible objects downsampled to V1's grid (a V1 position belongs to an object if most of its 4 × 4 retinal pixels do).

## Measures

- **Mean object IoU:** objects are matched to groups one-to-one, greedily, best first; an unmatched object scores 0.
- **ARI:** the adjusted Rand index between the group labels and the object labels over active V1 positions, with background as one label.

## Conditions

1. **Learned field.**
2. **Newborn field** (zero): every affinity is zero, so the read-out finds no groups and every object scores 0.
3. **Similarity baseline:** the field fixed to the identity (A[f, g, Δ] = 1 if f = g), which groups places with the same V1 types nearby. This is grouping by feature similarity, without learning.

## Pass criteria (declared in advance)

- The learned field's ARI exceeds the similarity baseline's by at least 0.05.
- The learned field's mean object IoU is at least 0.5.

The engineered scene graph's object IoU on comparable scenes is about 0.8 (protocol 8), and is reported for reference. Grouping from still features alone, without tracking, is not expected to reach it.

Failures are reported as failures.

## Amendment (2026-10-29, before any test measurement)

The learning rule counts a pair of places when at least one of them moves (MT speed above 0.2 retinal px per frame), not only when both do. Reason and the check behind it (training seed 900 only): see the design's amendment. Stimuli, measures, conditions and pass criteria are unchanged.
