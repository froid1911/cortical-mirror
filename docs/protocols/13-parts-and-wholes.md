# Protocol 13: Parts and wholes

Written before the first measurement.

## Claim tested

A scene graph explains a body as one thing made of parts. Infants group by common fate: surfaces that move together belong together, even when they also move relative to each other (Kellman & Spelke, 1983). In the graph, every pair of object nodes has a binding strength that moves toward their common fate: how well their recent displacements agree, and how close they are. Strongly bound nodes form a group, a parent node of its parts. Nothing about which parts make a creature is given.

## Stimulus

- **Creature world:** 256 × 192, a 1/f background, and 2–3 creatures (`2 + seed % 2`), 300 frames.
- Each creature has a torso (radius 12–16) and one or two limbs (radius 6–9, `1 + seed % 2`).
- The torso drifts and bounces like a v0.4 object. Each limb swings about the torso at a distance of 22–26 px: its angle oscillates by ±0.6 rad with a period of 60–90 frames.
- **Test seeds:** 6000–6019. **Development seeds:** 600–639.

## Procedure

From frame 200 on, every 10th frame:

1. Match nodes to parts (torsos and limbs) one-to-one, greedily by visible-mask IoU, keeping pairs with IoU ≥ 0.3.
2. For every pair of matched nodes, compare "same creature" in the truth with "same group" in the graph. A node without a group is its own group.

At frame 300, delete each creature whose parts are all matched:

- **Graph:** delete the matched nodes' group, or the nodes themselves if they are ungrouped.
- **Bitmap:** inpaint the union of those nodes' masks, as in protocol 10.

Score both against the truth rendered without the creature, inside the creature's full masks dilated by 2 px.

## Measures

- **Grouping recall:** among matched pairs from the same creature, the fraction in the same group.
- **Grouping specificity:** among matched pairs from different creatures, the fraction in different groups.
- **Creature delete:** mean MSE of the graph over the bitmap's.

## Conditions

1. **Binding:** the graph as built.
2. **Proximity only (baseline):** group every pair of matched nodes closer than 40 px.

## Pass criteria (declared in advance)

For the binding graph:

- recall ≥ 0.7;
- specificity ≥ 0.9;
- creature-delete ratio ≤ 0.75.

The proximity baseline is reported.

Failures are reported as failures.
