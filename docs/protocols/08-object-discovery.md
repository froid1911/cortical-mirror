# Protocol 8: Object discovery

Written before the first measurement on held-out scenes.

## Claim tested

Infants segment the world into objects by common fate: what moves together is one thing (Spelke, 1990). The scene level is given object files (slots with a position, velocity and depth) but not what any object looks like or which pixels belong to it. The test asks whether its nodes come to coincide with the objects in the scene, from the input alone.

## Stimulus

- The sprite world (`cortical_mirror/scene/world.py`), 256 × 192, no occluders.
- 40 held-out scenes, seeds 1000–1039. Scene `s` has `1 + s % 4` objects, so each count from 1 to 4 appears 10 times.
- 300 frames per scene.
- Seeds 0–19 were used while building the scene level; they are never measured here.

## Procedure

1. A fresh graph watches each scene frame by frame.
2. From frame 200 on, every 10th frame, take each true object's visible mask and every node's mask (alpha > 0.5, as placed in the world).
3. Match nodes to objects one-to-one, greedily by IoU, best pair first. An object without a node scores IoU 0.

## Measures

- **Mean IoU:** over objects, samples and scenes.
- **Count accuracy:** the fraction of samples whose node count equals the object count.

## Conditions

1. **Learning:** the graph as built.
2. **Frozen:** the same graph with learning off. Nodes still form and track, but their appearance and mask stay as they were at birth, and the background stays the first frame.
3. **Motion baseline (no objects):** the connected components of frame differences (|frame − frame 8 back| > 0.04, closed with a 7 × 7 kernel, components of at least 80 pixels), taken as nodes.

Nothing carries over from one scene to the next, so newborn and adult do not differ here. This protocol compares learning, no learning and a model-free baseline.

## Pass criteria (declared in advance)

All of the following for the learning graph:

- mean IoU ≥ 0.6;
- mean IoU at least 0.15 above both the frozen graph and the motion baseline;
- count accuracy ≥ 0.7.

Failures are reported as failures.
