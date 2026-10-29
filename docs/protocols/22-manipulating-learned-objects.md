# Protocol 22: Manipulating a scene of learned objects

Written before the first measurement.

## Claim tested

A scene whose objects are found by learned motion, with a memory of the background, explains the world well enough to show what it would look like after a change. This is protocol 10's claim, now for learned objects.

## Stimulus

As protocol 10, on fresh seeds:
- the sprite world, 256 × 192, no occluders;
- 40 scenes, **seeds 2100–2139**, scene `s` with `2 + s % 2` objects;
- the scene watches 240 frames.

## Procedure

As protocol 10, with one change in matching:
- **Matching:** each true object is matched to the object slot that covers most of its visible mask, one-to-one, greedily, best first. It counts as matched if that slot covers at least half of the object's visible pixels. (Protocol 10 matched by IoU ≥ 0.3. Learned masks are larger than the objects, by design of what the cortex can see, so coverage is the fair test of "this slot is that object".)
- For each matched object: delete it, and move it 30 px to the right (to the left if its centre is right of the middle).
- **Truth:** the sprite world rendered with the same edit.
- **Bitmap baseline:** the current frame edited with the same slot's mask: inpainted for delete; inpainted and the masked pixels pasted at the new place for move. Both use the same segmentation and differ only in what lies behind it.
- **Scoring region and measure:** protocol 10's (the union of the object's full true mask before and after, dilated by 2 px; mean squared error).

The engineered scene graph's protocol 10 on the same seeds is run and reported as the yardstick.

## Pass criteria (declared in advance)

1. **Delete:** the learned scene's mean MSE ≤ 0.75 × the bitmap baseline's.
2. **Move:** the learned scene's mean MSE < the bitmap baseline's.
3. At least 80% of objects matched.

Failures are reported as failures.
