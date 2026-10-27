# Protocol 10: Explanation by manipulation

Written before the first measurement on held-out scenes.

## Claim tested

"Es geht nicht nur darum, Bitmaps vorherzusagen, sondern die Welt zu erklären, so dass man sie manipulieren kann" (Joscha Bach). A model that has explained a scene can say what it would look like if it were changed. Delete an object, and the model must know what was behind it: amodal completion, which needs a memory of the background. Move an object, and the model must know its whole shape, including parts hidden at the moment. A bitmap with no explanation behind it can only guess.

## Stimulus

- The sprite world, 256 × 192, no occluders.
- 40 held-out scenes, seeds 2000–2039. Scene `s` has `2 + s % 2` objects.
- The graph watches 240 frames with learning on.

## Procedure

At frame 240, for each true object that some node matches with visible-mask IoU ≥ 0.3 (one-to-one, greedy by IoU), make two edits:

- **delete** the object;
- **move** it 30 pixels to the right, or to the left if its centre is right of the middle.

**Truth:** the sprite world renders itself with the same edit.

**Graph:** `graph.edit(delete=...)` or `graph.edit(move=...)` on the matched node.

**Bitmap baseline:** the current frame, edited with the same node's mask (alpha > 0.5), so both use the same segmentation and differ only in what lies behind it.
- *Delete:* inpaint the mask (OpenCV Telea, radius 5).
- *Move:* inpaint the mask, then paste the masked pixels at the moved position.

**Scoring region:** the union of the object's full true mask before and after the edit, dilated by 2 pixels. All three images are scored by mean squared error against the truth inside it.

Objects without a match are counted and reported, not scored.

## Measures

- Mean MSE per edit type for the graph and for the baseline.
- The ratio graph / baseline per edit type.
- The fraction of edits where the graph beats the baseline.
- The number of objects matched.

## Pass criteria (declared in advance)

- **Delete:** the graph's mean MSE is at most 0.75 × the baseline's.
- **Move:** the graph's mean MSE is below the baseline's.
- At least 80% of objects are matched.

Failures are reported as failures.
