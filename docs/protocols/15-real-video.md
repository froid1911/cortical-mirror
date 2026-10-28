# Protocol 15: Real video with a moving camera

Written before the first measurement.

## Claim tested

In v0.4, on real video, the graph's expectation of the next frame was 1.9 × worse than simply repeating the last frame. Much real video is filmed by a moving camera, and v0.4 took every pan for the whole world moving. v0.5 registers the camera's motion against a canvas, describes objects in the cortex's code, and predicts only what it explains (elsewhere, the last frame). Does it now predict real video better than doing nothing?

## Stimulus

- UCF101 test split (official split 1).
- 40 videos, chosen by a fixed seed (0) from the sorted test list.
- The first 150 frames of each, at 256 × 192.
- Development uses the validation split only.

## Procedure

Each condition watches each clip from a fresh start. For frames 50–149, score:

- the expectation the graph forms before the frame, against the frame (mean squared error, luminance);
- the last frame, against the frame.

## Measures

- **Ratio:** expectation MSE over last-frame MSE, pooled over all scored frames of all clips.
- The fraction of clips where the ratio is below 1.

## Conditions

1. **v0.4:** the scene graph, luminance, no registration.
2. **Registration, luminance:** the video graph with camera registration, matching on brightness.
3. **Registration, cortex code:** the video graph as built for v0.5.

## Pass criteria (declared in advance)

- Condition 3's ratio is below 0.95.

All three are reported.

Failures are reported as failures.
