# Protocol 17: Learned motion, used to predict real video

Written before the first measurement.

## Claim tested

v0.6 predicted real video from moving parts, using engineered optical flow as its motion sense. v0.7 replaces the flow with motion read from learned MT cells. The test asks:
- whether the learned motion is used and good enough to beat repeating the last frame;
- whether the prediction depends on it, so that silencing MT breaks it.

## Stimulus

- UCF101 test split (official split 1).
- 40 clips drawn with seed 2, excluding the clips of seeds 0 and 1 (used by protocol 15's runs).
- The first 150 frames of each, at 256 × 192.

## Procedure

As protocol 15. Each condition watches each clip from a fresh start. For frames 50–149, its expectation of the next frame and the last frame are both scored against the frame (MSE, brightness).

## Conditions

1. **Learned motion (v0.7):** the video graph's part motion, with velocity read from MT (the adult). No other motion signal enters the prediction.
2. **MT silenced:** the same, with every responsibility uniform, which yields velocity zero everywhere.
3. **Optical flow (v0.6, yardstick):** the same graph with Farnebäck flow.

## Measures

The ratio of expectation MSE to last-frame MSE, pooled over clips, per condition; the fraction of clips below 1.

## Pass criteria (declared in advance)

- Learned motion: ratio < 1.0.
- MT silenced: ratio ≥ 0.99. Without MT, nothing predicts motion.

The optical-flow yardstick is reported without a criterion. v0.7 succeeds even if learned motion is worse than the flow.

Failures are reported as failures.
