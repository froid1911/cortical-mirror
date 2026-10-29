# Protocol 21: Objects held by links relearned from integrated motion

Written before the first measurement.

## Why a new protocol

Protocol 20 failed criteria 1 and 3 on test seeds 9000–9019. Afterwards, on training seeds only, the learned links were retrained with the motion the slots use: MT comparing V1 four frames apart, instead of frame to frame. On held-out training seeds 950–953 the slots then kept 88% of their IoU through the still moment (72% with the old links), with moving IoU unchanged (0.295). Protocol 20's test seeds have been seen, so the change is judged on fresh ones.

## What changes from protocol 20

- **Links:** protocol 19's rule, learned from MT integrated over 4 frames (smoothing 3), on training seeds 900–949, 300 frames each (`tools/train_links.py`).
- **Test seeds 9100–9119.**

Everything else is protocol 20's, amendments included:
- stimuli: moving scenes, and a 30-frame still moment from frame 150;
- slot settings: σ = 0.25, β = 2, blur 0.2, 3 EM rounds, merge tolerance = σ;
- MT read-out: gap 4, smoothing 3;
- measures and conditions.

## Pass criteria (declared in advance, as in protocol 20)

1. Moving scenes: the slots' mean object IoU ≥ 0.5.
2. Silenced MT: mean object IoU ≤ 0.1.
3. Still moment, with links: mean IoU at the end of the pause ≥ 0.8 × the mean IoU before it.

**Expected from development:** criterion 1 fails. After protocol 20, a perfect "is it moving" map with MT's coverage and spill bounds the IoU near 0.29 (see the cortex report). Criteria 2 and 3 are expected to pass.

Failures are reported as failures.
