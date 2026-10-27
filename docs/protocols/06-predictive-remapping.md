# Protocol 6: Predictive remapping (efference copy)

Written before the first measurement on a trained network.

## Claim tested

Before a saccade, the brain sends a copy of the motor command to visual areas. This lets it anticipate how the image will shift: neurons start responding to what will fall into their receptive field after the eye moves (Duhamel, Colby & Goldberg, 1992; Sommer & Wurtz, 2006). Here the cortex has one learned temporal kernel per eye movement and is told which movement is about to happen. Nothing about the resulting image shift is built in. The test asks whether the cortex learned to use the copy: is its prediction of the post-saccade frame better with the copy than without it?

## Stimulus

- Static worlds (256 × 192) of natural-statistics (1/f) texture, 8 different seeds.
- The eye starts at the centre.

## Procedure

For each world and each of the 8 saccade directions, with learning frozen:

1. Settle V1 from rest on the centre view. The view is steady, so the lagged channels hold the same image.
2. Move the eye one saccade in that direction. Settle V1 from rest on the new view; the lagged channels now hold the pre-saccade view.
3. Carry the pre-saccade activity forward through two kernels:
   - **told:** the learned temporal kernel of that movement (the efference copy);
   - **not told:** the kernel of "no movement".
4. Score each against the actual post-saccade activity, as the mean squared error over all cells and positions.
5. **Direction control:** also forecast the same transition with the kernel of the *opposite* movement. A kernel that merely damps activity after any saccade would beat "no movement" without remapping anything. It would do just as well with the wrong direction, so it fails this control.

Every activity is settled from rest in the same way, so only the learned kernel differs between the two forecasts. Remapping is a property of neural responses, so the score is taken in activity space, as in physiology. An earlier draft scored in retinal-code space and from the running state; there, reconstruction error and differences in how settling starts swamped the effect. Ideal shift kernels scored about 0.91 whichever direction they shifted. This amendment was made before any measurement.

## Measures

- **Remapping ratio** = mean error with the copy / mean error without it, over all worlds and directions. Below 1 means the copy helps.
- The same ratio per direction.
- **Direction specificity** = mean error with the movement's own kernel / mean error with the opposite movement's kernel. Below 1 means the kernels encode which way the eye moved.

## Conditions

1. The untrained phase 3 network. Every kernel starts as "no change", so the ratio is 1 by construction.
2. Newborn after its experience.
3. The phase 3 adult.

## Pass criteria (declared in advance)

In the adult, both:

- the remapping ratio is below 0.95;
- the direction specificity is below 0.95.

The direction control was added after a review found that a uniformly damping kernel could pass the ratio alone. It was added before any adult measurement.

**Ceiling, measured before any adult measurement.** On an untrained V1, ideal kernels that shift activity exactly as the image shifts score 0.895. The same kernels shifting the wrong way score 1.474. Perfect remapping cannot reach 0 here, because the lagged channels still hold the pre-saccade view. The first draft's criterion, below 0.8, would have been unreachable even for perfect remapping, so it was amended before any measurement.

Failures are reported as failures.
