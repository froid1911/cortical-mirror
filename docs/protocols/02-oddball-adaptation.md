# Protocol 2: Oddball, repetition suppression and prediction error

Written before the first measurement on a trained network.

## Claim tested

Predictive coding says a repeated stimulus becomes expected, so its error response shrinks (repetition suppression), and that a stimulus which violates a learned regularity produces extra error beyond its mere rarity (prediction error). Both effects are found in sensory cortex. Parras et al. (2017) separate them with a many-standards control. This protocol uses the same decomposition.

## Stimulus

Static gratings (period 8, contrast 0.8, phase 0). Each stimulus is shown for 4 frames, followed by the next stimulus without a blank. There are 200 stimulus presentations per sequence.

- **Oddball sequence:** a standard at 0° with probability 0.9 and a deviant at 90° with probability 0.1, in a pseudo-random order (seed 0). There are no two deviants in a row, and the sequence opens with at least 10 standards.
- **Many-standards control:** 10 orientations (0°, 18°, …, 162°, which includes 90°), each with probability 0.1. So 90° appears exactly as often as in the oddball sequence, but nothing is a standard.

## Procedure

1. Start each sequence from a copy of the network under test, so no learning carries over between sequences.
2. Learning stays on during the sequence, because adaptation is learning.
3. The response to a presentation is the network's **temporal prediction error** on the first frame of that stimulus: the mean squared difference between the predicted and actual retinal code.

## Measures

The three mean onset responses, after the first 20 presentations:

- **DEV:** 90° as a deviant;
- **STD:** 0° as a standard;
- **CTR:** 90° in the control.

With N = √(DEV² + STD² + CTR²):

- **iRS** = (CTR − STD) / N: repetition suppression;
- **iPE** = (DEV − CTR) / N: prediction error beyond rarity;
- **iMM** = (DEV − STD) / N = iRS + iPE: mismatch.

## Conditions

1. The untrained network.
2. Newborn after its experience.
3. The adult.

## Pass criteria (declared in advance)

- iMM > 0 and iRS > 0 for the learned network. iPE is reported whatever its sign.
- For each index, the learned network's value is compared with the untrained one. The adaptation must exceed what an unlearned network shows.

Failures are reported as failures.
