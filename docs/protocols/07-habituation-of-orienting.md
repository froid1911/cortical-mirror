# Protocol 7: Habituation of orienting

Written before the first measurement on a trained network.

## Claim tested

Attention and gaze are drawn to what is unexpected, not merely to what changes. A stimulus that keeps changing in a predictable way loses its pull (habituation of the orienting response; Sokolov, 1963), while unpredictable onsets keep it (Itti & Baldi, 2009, "Bayesian surprise").

In this system the reflex "look toward surprise" is built in. What is learned is what counts as surprising: the periphery's input minus the change it has learned to predict. If learning works, steady predictable motion should stop attracting the eye, and unpredictable onsets should take over. Because the eye is built to seek surprise, the plan's original measure ("fixations land on high surprise") would hold by construction, so this protocol replaces it.

## Stimulus

A 256 × 192 world, 600 frames:

- **Left half:** a 1/f texture panning steadily, 2 pixels per frame. It changes everywhere, all the time, but predictably.
- **Right half:** a static 1/f texture on which a bright 12 × 12 square appears for 3 frames at a random position, at random times (on average every 15 frames, seed 0). It changes rarely, but unpredictably.

## Procedure

1. Start the eye at the centre.
2. Run the phase 3 engine through the sequence with learning on, because habituation is learning. Each condition runs on a copy, so nothing carries over.
3. Record every saccade's horizontal direction.

## Measures

- **Fraction of saccades toward the onsets** (rightward) among all horizontal saccades, in:
  - the first 200 frames;
  - the last 300 frames.
- The same for a control: the untrained network with learning off, which has no habituation and whose surprise is raw change.

## Conditions

1. The untrained network with learning off (control).
2. The untrained network with learning on (newborn).
3. The phase 3 adult with learning on.

## Pass criteria (declared in advance)

- In the last 300 frames, the learning network's fraction of saccades toward the onsets exceeds the control's by at least 0.1.
- This must hold for the newborn or the adult.

Both are reported, and failures are reported as failures.
