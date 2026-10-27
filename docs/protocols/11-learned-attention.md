# Protocol 11: Attention learned from learning progress

Written before the first measurement.

## Claim tested

v0.3's eye had the reflex "look toward surprise" built in. It failed protocol 7: it stayed captured by motion it could never learn from. Curiosity in infants and in models of it (Oudeyer, Kaplan & Hafner, 2007; Schmidhuber, 1991) is better described as seeking *learning progress*: places where looking reduces prediction error. Pure noise is maximally surprising forever, and teaches nothing.

The scene graph's eye values each possible saccade target by learned weights over what the graph knows about the place:

- **unknown:** how unsure its background is;
- **error:** how badly it has been predicting there;
- **novelty:** how new the objects there are;
- **familiar:** how long the fovea has already spent there.

After each fixation, the weights change by a three-factor rule toward the progress the fixation brought (the relative drop of the prediction error in the fovea). The newborn's weights are zero, so it looks at random. The test asks whether learning gives an eye that avoids unlearnable noise, which a surprise reflex cannot, and that finds new things faster than chance.

## Stimulus

- Sprite worlds of 384 × 288, three moving objects, view 256 × 192, fovea 96 × 96 (sharp; the rest of the view blurred), saccades of 32 pixels every 10 frames.
- **TV noise:** a 96 × 96 patch of fresh random luminance every frame, centred on one of the four quadrant centres (chosen by seed).
- **A novel object:** a fourth object enters at frame 300, placed at the quadrant centre farthest from the gaze at that moment (never on the noise).
- 600 frames per trial.
- **Test:** seeds 5000–5019.
- **Training (adult):** seeds 500–539 with the same kind of world. Learning is on, and 20% of choices are random (exploration).

The motor map is the true one, as if learned already (protocol 12). This protocol is about where to look, not about telling self from world.

## Procedure

Each condition runs every test trial with its weights frozen. It records:

- each fixation's fovea in world coordinates (the experimenter knows the true gaze);
- the graph's background canvas at the end;
- the first fixation after frame 300 whose fovea contains the novel object's centre.

## Measures

- **Noise capture:** the fraction of fixations whose fovea overlaps the noise patch by at least half.
- **World error:** the mean squared error between the graph's background (mapped to world coordinates) and the true world background, over the world outside the noise patch, at frame 600.
- **Time to novelty:** the median number of frames from frame 300 until the novel object is first fixated (300 if never).

## Conditions

1. **Newborn:** zero weights (random gaze).
2. **Surprise reflex:** looks where the prediction error is largest (v0.3's rule).
3. **Adult:** weights learned in training.

## Pass criteria (declared in advance)

- **Setup check:** the surprise reflex's noise capture is at least 1.5 × the newborn's. If not, the noise is not a trap, and the test says nothing; this is reported.
- **Adult:**
  - noise capture at most 0.5 × the newborn's;
  - world error at most 0.9 × the newborn's;
  - median time to novelty at most 0.5 × the newborn's.

Failures are reported as failures.
