# Protocol 12: Telling self-caused change from the world moving

Written before the first measurement.

## Claim tested

When the eye moves, the image on the retina shifts, yet the world looks still. When the eye is moved passively, for instance pushed with a finger, the same retinal shift looks like the world jumping (Helmholtz, 1867; von Holst & Mittelstaedt, 1950). The difference is the efference copy: the brain predicts the consequences of its own commands and subtracts them.

In the scene graph, the eye is a node, and a learned motor map says how far each command moves the gaze. Any shift the map did not predict is attributed to the world moving. The newborn's map is all zeros. This protocol asks two questions:

- **Attribution:** does learning the map make the graph tell self-caused shifts from the world moving?
- **Prediction from memory:** before a saccade lands, does the graph know what it will see there?

## Stimulus

- Sprite worlds of 384 × 288 with 2 moving objects. The view is 256 × 192, and saccades are 32 pixels in 8 directions.
- **Test:** 20 seeds, 4000–4019.
- **Adult training:** 40 seeds, 400–439, 600 frames each. A saccade is made every 5 frames, in a direction chosen at random among those that keep the view inside the world.

## Procedure

For each test seed:

1. A graph with the condition's motor map watches 30 frames with the gaze at the world's centre. The map is frozen during the test.
2. Pick a saccade direction at random (seeded).
3. **Commanded:** move the gaze and report the command.
4. **Uncommanded:** from an identical copy of the graph, move the gaze by the same amount but report command 0 ("no movement"). The retinal input is identical.
5. In both, record how far the graph believes the world moved on that frame (`world_shift`), as a fraction of the true shift's length.
6. **Prediction from memory:**
   - Before the commanded saccade, the graph predicts the coming view: its canvas at the gaze its motor map expects.
   - Score the mean squared error against the actual view, over the pixels the canvas is confident about (confidence > 0.5).
   - The same is scored for the prediction "the view does not change" (the current view).

## Measures

- Commanded world-motion fraction: 0 means the world was seen as still, 1 as having moved by the whole shift.
- Uncommanded world-motion fraction.
- Prediction error from memory, divided by the no-change prediction's error.

## Conditions

1. **Newborn:** motor map zero.
2. **Adult:** motor map learned in training.

## Pass criteria (declared in advance)

- **Adult:** commanded fraction ≤ 0.1; uncommanded fraction ≥ 0.8; prediction ratio ≤ 0.5.
- **Newborn (confuses self and world):** commanded fraction ≥ 0.8.

Failures are reported as failures.
