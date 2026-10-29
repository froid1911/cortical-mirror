# Protocol 18: Self and world in learned motion

Written before the first measurement.

## Claim tested

When the eye moves, the image shifts, yet the world is seen as still. When the same shift happens without a command, the world is seen as moving (Helmholtz; von Holst & Mittelstaedt). Protocol 12 showed this with an engineered motor map and registration. Here it must happen inside the learned motion code:
- v0.7's MT motion cells sense the shift;
- an **efference map**, learned by Hebbian averaging, says which motion types each eye command makes active;
- the motion a command explains is the self part, and whatever remains is the world's.

## Stimulus and procedure

- Protocol 12's sprite world: 384 × 288, 2 moving objects, view 256 × 192, sent to the cortex at 128 × 96.
- Saccades of 8 view pixels (4 retinal pixels) in 8 directions.
- **Training** (only the efference map learns; MT is the frozen v0.7 adult):
  - seeds 800–839, 600 frames each;
  - a random small saccade every 5 frames, in a direction that keeps the view inside the world.
- **Test,** seeds 8000–8019:
  1. Watch 30 frames with the eye still.
  2. Make one saccade in a random (seeded) direction.
  3. **Commanded:** the command is told. **Uncommanded:** the same view shift, with command 0.
  4. For each, world motion = the median over the central region (half the view in each dimension) of |sensed velocity − self-caused velocity expected for the command told|.
  5. Divide by the true shift: 4 retinal px, or 4√2 for diagonals.

## Conditions

1. **Adult efference map** (trained).
2. **Newborn efference map** (uniform).

Both use the v0.7 MT adult.

## Measures

- Commanded and uncommanded world-motion fractions.
- The direction error of the sensed velocity against the true view shift: a check that MT senses the saccade at all.

## Pass criteria (declared in advance)

- **Adult map:** commanded fraction ≤ 0.3, and uncommanded fraction ≥ 0.7.
- **Newborn map:** commanded fraction ≥ 0.7.

Failures are reported as failures.
