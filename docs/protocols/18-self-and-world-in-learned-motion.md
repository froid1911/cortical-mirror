# Protocol 18: Self and world in learned motion

Written before the first measurement.

## Claim tested

When the eye moves, the image shifts, yet the world is seen as still. When the same shift happens without a command, the world is seen as moving (Helmholtz; von Holst & Mittelstaedt). Protocol 12 showed this with an engineered motor map and registration. Here it must happen inside the learned motion code:
- v0.7's MT motion cells sense the shift;
- an **efference map**, learned by Hebbian averaging, says which motion types each eye command makes active;
- the motion a command explains is the self part, and whatever remains is the world's.

## Stimulus and procedure

- Protocol 12's sprite world: 384 × 288, 2 moving objects, view 256 × 192, sent to the cortex at 128 × 96.
- Saccades of 4 view pixels (2 retinal pixels; amended, see below) in 8 directions.
- **Training** (only the efference map learns; MT is the frozen v0.7 adult):
  - seeds 800–839, 600 frames each;
  - a random small saccade every 5 frames, in a direction that keeps the view inside the world.
- **Test,** seeds 8000–8019:
  1. Watch 30 frames with the eye still.
  2. Make one saccade in a random (seeded) direction.
  3. **Commanded:** the command is told. **Uncommanded:** the same view shift, with command 0.
  4. For each, world motion = the median over the central region (half the view in each dimension) of |sensed velocity − self-caused velocity expected for the command told|.
  5. Divide by the true shift: 2 retinal px, or 2√2 for diagonals (amended).

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

## Amendment after a development check, before any test measurement (2026-10-29)

The development check used training seeds 850–859 and test seeds 8100–8109, not the protocol's. With saccades of 8 view pixels (2 V1 positions), MT sensed the shift's direction (error 7.5°) but only about 0.56 of its size:
- adult map: commanded 0.18, uncommanded 0.56;
- newborn map: 0.56.

The cause is the v0.7 MT adult's range. Its learned types move by one V1 position in all 8 directions, but by two positions only along three axes and one diagonal, because UCF101 rarely shows faster motion. A sharper read-out changed nothing (0.58).

The question here is whether the eye's command separates self from world, not MT's speed range. So the saccade size is amended to **4 view pixels** (2 retinal pixels, one V1 position), which the adult represents in every direction. Fractions are divided by the true shift of 2 retinal pixels (2√2 for diagonals). With 4-pixel saccades on the same development seeds:
- adult map: commanded 0.017, uncommanded 0.81;
- newborn map: 0.81;
- sensed direction error: 0.9°.

The criteria are unchanged.

Failures are reported as failures.
