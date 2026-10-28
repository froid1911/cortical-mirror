# Protocol 9: Object permanence

Written before the first measurement.

## Claim tested

Young infants look longer when an object that went behind a screen fails to reappear: they expected it to still exist (Baillargeon, Spelke & Wasserman, 1985). Before that age, out of sight is out of mind.

The scene level is given object files, but how long it keeps believing in a node it cannot see is one learned time constant, the **persistence** τ. The newborn starts with τ = 1 frame, so it forgets within about three frames.

τ learns from outcomes, a global signal applied to one shared parameter:
- **Up:** a forgotten node is recognised again (same look, same shape: re-identification is innate). It was forgotten too early, so τ moves toward the time it was unseen.
- **Down:** a forgotten node is never seen again within 150 frames. Belief in it would have been wasted, so τ moves toward 1.

The test asks whether learning gives the adult permanence that the newborn lacks, and whether the adult is then surprised when an object fails to reappear.

## Stimulus

**Test trials:** 20 seeds, 3000–3019. Each is the sprite world with:
- one small object (radius 7–9 before its shape harmonics), starting at (40, 96) and moving right at 1.5 px per frame;
- a 44-pixel-wide screen in the middle, wide enough to hide the object completely;
- 120 frames.

Each seed is run twice:
- **Reappear:** nothing is changed.
- **Vanish:** the object is removed from the world at the first frame it is completely hidden.

The **expected reappearance frame** t_r is the first frame after the hiding at which the object is visible again in the reappear trial.

**Training (adult):** seeds 100–299, 300 frames each, with persistence learning on. Each world has:
- `1 + seed % 3` objects of radius 7–12;
- one screen, 30–50 pixels wide (`30 + seed % 21`).

**Magic-world control:** the same training worlds, except that every object is removed from the world the first time it is completely hidden.

## Procedure

For every test trial, a fresh graph with the condition's τ watches the frames. τ is frozen during the test.

1. **Kept:** in the reappear trial, at frame t_r − 1, is there a node of any state within 8 pixels of the hidden object's true position?
2. **Position error:** that node's distance from the true position, for trials where a node was kept.
3. **Surprise:** before each frame, take the graph's expectation (`expect()`). Surprise is its mean squared difference from the frame:
   - over frames t_r to t_r + 14;
   - in the region from the screen's right edge to 36 pixels beyond it, rows 72–120.

## Measures

- Fraction kept.
- Median position error.
- Mean surprise in vanish trials and in reappear trials.
- **Violation ratio** = surprise(vanish) / surprise(reappear).
- τ after training, for the adult and for the magic-world control.

## Conditions

1. **Newborn:** τ = 1.
2. **Adult:** τ learned in training.
3. **Built-in permanence (reference):** τ = 60, set by hand. It shows what perfect keeping gives; it is not a learned result.
4. **Magic world:** reported as τ only.

## Pass criteria (declared in advance)

- **Adult:** kept ≥ 0.8; median position error ≤ 4 px; violation ratio ≥ 2.
- **Newborn (developmental contrast):** kept ≤ 0.2 and violation ratio < 1. That is, it is surprised by the reappearance, not by the vanishing.
- **Learned from the world:** the magic-world τ is less than half the adult's.

Failures are reported as failures.
