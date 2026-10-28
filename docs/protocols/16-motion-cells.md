# Protocol 16: Motion cells

Written before the first measurement.

## Claim tested

Direction selectivity in visual cortex is not present at eye opening. It develops with experience of motion (Li, Fitzpatrick & White, 2006; Li et al., 2008). Motion-sensitive cells (MT) respond to motion in one direction and not the opposite.

In v0.7, an MT level has motion cell types competing to explain how V1's activity changes from one frame to the next. Each type learns, with a local delta rule toward the future, a non-negative spatial kernel saying where activity goes next. Nothing about direction is given. The test asks whether experience makes the types become direction-selective, cover the directions, and let the direction of motion be read from them.

## Stimuli

- **Drifting gratings** through phase 1's V1:
  - 12 directions;
  - period 8 retinal pixels;
  - 1 and 2 pixels per frame;
  - 16 frames each.
- **Translating 1/f textures:**
  - 8 directions × speeds of 1, 2 and 4 retinal px per frame;
  - 20 frames each;
  - 10 textures (seeds 0–9).

## Procedure

1. Run the frames through V1 (frozen) and MT (frozen), and record each type's responsibility in the central region (central 16 × 12 V1 positions), averaged over frames 3 onwards.
2. **Direction selectivity per type:** DSI = (R_pref − R_opp) / (R_pref + R_opp) over the 12 grating directions, at the speed where the type responds most. Types with a peak responsibility below 0.05 are excluded.
3. **Coverage:** the direction of each type's learned displacement (the centre of mass of its kernel), for types whose displacement is at least 0.5 V1 positions long, binned into 8 compass sectors.
4. **Decoding:** on the textures, the population velocity (Σ responsibility × displacement) in the central region. Its direction is compared with the true direction.

## Measures

- Fraction of types with DSI > 0.5.
- Number of compass sectors covered.
- Median direction error of the decoded velocity, in degrees, over all texture trials.

## Conditions

1. **Untrained:** MT kernels at their newborn start (near "no change", small random differences).
2. **Adult:** MT raised on UCF101 training video.

## Pass criteria (declared in advance)

- **Adult:**
  - fraction of types with DSI > 0.5 ≥ 0.5;
  - ≥ 6 of 8 sectors covered;
  - median decoding error ≤ 30°.
- **Untrained:** fraction with DSI > 0.5 ≤ 0.1. Direction must come from experience.

## Amendment after a development check, before any adult measurement (2026-10-28)

A development check exposed a flaw in the direction-selectivity criterion. Compared were the untrained network and a 3-minute smoke-trained MT (the phase 1 V1 adult, textures 100–102, not the protocol's 0–9):
- the untrained network's fraction of types with DSI > 0.5 was **0.94**;
- the smoke network's was 0.81.

The competition between types makes every type's responsibility depend strongly on direction, even when its kernel is random. The DSI of responsibilities is therefore high by construction and cannot show learning. The untrained criterion (≤ 0.1) can never hold.

The other two measures showed the learning plainly:
- sectors covered: 0 untrained, 8 after 3 minutes;
- median decoding error: 66° untrained, 2.6° after 3 minutes.

**Amended criterion** (replacing "fraction of types with DSI > 0.5"):

- **Tuning agreement:** for each responsive type with a learned displacement at least 0.5 positions long, does its preferred grating direction lie within 45° of the direction of its own displacement? Grating direction θ moves the pattern along (cos θ, sin θ) in image coordinates (y down), the same frame as the displacement.
- **Pass:** a fraction of agreeing types ≥ 0.5 in the adult. In the untrained network, types have displacements shorter than 0.5, so they do not count; its fraction is reported.

The original DSI fraction is still reported for both conditions; under the original criterion the untrained condition fails. The sector and decoding criteria are unchanged.

Failures are reported as failures.
