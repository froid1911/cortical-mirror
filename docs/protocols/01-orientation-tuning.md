# Protocol 1: Orientation tuning (V1 simple cells)

Written before the first measurement on a trained network.

## Claim tested

V1 cells become orientation-selective through local learning on natural input, as cells in the primary visual cortex are (Hubel & Wiesel, 1962). Many learning methods produce this on natural images, so it is a necessary first check, not the main result.

## Stimulus

- Full-field static sinusoidal gratings at the retina's resolution (128 × 96).
- Spatial period: 8 retinal pixels.
- Contrast: 0.8 around a mean luminance of 0.5.
- 12 orientations, 0° to 165° in 15° steps.
- 8 phases per orientation.

## Procedure

1. Encode every grating with the retina.
2. Settle V1 for 30 iterations from zero activity, with weights frozen. Nothing is learned during the test.
3. A cell's response to a stimulus is its mean activity over the central 8 × 8 V1 positions.
4. Average the responses over the 8 phases.

## Measures

- **OSI** per cell: |Σ_θ R(θ) e^{2iθ}| / Σ_θ R(θ) (one minus circular variance). Only cells whose summed response exceeds 1e-4 are included.
- **Median OSI**, and the **fraction selective** (OSI > 0.3).

## Conditions

1. The untrained network (seed 0).
2. Newborn: the untrained network after its experience (stated in each report).
3. The adult.

## Pass criteria (declared in advance)

- The learned network's median OSI exceeds the untrained network's by at least 0.2.
- At least half of its responsive cells have OSI > 0.3.

Failures are reported as failures.
