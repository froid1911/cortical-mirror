# Protocol 4: Surround suppression

Written before the first measurement on a trained network.

## Claim tested

A V1 cell's response to a grating at its preferred orientation grows with the grating's size up to its receptive field, then falls as the grating extends into the surround (size tuning; e.g. Cavanaugh, Bair & Movshon, 2002). Predictive coding explains this as top-down prediction: a large uniform pattern is predictable from the higher level, so V1's unexplained activity shrinks (Rao & Ballard, 1999). The test asks whether the suppression comes from the learned feedback. The same V1 weights are therefore measured with and without the top-down term.

## Stimulus

- Circular grating patches centred on the middle of the retina, on mean grey (0.5).
- The grating runs at each cell's preferred orientation, taken from protocol 1's tuning curves.
- Period 8, contrast 0.8, 8 phases averaged.
- Diameters 4, 6, 8, 12, 16, 24, 32 and 48 retinal pixels.

## Procedure

1. The network is frozen during the test.
2. Settle the whole hierarchy for 30 iterations from zero activity, either:
   - with feedback: V1 is pulled toward V2's prediction; or
   - without feedback: the top-down term is removed, and V1 settles alone with the same weights.
3. A cell's response is its activity at the central 2 × 2 V1 positions, averaged over phases.

## Measures

- **Suppression index** per cell: (R_peak − R_largest) / R_peak.
- Its median over cells whose peak response exceeds 1e-4, with feedback and without.

## Conditions

1. The untrained phase 2 network.
2. The phase 2 adult. Newborn results are reported where available.

## Pass criteria (declared in advance)

In the adult, the median suppression index **with** feedback exceeds the median **without** feedback by at least 0.1.

Failures are reported as failures.
