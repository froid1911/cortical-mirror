# Protocol 5: End-stopping

Written before the first measurement on a trained network.

## Claim tested

End-stopped V1 cells respond best to short bars and less as a bar extends beyond their receptive field (Hubel & Wiesel, 1965). Rao & Ballard (1999) reproduced end-stopping in a predictive-coding network trained on natural images and attributed it to top-down prediction of long, predictable contours. As with protocol 4, the same V1 weights are measured with and without the top-down term.

## Stimulus

- Bright bars (luminance 1.0) on mean grey (0.5), 2 retinal pixels wide.
- Centred on the middle of the retina, at each cell's preferred orientation: the bar runs along the cell's stripes, from protocol 1.
- Lengths 2, 4, 6, 8, 12, 16 and 24 retinal pixels.

## Procedure

As in protocol 4: frozen, settled with feedback and without, with the response read at the central 2 × 2 V1 positions.

## Measures

- **End-stopping index** per cell: (R_peak − R_longest) / R_peak.
- Its median over responsive cells, with and without feedback, and the fraction of cells with an index above 0.3.

## Conditions

1. The untrained phase 2 network.
2. The phase 2 adult.

## Pass criteria (declared in advance)

In the adult, the median end-stopping index **with** feedback exceeds the median **without** feedback.

Failures are reported as failures.
