# Protocol 3: Direction selectivity

Written before the first measurement on a trained network.

## Claim tested

Many V1 cells respond to motion in one direction and much less to the opposite direction (Hubel & Wiesel, 1962). Spatiotemporal learning on natural video can produce such cells without being told about motion (van Hateren & Ruderman, 1998; Olshausen, 2003). Here direction must emerge from local learning over the retina's current and lagged channels. The lagged channels model lagged LGN cells, and they are the only built-in temporal structure.

## Stimulus

Drifting sinusoidal gratings:
- period 8 retinal pixels, contrast 0.8;
- 12 directions, 0°–330° in 30° steps;
- moving 1 pixel per frame for 16 frames;
- presented through the lagged retina, so every frame carries the current and the previous image.

## Procedure

1. The network is frozen during the test.
2. For every frame, settle V1 for 30 iterations from zero activity.
3. A cell's response to a direction is its mean central activity (the central 8 × 8 V1 positions) over frames 3–16. The first two frames are skipped while the lag fills.

## Measures

- **DSI** per cell: (R_pref − R_opp) / (R_pref + R_opp). R_pref is the response to the cell's best direction and R_opp the response to the direction 180° away. Only cells whose best response exceeds 1e-4 are included.
- **Median DSI**, and the **fraction selective** (DSI > 0.3).

## Conditions

1. The untrained phase 2 network.
2. Newborn after its experience.
3. The phase 2 adult.

## Pass criteria (declared in advance)

- The learned network's fraction of direction-selective cells (DSI > 0.3) is at least 0.2.
- That fraction is higher than the untrained network's.

Failures are reported as failures.
