# Protocol 14: Objects under changing light

Written before the first measurement.

## Claim tested

Seeing is largely indifferent to overall lighting, because the retina and early cortex code contrast, not brightness. v0.4's graph compared raw brightness. v0.5 describes objects in the cortex's code: the retina's normalised ON/OFF contrast, and the v0.3 adult's V1 activity. If that is the right description, objects should survive a change of light that breaks the brightness graph.

## Stimulus

- The sprite world (256 × 192, 3 objects, no occluders), 300 frames.
- Under a slow global change of light: frame t is shown as `gain(t) · image + offset(t)`, clipped to [0, 1].
  - `gain = 1 + 0.4 sin(2πt/100)`;
  - `offset = 0.1 sin(2πt/70)`.
- **Test seeds:** 7000–7019. **Development seeds:** 700–719.

## Procedure

As in protocol 8. From frame 200 on, every 10th frame, nodes are matched to the visible objects one-to-one by IoU.

## Measures

- Mean IoU and count accuracy under changing light.
- The same under constant light (no regression).

## Conditions

1. **Cortex code:** the graph as built for v0.5.
2. **Luminance:** the same graph matching on brightness, as in v0.4.

## Pass criteria (declared in advance)

- **Changing light:** the cortex graph's mean IoU is ≥ 0.6, and at least 0.15 above the luminance graph's.
- **Constant light:** the cortex graph's mean IoU is no more than 0.05 below the luminance graph's.

Failures are reported as failures.
