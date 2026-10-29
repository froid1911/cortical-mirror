# Protocol 23: Object files learn sharp shapes

Written before the first measurement.

## Claim tested

What moves with an object is the object. In an object's own frame, its pixels stay the same while the background slides past. An object file that keeps a running mean and variance at every place of its frame (`cortex/object_files.py`) learns the object's shape where the variance stays low. Following its object by the velocity its slot reports, and correcting its position by its own learned appearance, it gives masks sharper than the slots' V1 grid allows.

Protocols 20 and 21 failed their object IoU criterion (0.30 against 0.5). The development checks after them showed that frame pairs cannot reach 0.5 at V1's resolution. Object files integrate over time instead.

## Development (training seeds only)

- On seeds 950–953 the learned scene's pixel object IoU rose from 0.36 without files to 0.55 with them.
- On seeds 940–943 four settings were compared: the defaults (variance below 0.25 × the border's, search ±3 px, forgetting rate 0.05, mature after 30 frames) gave 0.49; ratio 0.5 gave 0.36, ratio 0.15 0.46, search ±5 px 0.41.
- The defaults are fixed. The criterion is expected to be met narrowly or missed narrowly.

## Stimulus

- **Moving scenes:** the sprite world, 3 objects, no occluders, **test seeds 9200–9239** (40 scenes), 300 frames; scored from frame 100, every 20th frame.
- **Manipulation:** protocol 22's setup on **fresh seeds 2300–2339**.

## Conditions

1. Learned scene with object files (MT's fast path, protocol 21's links).
2. The same without object files (protocol 22's scene).
3. With object files, MT silenced (zero velocity).

## Measures

- **Pixel object IoU:** objects' visible masks against the scene's masks (a mature file's shape, else the slot's region), matched one-to-one, greedily (protocol 19's measure, on pixels).
- **Manipulation:** protocol 22's delete and move MSE.

## Pass criteria (declared in advance)

1. With object files, mean pixel object IoU ≥ 0.5 on the moving scenes.
2. MT silenced: mean pixel object IoU ≤ 0.1.
3. Manipulation, move: mean MSE with object files below the mean MSE without them.

Reported without a criterion: the IoU without files; delete MSE with and without files; the fraction of objects matched.

Failures are reported as failures.
