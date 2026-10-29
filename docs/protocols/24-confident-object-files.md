# Protocol 24: Object files that know how sure they are

Written before the first measurement.

## Why

Protocol 23 passed as written, but its object files lost their objects: on training seeds 10 of 17 mature files sat off their object. Fewer objects were matched (67% against 85%), and deleting got worse. The move comparison was over different objects.

## What changed (development, training seeds 940–953 and development seeds 2200–2209 only)

- **A file knows how sure it is.** Its fit is the error of its prediction relative to the error of the rival explanation: the remembered background where the scene knows it, else an unrelated template. A file that drifted onto background learns it and explains it well, but no better than the background does, so it is not trusted.
  - Development showed this matters. Measuring fit against the file's own variance alone was unstable, and lost files were trusted.
- A mature file that cannot explain what it sees (fit ≥ 0.5) does not learn from it.
- A file that is not sure looks for its object again within 8 px of its slot's centre.
- The scene uses a file's sharp mask only while the file is sure; otherwise the slot's coarse mask.
- **On development seeds:**
  - pixel object IoU 0.58 on both seed groups (940–943, 950–953);
  - manipulation on 2200–2209: 88% matched, delete 0.12 × and move 0.55 × the bitmap. Without files: 92%, 0.19 × and 0.60 ×.

## Stimulus

- **Moving scenes:** the sprite world, 3 objects, **test seeds 9300–9339**, 300 frames, scored from frame 100, every 20th.
- **Manipulation:** protocol 22's setup on **test seeds 2400–2439**. Each scene is run twice, with and without object files. The two are compared on the objects both matched (paired by seed and object).

## Pass criteria (declared in advance)

1. Pixel object IoU with object files ≥ 0.5.
2. With object files, at least 80% of objects matched (protocol 22's bar).
3. Delete: with object files, mean MSE ≤ 0.75 × the bitmap's, and below the mean MSE without files on the same objects.
4. Move: with object files, mean MSE below the mean MSE without files on the same objects.
5. MT silenced: pixel object IoU ≤ 0.1.

Reported without a criterion: IoU without files; unpaired means; the fraction of objects where files do better.

Failures are reported as failures.
