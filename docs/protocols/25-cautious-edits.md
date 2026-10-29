# Protocol 25: Remove what could belong, paste what surely does

Written before the first measurement.

## Why

Protocol 24 failed one criterion. With object files, deleting was better on 61% of objects but worse on average (0.0017 against 0.0011 without files). When a sure file's sharp shape misses part of its object, deleting by that shape leaves the missed part behind.

## What changed

An edit now removes all that could belong to the object: a sure file's shape *and* the slot's coarse region. What it pastes elsewhere when moving is only what surely belongs: the sure shape. Nothing else changed: masks for matching and IoU, files, slots, MT and links are as in protocol 24.

Development on seeds 2200–2209 (protocol 24's development seeds) gave delete 0.0003 against 0.0015 without files (paired, 21 objects), 0.10 × the bitmap; move 0.0015 against 0.0043. Development seeds have been optimistic twice (protocols 23 and 24), so the test decides.

## Stimulus

Protocol 24's manipulation on **fresh seeds 2500–2539**: each scene run with and without object files, compared on the objects both matched.

## Pass criteria (declared in advance, protocol 24's criteria 2–4)

1. With object files, at least 80% of objects matched.
2. Delete: with object files, mean MSE ≤ 0.75 × the bitmap's, and below the mean MSE without files on the same objects.
3. Move: with object files, mean MSE below the mean MSE without files on the same objects.

Failures are reported as failures.
