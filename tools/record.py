"""Record webcam clips for a test set no model or setting has ever seen.

Clips are written as lossless FFV1 AVI at the camera's native size. Record them
before training or tuning anything on them, then score every engine with
`tools/benchmark.py --data <folder> --split all`.
"""

import argparse
import time
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="folder for the clips")
    parser.add_argument("--clips", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between clips")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    args.output.mkdir(parents=True, exist_ok=True)
    session = time.strftime("%Y%m%d-%H%M%S")
    try:
        for clip in range(1, args.clips + 1):
            print(f"clip {clip}/{args.clips} starts in {args.pause:.0f} s", flush=True)
            time.sleep(args.pause)
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("the camera stopped delivering frames")
            height, width = frame.shape[:2]
            path = args.output / f"webcam_{session}_c{clip:02d}.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height)
            )
            frames, deadline = 0, time.perf_counter() + args.seconds
            while time.perf_counter() < deadline:
                writer.write(frame)
                frames += 1
                ok, frame = capture.read()
                if not ok:
                    break
            writer.release()
            print(f"  wrote {frames} frames to {path}", flush=True)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
