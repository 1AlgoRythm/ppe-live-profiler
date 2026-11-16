"""Live UI: OpenCV window with boxes, FPS and a per-stage latency bar.

Keys: v = fp32/int8, s = 320/640, t = threaded capture on/off, q = quit.
"""
import argparse
import sys
import time

import cv2

from .pipeline import Cascade
from .runner import FrameLoop
from .sources import Source


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/test_video.mp4", help="0 for webcam, or a video file")
    ap.add_argument("--variant", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--size", type=int, choices=[320, 640], default=640)
    ap.add_argument("--threaded", action="store_true")
    ap.add_argument("--no-display", action="store_true", help="run the loop without a window (testing)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    ap.add_argument("--keys", default="", help="with --no-display: simulate key presses, one per frame")
    a = ap.parse_args(argv)

    src = Source(a.source, loop=True)
    cascades = {}

    def get(variant, size):                                 # sessions are built once, on first use
        if (variant, size) not in cascades:
            cascades[(variant, size)] = Cascade(variant, size)
        return cascades[(variant, size)]

    variant, size = a.variant, a.size
    src.set_threaded(a.threaded)
    loop, n = FrameLoop(), 0
    while True:
        label = f"{variant} {size} {'threaded' if src.threaded else 'sync'}  [v]ariant [s]ize [t]hread [q]uit"
        out = loop.step(src, get(variant, size), label)
        if out is None:
            break
        shown = out[0]
        n += 1
        if a.no_display:
            key = ord(a.keys[n - 1]) if n <= len(a.keys) else -1
        else:
            cv2.imshow("ppe-live-profiler", shown)
            key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or (a.max_frames and n >= a.max_frames):
            break
        if key == ord("v"):
            variant = "int8" if variant == "fp32" else "fp32"
        elif key == ord("s"):
            size = 320 if size == 640 else 640
        elif key == ord("t"):
            src.set_threaded(not src.threaded)
    mode = "threaded" if src.threaded else "sync"
    src.release()
    cv2.destroyAllWindows()
    print(f"{n} frames, last config: {variant} {size} {mode}")


if __name__ == "__main__":
    main()
