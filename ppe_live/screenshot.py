"""Save one frame of the test video, rendered by the app's own draw code, to docs/screenshot.png.

Runs the real cascade for --warm frames first so the latency bar shows measured values,
then renders --frame (default: the frame with the most PPE detections among the first --scan).
"""
import argparse

import cv2

from .pipeline import Cascade
from .runner import FrameLoop
from .sources import Source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/test_video.mp4")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--frame", type=int, default=-1, help="frame index; -1 = most PPE detections")
    ap.add_argument("--out", default="docs/screenshot.png")
    a = ap.parse_args()

    cascade = Cascade(a.variant, a.size)
    src, loop = Source(a.video), FrameLoop()
    label = f"{a.variant} {a.size} sync  [v]ariant [s]ize [t]hread [q]uit"
    best, i = None, 0
    while (out := loop.step(src, cascade, label)) is not None:
        shown, res, _ = out
        if (a.frame == i) or (a.frame < 0 and (best is None or len(res.ppe) > best[0])):
            best = (len(res.ppe), shown.copy(), i)
        if a.frame == i:
            break
        i += 1
    n, img, idx = best
    cv2.imwrite(a.out, img)
    print(f"saved {a.out}: frame {idx}, {n} PPE boxes")


if __name__ == "__main__":
    main()
