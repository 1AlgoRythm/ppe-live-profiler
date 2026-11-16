"""Build a headless test video from the 40 held-out person test images.

Each image is letterboxed (aspect kept, black bars) into a 1280x720 frame and held
for --hold frames. Output goes to data/test_video.mp4 (gitignored).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LAB = Path("/Users/satyaags/Projects/ppe-quantization-lab")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=LAB / "data/person/test/images")
    ap.add_argument("--split", type=Path, default=LAB / "data/split.json")
    ap.add_argument("--out", type=Path, default=Path("data/test_video.mp4"))
    ap.add_argument("--hold", type=int, default=5, help="frames per image")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()

    files = sorted(a.images.glob("*.jpg")) + sorted(a.images.glob("*.png"))
    expected = set(json.loads(a.split.read_text())["splits"]["test"]["files"])
    assert {f.stem for f in files} == expected, "image folder does not match split.json test files"

    a.out.parent.mkdir(exist_ok=True)
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.width, a.height))
    for f in files:
        img = cv2.imread(str(f))                            # (H, W, 3)
        s = min(a.width / img.shape[1], a.height / img.shape[0])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)))
        canvas = np.zeros((a.height, a.width, 3), np.uint8)
        y, x = (a.height - img.shape[0]) // 2, (a.width - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        for _ in range(a.hold):
            vw.write(canvas)
    vw.release()
    cap = cv2.VideoCapture(str(a.out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{len(files)} images -> {n} frames, {a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
