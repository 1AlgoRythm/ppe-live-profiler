"""Detector stages (preprocess -> inference -> postprocess) and the two-model cascade.

Every stage is timed with time.perf_counter and returned in milliseconds.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parent.parent / "models"
PERSON_NAMES = ["person"]
# Class order from project 1's data/ppe.yaml (7 classes).
PPE_NAMES = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit"]
STAGES = ("capture", "preprocess", "inference", "postprocess", "render")


def letterbox(img, size):
    """Resize keeping aspect ratio, pad to size x size with grey 114.

    Returns the padded image and (scale, pad_x, pad_y) to undo it.
    """
    h, w = img.shape[:2]                                    # (H, W, 3) BGR
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = cv2.copyMakeBorder(resized, pad_y, size - nh - pad_y, pad_x, size - nw - pad_x,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, scale, pad_x, pad_y                         # (size, size, 3)


def decode(output, scale, pad_x, pad_y, orig_w, orig_h, conf, iou):
    """YOLOv8 head output -> (boxes (N, 4) xyxy in the original image, scores (N,), classes (N,)).

    output: (1, 4 + nc, A) with A = 8400 at 640 or 2100 at 320. Rows 0..3 are
    cx, cy, w, h in letterboxed pixels; the other nc rows are class scores.
    """
    pred = output[0]                                        # (4 + nc, A)
    scores_all = pred[4:]                                   # (nc, A)
    best = scores_all.max(axis=0)                           # (A,)
    keep = np.flatnonzero(best >= conf)
    if keep.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    sel = pred[:, keep]                                     # (4 + nc, K)
    cls = sel[4:].argmax(axis=0)                            # (K,)
    score = sel[4:].max(axis=0)                             # (K,)
    cx, cy, w, h = sel[0], sel[1], sel[2], sel[3]           # (K,) each
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    bw, bh = w / scale, h / scale
    xywh = np.stack([x1, y1, bw, bh], 1)                    # (K, 4) x, y, w, h
    idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), score.tolist(), cls.tolist(), conf, iou)
    idx = np.array(idx, dtype=np.int64).reshape(-1)
    xywh, score, cls = xywh[idx], score[idx], cls[idx]
    boxes = np.stack([np.clip(xywh[:, 0], 0, orig_w), np.clip(xywh[:, 1], 0, orig_h),
                      np.clip(xywh[:, 0] + xywh[:, 2], 0, orig_w),
                      np.clip(xywh[:, 1] + xywh[:, 3], 0, orig_h)], 1)  # (N, 4) xyxy
    return boxes.astype(np.float32), score.astype(np.float32), cls.astype(np.int64)


class Detector:
    """One YOLOv8 ONNX model at one input size, on the ONNX Runtime CPU provider."""

    def __init__(self, name, variant, size, conf=0.25, iou=0.45, models_dir=MODELS):
        path = Path(models_dir) / f"{name}_{variant}_{size}.onnx"
        opts = ort.SessionOptions()                         # ORT defaults: all cores
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.size, self.conf, self.iou = size, conf, iou

    def __call__(self, img):
        """Returns (boxes, scores, classes, {'preprocess', 'inference', 'postprocess'} in ms)."""
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        lb, scale, px, py = letterbox(img, self.size)       # (S, S, 3) BGR
        x = cv2.dnn.blobFromImage(lb, 1 / 255.0, swapRB=True)  # (1, 3, S, S) RGB float32
        t1 = time.perf_counter()
        out = self.sess.run(None, {self.input_name: x})[0]  # (1, 4 + nc, A)
        t2 = time.perf_counter()
        boxes, scores, cls = decode(out, scale, px, py, w, h, self.conf, self.iou)
        t3 = time.perf_counter()
        ms = {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3,
              "postprocess": (t3 - t2) * 1e3}
        return boxes, scores, cls, ms


@dataclass
class FrameResult:
    persons: list = field(default_factory=list)   # [(x1, y1, x2, y2, score)]
    ppe: list = field(default_factory=list)       # [(x1, y1, x2, y2, score, class_id)] full-frame coords
    ms: dict = field(default_factory=dict)        # stage -> ms (summed over both models)
    infer_person_ms: float = 0.0
    infer_ppe_ms: float = 0.0


class Cascade:
    """Person model on the frame, then the PPE model on each person crop."""

    MIN_CROP = 16  # skip person boxes smaller than this (px) in either direction

    def __init__(self, variant, size):
        self.variant, self.size = variant, size
        self.person = Detector("person", variant, size)
        self.ppe = Detector("ppe", variant, size)

    def run(self, frame):
        res = FrameResult()
        ms = dict.fromkeys(("preprocess", "inference", "postprocess"), 0.0)
        boxes, scores, _, m = self.person(frame)
        for k in ms:
            ms[k] += m[k]
        res.infer_person_ms = m["inference"]
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            res.persons.append((float(x1), float(y1), float(x2), float(y2), float(s)))
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            if ix2 - ix1 < self.MIN_CROP or iy2 - iy1 < self.MIN_CROP:
                continue
            crop = frame[iy1:iy2, ix1:ix2]                  # (ch, cw, 3) view, no copy
            pb, ps, pc, m = self.ppe(crop)
            for k in ms:
                ms[k] += m[k]
            res.infer_ppe_ms += m["inference"]
            for (a, b, c, d), s2, k in zip(pb, ps, pc):     # crop coords -> frame coords
                res.ppe.append((float(a) + ix1, float(b) + iy1, float(c) + ix1, float(d) + iy1,
                                float(s2), int(k)))
        res.ms = ms
        return res
