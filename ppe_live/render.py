"""Draw detections, FPS and the per-stage latency bar onto a frame."""
import cv2

from .pipeline import PPE_NAMES, STAGES

GREEN, ORANGE = (0, 200, 0), (0, 140, 255)
STAGE_COLORS = {"capture": (200, 120, 40), "preprocess": (60, 180, 240), "inference": (60, 60, 230),
                "postprocess": (60, 200, 120), "render": (200, 200, 200)}


def draw_frame(frame, result, stage_ms, fps, label, bar_w=300):
    """Return a copy of frame with boxes, labels and the overlay panel.

    stage_ms: dict stage -> ms (typically a recent median) shown in the latency bar.
    """
    out = frame.copy()
    for x1, y1, x2, y2, s in result.persons:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
        _label(out, f"person {s:.2f}", int(x1), int(y1), GREEN)
    for x1, y1, x2, y2, s, k in result.ppe:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), ORANGE, 2)
        _label(out, f"{PPE_NAMES[k]} {s:.2f}", int(x1), int(y1), ORANGE)
    _panel(out, stage_ms, fps, label, bar_w)
    return out


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _panel(img, stage_ms, fps, label, bar_w):
    total = sum(stage_ms.get(s, 0.0) for s in STAGES) or 1.0
    h = 38 + 18 * len(STAGES) + 28
    roi = img[8:8 + h, 8:8 + bar_w + 20]
    roi[:] = (roi * 0.35).astype(roi.dtype)                 # darken the panel background
    cv2.putText(img, f"{fps:5.1f} FPS  {total:6.1f} ms", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, label, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    x = 16
    for s in STAGES:                                        # stacked bar, width ~ share of frame
        w = int(round(bar_w * stage_ms.get(s, 0.0) / total))
        cv2.rectangle(img, (x, 58), (x + w, 74), STAGE_COLORS[s], -1)
        x += w
    for i, s in enumerate(STAGES):
        y = 94 + 18 * i
        cv2.rectangle(img, (16, y - 10), (26, y), STAGE_COLORS[s], -1)
        cv2.putText(img, f"{s:<12}{stage_ms.get(s, 0.0):7.2f} ms", (32, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
