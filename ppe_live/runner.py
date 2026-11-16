"""One loop iteration shared by the UI and the benchmark: capture -> cascade -> render."""
import statistics
import time
from collections import deque

from .pipeline import STAGES
from .render import draw_frame


class FrameLoop:
    def __init__(self, window=30):
        self.recent = deque(maxlen=window)    # last per-frame records, for the overlay
        self.ends = deque(maxlen=window + 1)  # frame end times, for the FPS readout

    def recent_median(self):
        return {s: statistics.median(r[s] for r in self.recent) for s in STAGES} if self.recent else {}

    def fps(self):
        return (len(self.ends) - 1) / (self.ends[-1] - self.ends[0]) if len(self.ends) > 1 else 0.0

    def step(self, source, cascade, label):
        """Process one frame. Returns (rendered frame, FrameResult, record) or None at end of video."""
        t0 = time.perf_counter()
        frame, capture_ms = source.read()                       # (H, W, 3) BGR
        if frame is None:
            return None
        res = cascade.run(frame)
        t1 = time.perf_counter()
        shown = draw_frame(frame, res, self.recent_median(), self.fps(), label)
        t2 = time.perf_counter()
        rec = {"capture": capture_ms, **res.ms, "render": (t2 - t1) * 1e3,
               "frame_ms": (t2 - t0) * 1e3,
               "infer_person": res.infer_person_ms, "infer_ppe": res.infer_ppe_ms,
               "n_persons": len(res.persons), "n_ppe": len(res.ppe)}
        self.recent.append(rec)
        self.ends.append(t2)
        return shown, res, rec
