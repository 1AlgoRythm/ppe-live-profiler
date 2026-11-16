"""Frame sources: plain (blocking) capture and an optional background capture thread."""
import queue
import threading
import time
from collections import deque

import cv2


class Source:
    """cv2.VideoCapture wrapper. read() returns (frame, ms spent waiting for it).

    set_threaded(True) starts a reader thread. For a webcam it keeps only the newest
    frame (old ones are dropped, so latency stays low). For a video file it keeps a
    2-frame blocking queue, so no frame is skipped and benchmark runs are comparable.
    """

    def __init__(self, source, loop=False):
        self.is_file = not str(source).isdigit()
        self.cap = cv2.VideoCapture(str(source) if self.is_file else int(source))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open source {source!r}")
        self.loop = loop and self.is_file
        self.thread = None
        self.q = None
        self.pending = deque()          # frames the thread had read when it was stopped
        self.thread_read_ms = []        # per-frame time the thread spent in cap.read()
        self._stop = threading.Event()

    @property
    def threaded(self):
        return self.thread is not None

    def _read_one(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def _worker(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            frame = self._read_one()
            self.thread_read_ms.append((time.perf_counter() - t0) * 1e3)
            if not self.is_file and frame is not None:      # camera: keep only the newest
                while self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        break
            while not self._stop.is_set():
                try:
                    self.q.put(frame, timeout=0.05)
                    break
                except queue.Full:
                    continue
            if frame is None:
                return

    def set_threaded(self, on):
        if on and not self.threaded:
            self._stop.clear()
            self.q = queue.Queue(maxsize=2)
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
        elif not on and self.threaded:
            self._stop.set()
            self.thread.join()
            while not self.q.empty():
                item = self.q.get_nowait()
                if item is not None:
                    self.pending.append(item)
            self.thread, self.q = None, None

    def read(self):
        t0 = time.perf_counter()
        if self.pending:
            frame = self.pending.popleft()
        elif self.threaded:
            frame = self.q.get()
        else:
            frame = self._read_one()
        return frame, (time.perf_counter() - t0) * 1e3

    def release(self):
        self.set_threaded(False)
        self.cap.release()
