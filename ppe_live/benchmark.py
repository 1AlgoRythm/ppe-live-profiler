"""Headless benchmark: every (variant x size x threaded) config over the test video -> results/*.json."""
import argparse
import hashlib
import itertools
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

from .pipeline import STAGES, Cascade
from .runner import FrameLoop
from .sources import Source

EXTRA = ("infer_person", "infer_ppe", "frame_ms")


def p95(xs):
    return float(np.percentile(xs, 95))


def run_config(video, cascade, threaded, warmup):
    """Warm up on the first `warmup` frames, then time one pass over the whole video."""
    src = Source(video)
    loop = FrameLoop()
    for _ in range(warmup):
        loop.step(src, cascade, "warmup")
    src.release()

    src = Source(video)
    src.set_threaded(threaded)
    loop = FrameLoop()
    proc = psutil.Process()
    load_before = psutil.getloadavg()[0]                    # 1-min load average, whole machine
    proc.cpu_percent(None)                                  # start the CPU% interval
    t0 = time.perf_counter()
    recs = []
    while (out := loop.step(src, cascade, "")) is not None:
        recs.append(out[2])
    wall = time.perf_counter() - t0
    cpu = proc.cpu_percent(None)                            # % of ONE core, summed over threads
    thread_ms = list(src.thread_read_ms)
    src.release()
    return recs, wall, cpu, thread_ms, load_before


def summarize(runs):
    """runs: list of (recs, wall, cpu, thread_ms, load_before), one per repeat.

    Stage medians/p95 pool the frames of all repeats; FPS and CPU% are the median over repeats.
    """
    recs = [r for run in runs for r in run[0]]
    thread_ms = [x for run in runs for x in run[3]]
    mean_frame = statistics.fmean(r["frame_ms"] for r in recs)
    stages = {}
    for s in STAGES + EXTRA:
        xs = [r[s] for r in recs]
        stages[s] = {"median_ms": statistics.median(xs), "p95_ms": p95(xs), "mean_ms": statistics.fmean(xs),
                     "share_of_frame": statistics.fmean(xs) / mean_frame}
    per_run = [{"frames": len(r[0]), "wall_s": r[1], "fps": len(r[0]) / r[1], "cpu_percent": r[2],
                "load_avg_1min_before": r[4]} for r in runs]
    out = {"repeats": len(runs), "frames_pooled": len(recs),
           "fps": statistics.median(x["fps"] for x in per_run),
           "fps_min": min(x["fps"] for x in per_run), "fps_max": max(x["fps"] for x in per_run),
           "cpu_percent": statistics.median(x["cpu_percent"] for x in per_run),
           "runs": per_run, "stages": stages,
           "persons_per_run": sum(r["n_persons"] for r in runs[0][0]),
           "ppe_per_run": sum(r["n_ppe"] for r in runs[0][0])}
    if thread_ms:
        out["capture_thread_read_ms"] = {"median_ms": statistics.median(thread_ms), "p95_ms": p95(thread_ms)}
    return out


def meta(video):
    prov = Path("results/model_provenance.json")
    return {"date": datetime.now().astimezone().isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "onnxruntime": ort.__version__, "opencv": cv2.__version__,
            "numpy": np.__version__, "psutil": psutil.__version__, "machine": platform.platform(),
            "cpu_count": psutil.cpu_count(), "video": str(video),
            "video_sha256": hashlib.sha256(Path(video).read_bytes()).hexdigest(),
            "ort_threads": "default (all cores)", "providers": ["CPUExecutionProvider"],
            "model_provenance": json.loads(prov.read_text()) if prov.exists() else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="data/test_video.mp4")
    ap.add_argument("--out", default="results")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3, help="passes over the video per config (interleaved)")
    ap.add_argument("--variants", nargs="+", default=["fp32", "int8"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[320, 640])
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(exist_ok=True)
    info = meta(a.video)
    info["repeats"], info["warmup_frames"] = a.repeats, a.warmup
    configs = list(itertools.product(a.variants, a.sizes, (False, True)))
    cascades, runs, failed = {}, {c: [] for c in configs}, {}
    for rep in range(a.repeats):                            # interleave configs so drift hits all equally
        for cfg in configs:
            variant, size, threaded = cfg
            name = f"{variant}_{size}_{'threaded' if threaded else 'sync'}"
            if cfg in failed:
                continue
            try:
                if (variant, size) not in cascades:
                    cascades[(variant, size)] = Cascade(variant, size)
                run = run_config(a.video, cascades[(variant, size)], threaded, a.warmup)
                runs[cfg].append(run)
                print(f"rep {rep + 1} {name:22s} {len(run[0]) / run[1]:6.2f} FPS  load {run[4]:.1f}  "
                      f"cpu {run[2]:6.1f}%", flush=True)
            except Exception as e:                          # record, never estimate
                failed[cfg] = f"{type(e).__name__}: {e}"
                print(f"rep {rep + 1} {name:22s} FAILED {failed[cfg]}", flush=True)
    for cfg in configs:
        variant, size, threaded = cfg
        name = f"{variant}_{size}_{'threaded' if threaded else 'sync'}"
        doc = {"config": {"variant": variant, "size": size, "threaded": threaded}, "meta": info}
        if cfg in failed:
            doc["FAILED"] = failed[cfg]
        else:
            doc.update(summarize(runs[cfg]))
            doc["per_frame"] = [r for run in runs[cfg] for r in run[0]]
        (out_dir / f"{name}.json").write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
