"""Render the README results tables from results/*.json (markdown on stdout, or into README.md).

With --readme README.md, the text between <!-- results:start --> and <!-- results:end --> is replaced.
"""
import argparse
import json
from pathlib import Path

ORDER = [(v, s, t) for v in ("fp32", "int8") for s in (320, 640) for t in (False, True)]
STAGES = ("capture", "preprocess", "inference", "postprocess", "render")


def load(results_dir):
    docs = {}
    for f in sorted(Path(results_dir).glob("*_*_*.json")):
        d = json.loads(f.read_text())
        c = d["config"]
        docs[(c["variant"], c["size"], c["threaded"])] = (f, d)
    return docs


def tables(docs):
    out = ["Median / p95 ms per stage, summed over the person model and every PPE crop in a frame. "
           "FPS and CPU% are the median of the repeats; CPU% is percent of one core (100 = one core busy).", ""]
    head = "| variant | input | capture | " + " | ".join(STAGES[1:]) + " | FPS (min-max) | CPU% | source |"
    out += [head, "|" + "---|" * (head.count("|") - 1)]
    for cfg in ORDER:
        if cfg not in docs:
            continue
        f, d = docs[cfg]
        v, s, t = cfg
        label = f"| {v} | {s} {'threaded' if t else 'sync'} |"
        if "FAILED" in d:
            out.append(f"{label} FAILED: {d['FAILED']} |||||||`{f.as_posix()}` |")
            continue
        cells = [f"{d['stages'][k]['median_ms']:.2f} / {d['stages'][k]['p95_ms']:.2f}" for k in STAGES]
        out.append(f"{label} " + " | ".join(cells) +
                   f" | {d['fps']:.1f} ({d['fps_min']:.1f}-{d['fps_max']:.1f}) | {d['cpu_percent']:.0f} "
                   f"| `{f.as_posix()}` |")
    out += ["", "Share of mean frame time (stage mean / frame mean; capture thread time is not in the main loop):", ""]
    head2 = "| variant | input | " + " | ".join(STAGES) + " | of which person-infer | of which ppe-infer |"
    out += [head2, "|" + "---|" * (head2.count("|") - 1)]
    for cfg in ORDER:
        if cfg not in docs or "FAILED" in docs[cfg][1]:
            continue
        v, s, t = cfg
        st = docs[cfg][1]["stages"]
        sh = [f"{st[k]['share_of_frame'] * 100:.1f}%" for k in STAGES + ("infer_person", "infer_ppe")]
        out.append(f"| {v} | {s} {'threaded' if t else 'sync'} | " + " | ".join(sh) + " |")
    return "\n".join(out)


def headline(docs):
    ok = {c: d for c, (f, d) in docs.items() if "FAILED" not in d}
    lines = []
    best = max(ok, key=lambda c: ok[c]["fps"])
    lines.append(f"- Best config by median FPS: {best[0]} {best[1]} {'threaded' if best[2] else 'sync'} "
                 f"at {ok[best]['fps']:.1f} FPS.")
    for size in (320, 640):
        for t in (False, True):
            a, b = ok.get(("fp32", size, t)), ok.get(("int8", size, t))
            if a and b:
                lines.append(f"- INT8 vs FP32 end-to-end FPS, {size} {'threaded' if t else 'sync'}: "
                             f"{a['fps']:.1f} -> {b['fps']:.1f} ({(b['fps'] / a['fps'] - 1) * 100:+.0f}%); "
                             f"inference median {a['stages']['inference']['median_ms']:.1f} -> "
                             f"{b['stages']['inference']['median_ms']:.1f} ms.")
    for size in (320, 640):
        for v in ("fp32", "int8"):
            a, b = ok.get((v, size, False)), ok.get((v, size, True))
            if a and b:
                lines.append(f"- Threaded vs sync capture, {v} {size}: {a['fps']:.1f} -> {b['fps']:.1f} FPS "
                             f"({(b['fps'] / a['fps'] - 1) * 100:+.0f}%); capture stage median "
                             f"{a['stages']['capture']['median_ms']:.2f} -> {b['stages']['capture']['median_ms']:.2f} ms.")
    for c in ok:
        st = ok[c]["stages"]
        slow = max(STAGES, key=lambda k: st[k]["share_of_frame"])
        lines.append(f"- Slowest stage, {c[0]} {c[1]} {'threaded' if c[2] else 'sync'}: {slow} "
                     f"({st[slow]['share_of_frame'] * 100:.1f}% of mean frame time).")
    for v in ("fp32", "int8"):
        for size in (320, 640):
            d = ok.get((v, size, False))
            if d:
                lines.append(f"- Detections per 200-frame pass (conf 0.25), {v} {size}: "
                             f"{d['persons_per_run']} person, {d['ppe_per_run']} PPE.")
    loads = [r["load_avg_1min_before"] for d in ok.values() for r in d["runs"]]
    lines.append(f"- Machine load (1-min load average, {ok[best]['meta']['cpu_count']} cores) before each run: "
                 f"min {min(loads):.1f}, median {sorted(loads)[len(loads) // 2]:.1f}, max {max(loads):.1f}.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--readme")
    a = ap.parse_args()
    docs = load(a.results)
    text = tables(docs) + "\n\nHeadline numbers (computed from the same JSON):\n\n" + headline(docs)
    if not a.readme:
        print(text)
        return
    p = Path(a.readme)
    s = p.read_text()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    i, j = s.index(start) + len(start), s.index(end)
    p.write_text(s[:i] + "\n" + text + "\n" + s[j:])
    print(f"updated {a.readme}")


if __name__ == "__main__":
    main()
