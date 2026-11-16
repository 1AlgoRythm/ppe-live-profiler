"""Charts for the README, drawn only from results/*.json.  usage: .venv/bin/python scripts/make_charts.py"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from chart_style import setup, bars, save, legend_top, SERIES, SURFACE, INK2

ROOT = Path(__file__).resolve().parents[1]
R = {f.stem: json.load(open(f)) for f in (ROOT / "results").glob("*_*_*.json")}
(ROOT / "docs").mkdir(exist_ok=True)

fig, ax = setup(1, 2, w=11, h=4.2)
for a, mode in zip(ax, ["sync", "threaded"]):
    vals = [[R[f"{v}_{s}_{mode}"]["fps"] for s in (320, 640)] for v in ("fp32", "int8")]
    bars(a, ["320 px input", "640 px input"], ["fp32", "int8"], vals, "{:.1f}")
    a.set_title(f"{mode} capture", fontsize=10, color=INK2, loc="left"); a.set_ylabel("FPS (median of 5 runs)")
legend_top(fig, ["fp32", "int8"])
save(fig, ROOT / "docs/fps.png", "End-to-end FPS, full PPE cascade on CPU",
     "Source: results/<variant>_<size>_<capture>.json. Machine was shared with other jobs, so absolute FPS is noisy; min-max per run is in the table.")

STAGES = ["capture", "preprocess", "inference", "postprocess", "render"]
cfgs = [f"{v}_{s}_{m}" for v in ("fp32", "int8") for s in (320, 640) for m in ("sync", "threaded")]
fig, ax = setup(w=10, h=4.6)
ax.grid(axis="x", color="#e4e3df"); ax.grid(axis="y", visible=False)
for i, c in enumerate(cfgs):
    left = 0
    for si, st in enumerate(STAGES):
        w = R[c]["stages"][st]["share_of_frame"] * 100
        ax.barh(i, w, left=left, color=SERIES[si], edgecolor=SURFACE, linewidth=2, height=0.6, label=st if i == 0 else None)
        if st == "inference": ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        left += w
ax.set_yticks(range(len(cfgs))); ax.set_yticklabels([c.replace("_", " ") for c in cfgs]); ax.invert_yaxis()
ax.set_xlabel("% of mean frame time"); ax.set_xlim(0, 100)
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
save(fig, ROOT / "docs/stage_share.png", "Where each frame's time goes",
     "Source: stages.<stage>.share_of_frame in results/*.json. Exact percentages for every stage are in the table above.")
print("wrote docs/fps.png, docs/stage_share.png")
