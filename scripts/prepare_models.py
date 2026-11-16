"""Copy the project-1 ONNX models into ./models and make 320x320 variants.

The project-1 models have a fixed 640x640 input. YOLOv8's ONNX export bakes the
anchor grid (2 x 8400 points) and the stride table into constants, so a 320 model
needs those rebuilt for 2100 points (40*40 + 20*20 + 10*10). Nothing else in the
graph depends on the input size (Reshapes use -1; Resize uses scale factors).

Self-check: the same builder is run for 640 and must reproduce the original
constants exactly, otherwise the script aborts.

The source folder is only ever READ.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper as nh

SRC_DEFAULT = Path("/Users/satyaags/Projects/ppe-quantization-lab/models")
STRIDES = (8, 16, 32)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anchors(size):
    """YOLOv8 anchor centres (1, 2, N) and per-anchor stride (1, N) for a square input."""
    pts, strd = [], []
    for s in STRIDES:
        n = size // s
        sx = np.arange(n, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(sx, sx)                       # (n, n) each, x runs fastest
        pts.append(np.stack([gx.ravel(), gy.ravel()], 0))  # (2, n*n)
        strd.append(np.full(n * n, s, dtype=np.float32))   # (n*n,)
    return np.concatenate(pts, 1)[None], np.concatenate(strd)[None]  # (1, 2, N), (1, N)


def patch(model, size):
    """Return the model with its anchor-dependent constants rebuilt for `size`."""
    pts, strd = anchors(size)
    n = pts.shape[2]
    new = {
        "/model.22/dfl/Constant_output_0": np.array([1, 4, 16, n], dtype=np.int64),
        "/model.22/dfl/Constant_1_output_0": np.array([1, 4, n], dtype=np.int64),
        "/model.22/Constant_12_output_0": pts,
        "/model.22/Constant_13_output_0": pts,
        "/model.22/Constant_15_output_0": strd,
    }
    for init in model.graph.initializer:
        if init.name in new:
            old = nh.to_array(init)
            val = new[init.name].astype(old.dtype)
            init.CopyFrom(nh.from_array(val, init.name))
    for t in model.graph.input:
        dims = t.type.tensor_type.shape.dim
        dims[2].dim_value, dims[3].dim_value = size, size
    for t in model.graph.output:
        t.type.tensor_type.shape.dim[2].dim_value = n
    del model.graph.value_info[:]
    return model


def self_check(model):
    """The builder at 640 must reproduce the original constants bit for bit."""
    pts, strd = anchors(640)
    got = {i.name: nh.to_array(i) for i in model.graph.initializer}
    assert np.array_equal(got["/model.22/Constant_12_output_0"], pts)
    assert np.array_equal(got["/model.22/Constant_13_output_0"], pts)
    assert np.array_equal(got["/model.22/Constant_15_output_0"], strd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--provenance", type=Path, default=Path("results/model_provenance.json"))
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    prov = {}
    for name in ("person", "ppe"):
        for variant in ("fp32", "int8"):
            src = a.src / f"{name}_{variant}.onnx"
            dst640 = a.out / f"{name}_{variant}_640.onnx"
            dst320 = a.out / f"{name}_{variant}_320.onnx"
            shutil.copyfile(src, dst640)
            self_check(onnx.load(dst640))
            onnx.save(patch(onnx.load(dst640), 320), dst320)
            sess = ort.InferenceSession(str(dst320), providers=["CPUExecutionProvider"])
            out = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, 320, 320), np.float32)})[0]
            prov[f"{name}_{variant}"] = {
                "source": str(src), "source_sha256": sha256(src),
                "copy_640": dst640.name, "sha256_640": sha256(dst640),
                "patched_320": dst320.name, "sha256_320": sha256(dst320),
                "output_shape_320": list(out.shape),
            }
            print(name, variant, "320 output", out.shape)
    a.provenance.write_text(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
