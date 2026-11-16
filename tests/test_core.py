import numpy as np

from ppe_live.pipeline import decode, letterbox
from ppe_live.sources import Source


def test_letterbox_shape_and_scale():
    img = np.zeros((480, 640, 3), np.uint8)
    out, scale, px, py = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    assert scale == 0.5 and px == 0 and py == 40


def test_decode_roundtrip_and_nms():
    # one real box + a duplicate of it: NMS must keep one. Box centre (160,160), size 100, in 320 space.
    out = np.zeros((1, 5, 2), np.float32)
    out[0, :, 0] = [160, 160, 100, 100, 0.9]
    out[0, :, 1] = [162, 160, 100, 100, 0.8]
    boxes, scores, cls = decode(out, scale=0.5, pad_x=0, pad_y=40, orig_w=640, orig_h=480, conf=0.25, iou=0.45)
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], [220, 140, 420, 340], atol=1e-3)  # undo pad 40 and scale 0.5


def test_decode_empty():
    boxes, scores, cls = decode(np.zeros((1, 5, 10), np.float32), 1, 0, 0, 100, 100, 0.25, 0.45)
    assert boxes.shape == (0, 4)


def test_threaded_source_gives_every_frame(tmp_path=None):
    n_sync, n_thr = 0, 0
    s = Source("data/test_video.mp4")
    while s.read()[0] is not None:
        n_sync += 1
    s = Source("data/test_video.mp4")
    s.set_threaded(True)
    while s.read()[0] is not None:
        n_thr += 1
    s.release()
    assert n_sync == n_thr == 200
