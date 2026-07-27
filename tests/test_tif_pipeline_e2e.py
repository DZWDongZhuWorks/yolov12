"""run_tif_pipeline 的合成端到端測試（stub 模型，無 GPU / 無權重）。

流程：合成 4000x4000 大圖（白色矩形刻意橫跨 1600/overlap=320 的切片縫）
+ 手寫 TFW（EPSG:3826）→ 以 stub 模型（亮區偵測）跑 process_stem →
驗證 (a) 跨縫矩形在 GeoJSON 中恰好出現一次（縫合去重）、
(b) feature 座標與獨立 pyproj 換算一致、(c) 不留切片時磁碟只有 GeoJSON。
"""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from pyproj import Transformer

import run_tif_pipeline
from app_utils.tool.tfw2lonlat import WorldFile

# ---- 合成場景 ----
IMG_W, IMG_H = 4000, 4000
# TFW：像素 0.1m、左上角 (250000, 2650000)，TWD97
TFW_A, TFW_E = 0.1, -0.1
TFW_C, TFW_F = 250000.0, 2650000.0
# 矩形 (x1, y1, x2, y2)：
RECTS = [
    (1500, 700, 1750, 760),    # 橫跨 x=1600 的第一道縫（在 overlap 帶 1280~1600 內延伸出去）
    (400, 400, 700, 520),      # 完全在單一切片內
    (2500, 2450, 2620, 2700),  # 橫跨 y=2560 附近的縫
]


class _T:
    """最小 tensor 替身。"""

    def __init__(self, values):
        self._v = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._v


class _StubResult:
    def __init__(self, masks, cls, conf, shape):
        self.masks = SimpleNamespace(data=np.asarray(masks, dtype=np.float32)) if masks else None
        arr = SimpleNamespace(cls=_T(cls), conf=_T(conf))
        arr.__len__ = lambda self=arr: len(cls)
        self.boxes = _Boxes(cls, conf) if cls else None
        self.orig_shape = shape

    def cpu(self):
        return self


class _Boxes:
    def __init__(self, cls, conf):
        self.cls = _T(cls)
        self.conf = _T(conf)

    def __len__(self):
        return int(self.cls.numpy().shape[0])


class _StubModel:
    """以亮度閾值 + 連通元件充當「分割模型」：每個亮區是一個 class 0 instance。"""

    names = {0: "white_rect"}

    def predict(self, source, **kwargs):
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        binary = (gray > 200).astype(np.uint8)
        n, labels = cv2.connectedComponents(binary)
        masks, cls, conf = [], [], []
        for i in range(1, n):
            m = (labels == i).astype(np.float32)
            if m.sum() < 50:
                continue
            masks.append(m)
            cls.append(0)
            conf.append(0.9)
        return [_StubResult(masks, cls, conf, source.shape[:2])]


@pytest.fixture(scope="module")
def synthetic_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("tif_pipeline")
    tif_dir = root / "tif"
    out_dir = root / "out"
    tif_dir.mkdir()
    out_dir.mkdir()

    img = np.full((IMG_H, IMG_W, 3), 60, dtype=np.uint8)  # 深灰底（非空白）
    rng = np.random.default_rng(42)
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)  # 加雜訊避免 is_blank
    for x1, y1, x2, y2 in RECTS:
        img[y1:y2, x1:x2] = 255
    cv2.imwrite(str(tif_dir / "9999999.tif"), img)
    (tif_dir / "9999999.tfw").write_text(
        f"{TFW_A}\n0.0\n0.0\n{TFW_E}\n{TFW_C}\n{TFW_F}\n", encoding="utf-8"
    )
    return tif_dir, out_dir


def _run(tif_dir: Path, out_dir: Path):
    args = SimpleNamespace(
        tile_size=1600, overlap=320, imgsz=1600, conf=0.25, device="auto",
        src_epsg=3826, stitch_gap=1, keep_tiles=False, bigmap=False, bigmap_scale=0.5,
    )
    cfg = {
        "allowed_class_ids": None,
        "mask_steps_parsed": None,
        "polygon_steps_parsed": [],
    }
    transformer = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)
    run_tif_pipeline.process_stem(
        "9999999", tif_dir / "9999999.tif", tif_dir / "9999999.tfw",
        {"stub.pt": _StubModel()}, cfg, args, transformer, out_dir,
    )
    out_path = out_dir / "9999999__stub.geojson"
    assert out_path.exists()
    with out_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_end_to_end(synthetic_dirs):
    tif_dir, out_dir = synthetic_dirs
    geojson = _run(tif_dir, out_dir)
    features = geojson["features"]

    # (a) 每個矩形恰好一個 feature —— 跨縫者經縫合去重後不重複
    assert len(features) == len(RECTS)

    transformer = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)
    wf = WorldFile(TFW_A, 0.0, 0.0, TFW_E, TFW_C, TFW_F)

    # 依 bbox_pixel 面積由大到小配對 feature 與矩形
    def center(feat):
        bx = feat["properties"]["bbox_pixel"]
        return ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)

    matched = set()
    for x1, y1, x2, y2 in RECTS:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hits = [
            i for i, feat in enumerate(features)
            if abs(center(feat)[0] - cx) < 10 and abs(center(feat)[1] - cy) < 10
        ]
        assert len(hits) == 1, f"矩形 ({x1},{y1},{x2},{y2}) 應恰好對應一個 feature，got {len(hits)}"
        matched.add(hits[0])

        # (b) 座標驗證：feature 幾何的經緯度範圍應與獨立換算的矩形角點吻合
        feat = features[hits[0]]
        geom = feat["geometry"]
        assert geom["type"] == "Polygon"
        ring = np.asarray(geom["coordinates"][0])
        exp_lon1, exp_lat1 = transformer.transform(*wf.pixel_to_map(x1, y1))
        exp_lon2, exp_lat2 = transformer.transform(*wf.pixel_to_map(x2, y2))
        tol = abs(transformer.transform(*wf.pixel_to_map(0, 0))[0]
                  - transformer.transform(*wf.pixel_to_map(3, 0))[0])  # ~3px 容差
        assert abs(ring[:, 0].min() - min(exp_lon1, exp_lon2)) < tol
        assert abs(ring[:, 0].max() - max(exp_lon1, exp_lon2)) < tol
        assert abs(ring[:, 1].min() - min(exp_lat1, exp_lat2)) < tol
        assert abs(ring[:, 1].max() - max(exp_lat1, exp_lat2)) < tol

        # 跨縫矩形應標記多個來源切片
        if (x1, y1, x2, y2) == RECTS[0]:
            assert len(feat["properties"]["tile_seqs"]) >= 2

    assert matched == set(range(len(features)))

    # (c) 不留切片：輸出資料夾內不應有 tiles/
    assert not (out_dir / "tiles").exists()
