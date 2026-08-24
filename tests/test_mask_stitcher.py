"""mask_stitcher 的單元測試。

涵蓋：
1. extract_instances_from_result：稀疏全域 instance 抽取（含類別篩選）
2. merge_instances：跨切片斷裂物件癒合、重疊重複去重、不同類別/不相交不合併
3. instances_to_objects：全域座標輪廓、洞保留、schema 與 build_objects_from_result 一致
4. apply_mask_steps_to_instances：抽取後的 mask 優化仍可用（重構回歸）
"""
import numpy as np

from app_utils.tool import mask_stitcher
from app_utils.inference_optimizations import (
    apply_mask_steps_to_instances,
    parse_mask_steps,
    resolve_mask_step_filters,
)


class _ArrayWrapper:
    def __init__(self, values, dtype=np.float32):
        self._values = np.asarray(values, dtype=dtype)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, cls, conf):
        self.cls = _ArrayWrapper(cls)
        self.conf = _ArrayWrapper(conf)

    def __len__(self):
        return int(self.cls.numpy().shape[0])


class _Masks:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)


class _Result:
    def __init__(self, masks, cls, conf, orig_shape):
        self.masks = _Masks(masks)
        self.boxes = _Boxes(cls, conf)
        self.orig_shape = orig_shape


def _rect_mask(h, w, y1, y2, x1, x2):
    m = np.zeros((h, w), dtype=np.float32)
    m[y1:y2, x1:x2] = 1.0
    return m


# ---------- extract_instances_from_result ----------

class TestExtractInstances:
    def test_global_offsets_and_crop(self):
        mask = _rect_mask(100, 100, 10, 30, 20, 60)
        result = _Result([mask], cls=[2], conf=[0.9], orig_shape=(100, 100))
        instances = mask_stitcher.extract_instances_from_result(result, tile_x=1000, tile_y=2000, tile_seq=7)
        assert len(instances) == 1
        inst = instances[0]
        assert inst["cls_id"] == 2
        assert inst["conf"] == np.float32(0.9)
        assert (inst["x0"], inst["y0"]) == (1020, 2010)
        assert inst["mask"].shape == (20, 40)
        assert inst["mask"].all()
        assert inst["tile_seqs"] == [7]

    def test_class_filter(self):
        masks = [_rect_mask(50, 50, 0, 10, 0, 10), _rect_mask(50, 50, 20, 30, 20, 30)]
        result = _Result(masks, cls=[1, 3], conf=[0.5, 0.6], orig_shape=(50, 50))
        instances = mask_stitcher.extract_instances_from_result(result, 0, 0, allowed_class_ids=[3])
        assert len(instances) == 1
        assert instances[0]["cls_id"] == 3

    def test_empty_mask_skipped(self):
        result = _Result([np.zeros((50, 50), dtype=np.float32)], cls=[0], conf=[0.5], orig_shape=(50, 50))
        assert mask_stitcher.extract_instances_from_result(result, 0, 0) == []


# ---------- merge_instances ----------

def _inst(cls_id, x0, y0, mask, conf=0.5, seqs=None):
    return {
        "cls_id": cls_id, "conf": conf, "x0": x0, "y0": y0,
        "mask": np.asarray(mask, dtype=np.uint8), "tile_seqs": seqs or [0],
    }


class TestMergeInstances:
    def test_overlapping_duplicates_merge_to_one(self):
        # 同一物件被兩個重疊切片各偵測一次（mask 大部分重疊）
        a = _inst(1, 100, 100, np.ones((20, 30)), conf=0.7, seqs=[1])
        b = _inst(1, 110, 100, np.ones((20, 30)), conf=0.9, seqs=[2])
        merged = mask_stitcher.merge_instances([a, b])
        assert len(merged) == 1
        m = merged[0]
        assert m["conf"] == 0.9  # 取最大
        assert m["tile_seqs"] == [1, 2]
        assert (m["x0"], m["y0"]) == (100, 100)
        assert m["mask"].shape == (20, 40)
        assert m["mask"].all()

    def test_cut_object_heals(self):
        # 長物件橫跨切片縫：兩半在重疊帶各有一小段共同像素
        left = _inst(0, 0, 50, np.ones((10, 60)))
        right = _inst(0, 55, 50, np.ones((10, 60)))
        merged = mask_stitcher.merge_instances([left, right])
        assert len(merged) == 1
        assert merged[0]["mask"].shape == (10, 115)
        assert merged[0]["mask"].all()

    def test_adjacent_within_gap_tolerance(self):
        # 恰好貼邊不重疊（overlap=0 的切法），gap=1 應合併
        a = _inst(0, 0, 0, np.ones((10, 50)))
        b = _inst(0, 50, 0, np.ones((10, 50)))
        assert len(mask_stitcher.merge_instances([a, b], gap=1)) == 1
        assert len(mask_stitcher.merge_instances([a, b], gap=0)) == 2

    def test_different_class_not_merged(self):
        a = _inst(0, 100, 100, np.ones((20, 20)))
        b = _inst(1, 105, 105, np.ones((20, 20)))
        assert len(mask_stitcher.merge_instances([a, b])) == 2

    def test_disjoint_not_merged(self):
        a = _inst(0, 0, 0, np.ones((10, 10)))
        b = _inst(0, 500, 500, np.ones((10, 10)))
        assert len(mask_stitcher.merge_instances([a, b])) == 2

    def test_bbox_overlap_but_pixels_disjoint(self):
        # 兩個 L 形 bbox 相交但像素不相交 → 不可合併
        m1 = np.zeros((30, 30), dtype=np.uint8); m1[:5, :] = 1
        m2 = np.zeros((30, 30), dtype=np.uint8); m2[-5:, :] = 1
        a = _inst(0, 0, 0, m1)
        b = _inst(0, 0, 10, m2)  # bbox 相交，但像素相距 >1px
        assert len(mask_stitcher.merge_instances([a, b], gap=1)) == 2

    def test_transitive_chain(self):
        # A-B 相交、B-C 相交、A-C 不相交 → 三者合一（union-find 遞移）
        a = _inst(0, 0, 0, np.ones((10, 40)))
        b = _inst(0, 35, 0, np.ones((10, 40)))
        c = _inst(0, 70, 0, np.ones((10, 40)))
        merged = mask_stitcher.merge_instances([a, b, c])
        assert len(merged) == 1
        assert merged[0]["mask"].shape == (10, 110)


# ---------- instances_to_objects ----------

class TestInstancesToObjects:
    def test_rectangle_global_coords(self):
        inst = _inst(2, 5000, 6000, np.ones((40, 80)), conf=0.8, seqs=[3, 4])
        objects = mask_stitcher.instances_to_objects([inst], names={2: "zebra"})
        assert len(objects) == 1
        obj = objects[0]
        assert obj["class_id"] == 2
        assert obj["class_name"] == "zebra"
        assert obj["confidence"] == 0.8
        assert obj["tile_seqs"] == [3, 4]
        assert obj["bbox_xyxy"] == [5000.0, 6000.0, 5079.0, 6039.0]
        assert len(obj["polygons"]) == 1
        ring = np.asarray(obj["polygons"][0])
        # 閉合、且座標落在全域 bbox 附近
        assert (ring[0] == ring[-1]).all()
        assert ring[:, 0].min() >= 4999 and ring[:, 0].max() <= 5080
        assert ring[:, 1].min() >= 5999 and ring[:, 1].max() <= 6040

    def test_donut_preserves_hole(self):
        m = np.ones((60, 60), dtype=np.uint8)
        m[20:40, 20:40] = 0
        inst = _inst(0, 100, 200, m)
        objects = mask_stitcher.instances_to_objects([inst], names={0: "ring"})
        assert len(objects) == 1
        holes = objects[0]["holes"]
        assert len(holes) == 1 and len(holes[0]) == 1
        hole = np.asarray(holes[0][0])
        assert 115 <= hole[:, 0].min() <= 125  # 洞座標也在全域座標系

    def test_mask_step_split_yields_two_objects(self):
        # 一個 instance 內兩個相距很遠的元件，套 split 步驟後應成為兩個物件
        m = np.zeros((20, 100), dtype=np.uint8)
        m[:, :20] = 1
        m[:, 80:] = 1
        inst = _inst(0, 0, 0, m)
        steps = parse_mask_steps([["split", 1]])
        objects = mask_stitcher.instances_to_objects([inst], names={0: "x"}, mask_steps=steps)
        assert len(objects) == 2

    def test_polygon_steps_applied(self):
        inst = _inst(0, 0, 0, np.ones((20, 200)))
        steps = [{"name": "export_line", "count": 1, "min_aspect": 0.0}]
        objects = mask_stitcher.instances_to_objects([inst], names={0: "lane"}, polygon_steps=steps)
        assert len(objects) == 1
        # export_line 轉為開放線 → _sync_line_string_payloads 應產生 line_strings
        assert objects[0].get("line_strings")


# ---------- apply_mask_steps_to_instances（重構回歸） ----------

class TestApplyMaskStepsToInstances:
    def test_dilate_grows_and_preserves_extra_keys(self):
        binary = np.zeros((30, 30), dtype=np.uint8)
        binary[10:20, 10:20] = 1
        item = {"binary": binary, "cls_id": 0, "conf": 0.5, "origin": (7, 8)}
        steps = resolve_mask_step_filters(parse_mask_steps([["dilate", 1]]), {0: "x"})
        out, timings = apply_mask_steps_to_instances([item], steps)
        assert len(out) == 1
        assert out[0]["binary"].sum() > binary.sum()
        assert out[0]["origin"] == (7, 8)  # 額外欄位保留
        assert timings and timings[0][0].startswith("dilate")

    def test_class_filter_skips(self):
        binary = np.zeros((30, 30), dtype=np.uint8)
        binary[10:20, 10:20] = 1
        item = {"binary": binary.copy(), "cls_id": 5, "conf": 0.5}
        # row 格式：[name, count, morph_kernel, blur_kernel, blur_threshold,
        #           min_component_area, max_hole_area, merge_iou_threshold, classes]
        steps = resolve_mask_step_filters(
            parse_mask_steps([["dilate", 1, 3, 3, 0.5, 0, 0, 0.1, "0"]]), {0: "x", 5: "y"}
        )
        out, _ = apply_mask_steps_to_instances([item], steps)
        assert (out[0]["binary"] == binary).all()
