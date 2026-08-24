"""旋轉切片的 GeoJSON 轉換回歸測試。

背景：部分 HD 切片是「直式裁切（檔名 _WxH）、旋轉 90° 後儲存」，
且不同批次的旋轉方向可能不同（順時針 cw / 逆時針 ccw，例如 110-2 全為 ccw）。
推論座標基於儲存方向，轉 GeoJSON 前必須依方向反旋轉回原始方向再套 crop 偏移。
偵測規則：檔名 WxH 與 JSON 記錄的實際影像尺寸恰好對調；方向由 rotation map 提供。
"""
import numpy as np
import cv2
import pytest

from app_utils.tool.tfw2lonlat import (
    WorldFile,
    _detect_stored_rotation,
    _derotate_coords,
    convert_yolo_json_to_geojson,
)
from pyproj import Transformer


def _identity_args():
    wf = WorldFile(A=1.0, D=0.0, B=0.0, E=1.0, C=0.0, F=0.0)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
    return wf, transformer


def _payload(file_name, width, height, objects):
    return {
        "image": {"file_name": file_name, "width": width, "height": height},
        "model": {"name": "test"},
        "objects": objects,
    }


def _line_object():
    return {
        "class_id": 0, "class_name": "line", "confidence": 0.9,
        "bbox_xyxy": [10, 20, 30, 40],
        "polygons": [[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 20.0]]],
        "line_strings": [{"type": "LineString",
                          "coordinates": [[10.0, 20.0], [30.0, 40.0]]}],
    }


def test_detect_rotation():
    # 檔名直式 1080x1920、實際橫式 1920x1080 -> 旋轉切片，回傳檔名 (W, H)
    assert _detect_stored_rotation(
        {"file_name": "t_0001_100_200_1080x1920.png", "width": 1920, "height": 1080}
    ) == (1080.0, 1920.0)
    # 尺寸一致 -> 無旋轉
    assert _detect_stored_rotation(
        {"file_name": "t_0001_100_200_1920x1080.png", "width": 1920, "height": 1080}
    ) is None
    # 無 WxH token -> 無旋轉
    assert _detect_stored_rotation(
        {"file_name": "uploaded_image.png", "width": 1920, "height": 1080}
    ) is None


def test_derotate_formulas_match_cv2_rotation():
    """機器驗證：反旋轉公式必須與 cv2.rotate 的實際像素搬移互為反函數。"""
    w_p, h_p = 4, 6  # 直式原圖
    portrait = np.arange(w_p * h_p, dtype=np.int32).reshape(h_p, w_p)

    for direction, cv2_rot in (("cw", cv2.ROTATE_90_CLOCKWISE),
                               ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
        stored = cv2.rotate(portrait, cv2_rot)
        # 對儲存影像的每個像素中心 (x+0.5, y+0.5)，反旋轉後應落在原圖同值像素內
        for y_l in range(stored.shape[0]):
            for x_l in range(stored.shape[1]):
                val = stored[y_l, x_l]
                (x_p, y_p), = _derotate_coords([[x_l + 0.5, y_l + 0.5]], w_p, h_p, direction)
                assert portrait[int(y_p), int(x_p)] == val, (
                    f"{direction}: stored({x_l},{y_l}) -> portrait({x_p},{y_p}) 不符"
                )


def test_rotated_slice_cw():
    # cw 儲存：(x, y) -> (y, H - x)；(10,20) -> (20, 1910) -> +crop(100,200) = (120, 2110)
    payload = _payload("t_0001_100_200_1080x1920.png", 1920, 1080, [_line_object()])
    wf, transformer = _identity_args()
    geojson = convert_yolo_json_to_geojson(
        payload, wf, transformer, 100.0, 200.0,
        rotation_map={"t_0001_100_200_1080x1920": "cw"},
    )
    geom = geojson["features"][0]["geometry"]
    assert geom["coordinates"][0] == pytest.approx([120.0, 2110.0])
    assert geom["coordinates"][1] == pytest.approx([140.0, 2090.0])
    bbox = geojson["features"][0]["properties"]["bbox_pixel"]
    assert bbox == pytest.approx([20.0, 1890.0, 40.0, 1910.0])


def test_rotated_slice_ccw():
    # ccw 儲存：(x, y) -> (W - y, x)；(10,20) -> (1060, 10) -> +crop(100,200) = (1160, 210)
    payload = _payload("t_0001_100_200_1080x1920.png", 1920, 1080, [_line_object()])
    wf, transformer = _identity_args()
    geojson = convert_yolo_json_to_geojson(
        payload, wf, transformer, 100.0, 200.0,
        rotation_map={"t_0001_100_200_1080x1920": "ccw"},
    )
    geom = geojson["features"][0]["geometry"]
    assert geom["coordinates"][0] == pytest.approx([1160.0, 210.0])
    # (30,40) -> (1040, 30) -> +crop = (1140, 230)
    assert geom["coordinates"][1] == pytest.approx([1140.0, 230.0])
    bbox = geojson["features"][0]["properties"]["bbox_pixel"]
    # 角點 (1060,10),(1040,30) -> [1040,10,1060,30]
    assert bbox == pytest.approx([1040.0, 10.0, 1060.0, 30.0])


def test_rotated_slice_without_map_defaults_to_cw():
    payload = _payload("t_0001_100_200_1080x1920.png", 1920, 1080, [_line_object()])
    wf, transformer = _identity_args()
    geojson = convert_yolo_json_to_geojson(payload, wf, transformer, 100.0, 200.0)
    assert geojson["features"][0]["geometry"]["coordinates"][0] == pytest.approx([120.0, 2110.0])


def test_unrotated_slice_unchanged():
    payload = _payload("t_0001_100_200_1920x1080.png", 1920, 1080, [_line_object()])
    wf, transformer = _identity_args()
    geojson = convert_yolo_json_to_geojson(payload, wf, transformer, 100.0, 200.0)
    geom = geojson["features"][0]["geometry"]
    assert geom["coordinates"][0] == pytest.approx([110.0, 220.0])
    assert geojson["features"][0]["properties"]["bbox_pixel"] == [10, 20, 30, 40]
