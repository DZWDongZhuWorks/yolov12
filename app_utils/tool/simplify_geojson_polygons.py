#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simplify_geojson_polygons_to_rect_or_line.py

Read GeoJSON FeatureCollection, simplify Polygon/MultiPolygon features with
minAreaRect (rectangle polygon) or export as LineString (longest edge).

Usage:
  python simplify_geojson_polygons_to_rect_or_line.py \
    -j input.geojson -o output.geojson -c white_line --export-line

Notes:
- Filtering uses feature.properties.class_name
- Geometry calculation is done in dst CRS (planar), then transformed back.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from pyproj import Transformer


def _ensure_featurecollection(gj: Dict[str, Any]) -> None:
    if not isinstance(gj, dict) or gj.get("type") != "FeatureCollection":
        raise ValueError("輸入必須是 GeoJSON FeatureCollection（type=FeatureCollection）")
    if "features" not in gj or not isinstance(gj["features"], list):
        raise ValueError("FeatureCollection 必須包含 features: []")


def _get_class_name(feature: Dict[str, Any]) -> Optional[str]:
    props = feature.get("properties")
    if isinstance(props, dict):
        v = props.get("class_name")
        if isinstance(v, str):
            return v
    return None


def _transform_points(
    xy: np.ndarray,
    transformer: Transformer,
) -> np.ndarray:
    """
    xy: Nx2
    return: Nx2
    """
    xs = xy[:, 0].astype(float)
    ys = xy[:, 1].astype(float)
    tx, ty = transformer.transform(xs, ys)
    return np.stack([tx, ty], axis=1)


def _ring_to_xy(ring: List[List[float]]) -> np.ndarray:
    """
    ring: [[x,y], [x,y], ...] possibly closed (last equals first)
    returns Nx2 float array (not necessarily closed)
    """
    if not ring or not isinstance(ring, list):
        raise ValueError("Polygon ring 為空或非 list")
    arr = np.asarray(ring, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Polygon ring 座標格式錯誤，需為 [[x,y], ...]")
    arr = arr[:, :2]
    # 去掉結尾重複首點的閉合點（避免退化）
    if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    if len(arr) < 3:
        raise ValueError("Polygon 外環點數不足（<3）")
    return arr


def _min_area_rect_from_ring_xy(xy_planar: np.ndarray) -> np.ndarray:
    """
    xy_planar: Nx2 in planar CRS
    returns: 4x2 rectangle points (float) in planar CRS, ordered by cv2.boxPoints
    """
    pts = xy_planar.astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)    # 4x2
    return box.astype(float)


def _rect_to_polygon_coords(rect_xy: np.ndarray) -> List[List[float]]:
    """
    rect_xy: 4x2
    return GeoJSON polygon ring coordinates (closed)
    """
    ring = rect_xy.tolist()
    ring.append(ring[0])
    return ring


def _rect_to_longest_edge_line(rect_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    rect_xy: 4x2
    choose the longest edge among 4 edges, return its endpoints (2 points).
    """
    # edges: (i -> i+1)
    best_len = -1.0
    best = (rect_xy[0], rect_xy[1])
    for i in range(4):
        p = rect_xy[i]
        q = rect_xy[(i + 1) % 4]
        d = float(np.hypot(*(q - p)))
        if d > best_len:
            best_len = d
            best = (p, q)
    return best[0], best[1]


def _chain_between_points(ring_xy: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """Get ring chain from start_idx to end_idx (inclusive) following ring order."""
    if start_idx <= end_idx:
        return ring_xy[start_idx : end_idx + 1]
    return np.concatenate([ring_xy[start_idx:], ring_xy[: end_idx + 1]], axis=0)


def _resample_polyline_by_t(polyline: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    """Resample polyline using normalized arc-length t in [0,1]."""
    if len(polyline) < 2:
        raise ValueError("polyline 點數不足")
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(polyline[:1], len(t_values), axis=0)

    targets = np.clip(t_values, 0.0, 1.0) * total
    out = np.empty((len(targets), 2), dtype=float)
    j = 0
    for i, tar in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] < tar:
            j += 1
        if j + 1 >= len(cum):
            out[i] = polyline[-1]
            continue
        denom = cum[j + 1] - cum[j]
        if denom <= 1e-12:
            out[i] = polyline[j]
        else:
            alpha = (tar - cum[j]) / denom
            out[i] = polyline[j] * (1.0 - alpha) + polyline[j + 1] * alpha
    return out


def _extract_lane_centerline_from_polygon(xy_planar: np.ndarray, samples: int = 32) -> np.ndarray:
    """
    Convert elongated polygon contour to center line by pairing two boundary chains.
    Endpoints are estimated as the farthest vertex pair.
    """
    n = len(xy_planar)
    if n < 4:
        raise ValueError("Polygon 點數不足，無法抽取中心線")

    d2 = np.sum((xy_planar[:, None, :] - xy_planar[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    if i == j:
        raise ValueError("無法找到有效端點")

    chain_a = _chain_between_points(xy_planar, int(i), int(j))
    chain_b = _chain_between_points(xy_planar, int(j), int(i))

    # reverse chain_b so both chains have same endpoint order
    chain_b = chain_b[::-1]

    m = max(8, int(samples))
    t_values = np.linspace(0.0, 1.0, m)
    a_rs = _resample_polyline_by_t(chain_a, t_values)
    b_rs = _resample_polyline_by_t(chain_b, t_values)
    center = (a_rs + b_rs) / 2.0
    return center


def _simplify_centerline_points(
    line_xy: np.ndarray,
    merge_distance: float,
    turn_threshold_deg: float,
) -> np.ndarray:
    """
    Merge spatially close points while preserving endpoints and turn points.
    Turn points are decided by angle change from adjacent vectors.
    """
    n = len(line_xy)
    if n <= 2:
        return line_xy

    turn_threshold_rad = np.deg2rad(max(0.0, float(turn_threshold_deg)))
    critical = np.zeros(n, dtype=bool)
    critical[0] = True
    critical[-1] = True

    for idx in range(1, n - 1):
        v1 = line_xy[idx] - line_xy[idx - 1]
        v2 = line_xy[idx + 1] - line_xy[idx]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-12 or n2 <= 1e-12:
            critical[idx] = True
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        ang = float(np.arccos(cosang))
        if ang >= turn_threshold_rad:
            critical[idx] = True

    out: List[np.ndarray] = [line_xy[0]]
    last = line_xy[0]
    merge_d = max(0.0, float(merge_distance))
    for idx in range(1, n - 1):
        p = line_xy[idx]
        if critical[idx]:
            out.append(p)
            last = p
            continue
        if float(np.linalg.norm(p - last)) >= merge_d:
            out.append(p)
            last = p
    out.append(line_xy[-1])

    # remove accidental consecutive duplicates
    dedup = [out[0]]
    for p in out[1:]:
        if float(np.linalg.norm(p - dedup[-1])) > 1e-12:
            dedup.append(p)
    return np.asarray(dedup, dtype=float)


def _simplify_polygon_geometry(
    geom: Dict[str, Any],
    fwd: Transformer,
    inv: Transformer,
    export_line: bool,
    lane_line_mode: bool,
    line_merge_distance: float,
    line_turn_threshold_deg: float,
    line_samples: int,
) -> Dict[str, Any]:
    """
    Simplify a Polygon geometry to rectangle polygon or LineString.
    Only uses exterior ring (coordinates[0]).
    """
    coords = geom.get("coordinates")
    if not isinstance(coords, list) or len(coords) == 0:
        return geom

    exterior = coords[0]
    xy_src = _ring_to_xy(exterior)          # Nx2 in src CRS
    xy_planar = _transform_points(xy_src, fwd)

    if export_line:
        if lane_line_mode:
            line_planar = _extract_lane_centerline_from_polygon(xy_planar, samples=line_samples)
            line_planar = _simplify_centerline_points(
                line_planar,
                merge_distance=line_merge_distance,
                turn_threshold_deg=line_turn_threshold_deg,
            )
        else:
            rect_planar = _min_area_rect_from_ring_xy(xy_planar)  # 4x2 planar
            p1, p2 = _rect_to_longest_edge_line(rect_planar)
            line_planar = np.stack([p1, p2], axis=0)
        line_src = _transform_points(line_planar, inv)
        return {
            "type": "LineString",
            "coordinates": line_src.tolist(),
        }
    else:
        rect_planar = _min_area_rect_from_ring_xy(xy_planar)  # 4x2 planar
        rect_src = _transform_points(rect_planar, inv)
        ring = _rect_to_polygon_coords(rect_src)
        return {
            "type": "Polygon",
            "coordinates": [ring],
        }


def _simplify_multipolygon_geometry(
    geom: Dict[str, Any],
    fwd: Transformer,
    inv: Transformer,
    export_line: bool,
    lane_line_mode: bool,
    line_merge_distance: float,
    line_turn_threshold_deg: float,
    line_samples: int,
) -> Dict[str, Any]:
    """
    Simplify a MultiPolygon.
    - If export_line: output MultiLineString (one line per polygon)
    - Else: output MultiPolygon where each polygon becomes a rectangle polygon
    """
    coords = geom.get("coordinates")
    if not isinstance(coords, list) or len(coords) == 0:
        return geom

    if export_line:
        lines: List[List[List[float]]] = []
        for poly in coords:
            # poly: [exterior, hole1, ...]
            if not isinstance(poly, list) or len(poly) == 0:
                continue
            exterior = poly[0]
            try:
                xy_src = _ring_to_xy(exterior)
                xy_planar = _transform_points(xy_src, fwd)
                if lane_line_mode:
                    line_planar = _extract_lane_centerline_from_polygon(xy_planar, samples=line_samples)
                    line_planar = _simplify_centerline_points(
                        line_planar,
                        merge_distance=line_merge_distance,
                        turn_threshold_deg=line_turn_threshold_deg,
                    )
                else:
                    rect_planar = _min_area_rect_from_ring_xy(xy_planar)
                    p1, p2 = _rect_to_longest_edge_line(rect_planar)
                    line_planar = np.stack([p1, p2], axis=0)
                line_src = _transform_points(line_planar, inv)
                lines.append(line_src.tolist())
            except Exception:
                # 遇到壞幾何就保守跳過
                continue

        if not lines:
            return geom

        return {"type": "MultiLineString", "coordinates": lines}

    else:
        rects: List[List[List[List[float]]]] = []
        for poly in coords:
            if not isinstance(poly, list) or len(poly) == 0:
                continue
            exterior = poly[0]
            try:
                xy_src = _ring_to_xy(exterior)
                xy_planar = _transform_points(xy_src, fwd)
                rect_planar = _min_area_rect_from_ring_xy(xy_planar)
                rect_src = _transform_points(rect_planar, inv)
                ring = _rect_to_polygon_coords(rect_src)
                rects.append([[ring]])  # MultiPolygon: [ [ [ring], [hole]... ] ... ]
            except Exception:
                continue

        if not rects:
            return geom

        # rects currently: [ [[ring]] , [[ring]] ... ] but needs: [ [ring] ] per polygon
        mp = []
        for item in rects:
            # item == [[ring]]
            mp.append(item[0])  # [ring]
        return {"type": "MultiPolygon", "coordinates": mp}


def simplify_geojson(
    gj: Dict[str, Any],
    classes: Optional[List[str]],
    src_crs: str,
    dst_crs: str,
    export_line: bool,
    lane_line_mode: bool = False,
    line_merge_distance: float = 0.5,
    line_turn_threshold_deg: float = 35.0,
    line_samples: int = 32,
) -> Dict[str, Any]:
    _ensure_featurecollection(gj)

    # always_xy=True: treat input as (x, y) = (lon, lat) for EPSG:4326
    fwd = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    inv = Transformer.from_crs(dst_crs, src_crs, always_xy=True)

    class_set = set(classes) if classes else None

    out = json.loads(json.dumps(gj))  # deep copy via JSON
    features = out["features"]

    changed = 0
    skipped_type = 0
    skipped_class = 0

    for i, feat in enumerate(features):
        if not isinstance(feat, dict) or feat.get("type") != "Feature":
            continue

        cname = _get_class_name(feat)
        if class_set is not None and cname not in class_set:
            skipped_class += 1
            continue

        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            continue

        gtype = geom.get("type")
        if gtype == "Polygon":
            try:
                feat["geometry"] = _simplify_polygon_geometry(
                    geom,
                    fwd,
                    inv,
                    export_line,
                    lane_line_mode,
                    line_merge_distance,
                    line_turn_threshold_deg,
                    line_samples,
                )
                changed += 1
            except Exception:
                # 保守：壞幾何就不動
                continue
        elif gtype == "MultiPolygon":
            try:
                feat["geometry"] = _simplify_multipolygon_geometry(
                    geom,
                    fwd,
                    inv,
                    export_line,
                    lane_line_mode,
                    line_merge_distance,
                    line_turn_threshold_deg,
                    line_samples,
                )
                changed += 1
            except Exception:
                continue
        else:
            skipped_type += 1

    out.setdefault("properties", {})
    # 記錄一些處理資訊（不破壞既有語意）
    out["properties"]["simplify_note"] = {
        "changed_features": changed,
        "skipped_by_class": skipped_class,
        "skipped_by_geometry_type": skipped_type,
        "export_line": export_line,
        "src_crs": src_crs,
        "dst_crs": dst_crs,
        "classes_filter": classes or [],
        "lane_line_mode": lane_line_mode,
        "line_merge_distance": line_merge_distance,
        "line_turn_threshold_deg": line_turn_threshold_deg,
        "line_samples": line_samples,
    }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--json", required=True, help="輸入 GeoJSON (FeatureCollection)")
    ap.add_argument("-o", "--output", required=True, help="輸出 GeoJSON")
    ap.add_argument("-c", "--classes", nargs="*", default=None, help="只處理指定 class_name（可多個）")
    ap.add_argument("--src-crs", default="EPSG:4326", help="輸入座標系（預設 EPSG:4326）")
    ap.add_argument("--dst-crs", default="EPSG:3857", help="計算用平面座標系（預設 EPSG:3857）")
    ap.add_argument("--export-line", action="store_true", help="輸出 LineString / MultiLineString（取最長邊）")
    ap.add_argument(
        "--lane-line-mode",
        action="store_true",
        help="啟用道路標線模式：將長條 Polygon 轉中心線，並做近點合併簡化",
    )
    ap.add_argument(
        "--line-merge-distance",
        type=float,
        default=0.5,
        help="道路標線模式下，近點合併距離（單位為 dst-crs，如 EPSG:3857 為公尺）",
    )
    ap.add_argument(
        "--line-turn-threshold-deg",
        type=float,
        default=35.0,
        help="道路標線模式下，轉折點保留角度門檻（度）",
    )
    ap.add_argument(
        "--line-samples",
        type=int,
        default=32,
        help="道路標線模式下，中心線初始抽樣點數（越大越細）",
    )

    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        gj = json.load(f)

    out = simplify_geojson(
        gj=gj,
        classes=args.classes,
        src_crs=args.src_crs,
        dst_crs=args.dst_crs,
        export_line=args.export_line,
        lane_line_mode=args.lane_line_mode,
        line_merge_distance=args.line_merge_distance,
        line_turn_threshold_deg=args.line_turn_threshold_deg,
        line_samples=args.line_samples,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    note = out.get("properties", {}).get("simplify_note", {})
    print(
        f"已處理 GeoJSON，輸出到 {args.output} "
        f"(changed={note.get('changed_features')}, "
        f"skipped_class={note.get('skipped_by_class')}, "
        f"skipped_type={note.get('skipped_by_geometry_type')}, "
        f"export_line={note.get('export_line')}, "
        f"lane_line_mode={note.get('lane_line_mode')}, "
        f"line_merge_distance={note.get('line_merge_distance')}, "
        f"line_turn_threshold_deg={note.get('line_turn_threshold_deg')}, "
        f"line_samples={note.get('line_samples')}, "
        f"classes={note.get('classes_filter')})"
    )


if __name__ == "__main__":
    main()
