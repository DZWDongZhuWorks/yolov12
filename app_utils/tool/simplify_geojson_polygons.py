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


def _simplify_polygon_geometry(
    geom: Dict[str, Any],
    fwd: Transformer,
    inv: Transformer,
    export_line: bool,
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

    rect_planar = _min_area_rect_from_ring_xy(xy_planar)  # 4x2 planar

    if export_line:
        p1, p2 = _rect_to_longest_edge_line(rect_planar)
        line_planar = np.stack([p1, p2], axis=0)
        line_src = _transform_points(line_planar, inv)
        return {
            "type": "LineString",
            "coordinates": line_src.tolist(),
        }
    else:
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
                rect_planar = _min_area_rect_from_ring_xy(xy_planar)
                p1, p2 = _rect_to_longest_edge_line(rect_planar)
                line_src = _transform_points(np.stack([p1, p2], axis=0), inv)
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
                feat["geometry"] = _simplify_polygon_geometry(geom, fwd, inv, export_line)
                changed += 1
            except Exception:
                # 保守：壞幾何就不動
                continue
        elif gtype == "MultiPolygon":
            try:
                feat["geometry"] = _simplify_multipolygon_geometry(geom, fwd, inv, export_line)
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

    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        gj = json.load(f)

    out = simplify_geojson(
        gj=gj,
        classes=args.classes,
        src_crs=args.src_crs,
        dst_crs=args.dst_crs,
        export_line=args.export_line,
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
        f"classes={note.get('classes_filter')})"
    )


if __name__ == "__main__":
    main()
