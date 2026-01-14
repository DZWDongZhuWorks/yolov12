import argparse
import json
from copy import deepcopy

import cv2
import numpy as np


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def polygon_to_min_rect(
    polygon: list[list[float]],
    min_aspect: float = 0.0,
) -> list[list[float]]:
    """
    把 polygon 簡化成旋轉矩形 (4 點)。
    若不符合 min_aspect（長邊/短邊），則回傳原 polygon。
    """

    pts = np.array(polygon, dtype=np.float32)

    # 點太少沒辦法算矩形，直接跳過
    if pts.shape[0] < 3:
        return polygon

    rect = cv2.minAreaRect(pts)  # ((cx, cy), (w, h), angle)
    (w, h) = rect[1]

    # 無效或退化的矩形
    if w <= 0 or h <= 0:
        return polygon

    long_side = max(w, h)
    short_side = min(w, h)

    if min_aspect > 0:
        aspect = long_side / (short_side + 1e-6)
        if aspect < min_aspect:
            # 不夠「長條」，保留原 polygon
            return polygon

    box = cv2.boxPoints(rect)  # 4x2
    box = box.tolist()  # [[x1,y1],...[x4,y4]]
    return box


def polygon_to_min_rect_and_line(
    polygon: list[list[float]],
    min_aspect: float = 0.0,
):
    """
    跟 polygon_to_min_rect 類似，但同時回傳：
    - rect_poly: 旋轉矩形四點 polygon（或原 polygon）
    - line_segment: 若有長條矩形，回傳長軸兩端點 [[x1,y1],[x2,y2]]，否則 None
    - line_vector: 若有長條矩形，回傳線段向量 [vx, vy]，否則 None
    """
    pts = np.array(polygon, dtype=np.float32)

    if pts.shape[0] < 3:
        # 無法算矩形，直接回傳原 polygon，且無線段資訊
        return polygon, None, None

    rect = cv2.minAreaRect(pts)  # ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), angle = rect

    if w <= 0 or h <= 0:
        return polygon, None, None

    long_side = max(w, h)
    short_side = min(w, h)

    if min_aspect > 0:
        aspect = long_side / (short_side + 1e-6)
        if aspect < min_aspect:
            # 不夠長條，保留原 polygon，不算線段
            return polygon, None, None

    # 旋轉矩形四點
    box = cv2.boxPoints(rect)
    box = box.tolist()

    # 計算長軸方向（沿著長邊）
    if w >= h:
        # 長邊沿著 rect angle
        theta = np.deg2rad(angle)
        L = w
    else:
        # 長邊在另一個方向，angle + 90 度
        theta = np.deg2rad(angle + 90.0)
        L = h

    # 單位方向向量
    ux = np.cos(theta)
    uy = np.sin(theta)

    # 從中心往兩側各走 L/2
    half_L = 0.5 * L
    dx = ux * half_L
    dy = uy * half_L

    p1 = [float(cx - dx), float(cy - dy)]
    p2 = [float(cx + dx), float(cy + dy)]
    line_segment = [p1, p2]
    line_vector = [p2[0] - p1[0], p2[1] - p1[1]]  # 長度約等於 L

    return box, line_segment, line_vector


def update_bbox_xyxy_from_polygon(polygon: list[list[float]]) -> list[float]:
    """
    根據 polygon（四點矩形或一般多邊形）更新 bbox_xyxy。
    """
    pts = np.array(polygon, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]

    x_min = float(xs.min())
    y_min = float(ys.min())
    x_max = float(xs.max())
    y_max = float(ys.max())

    return [x_min, y_min, x_max, y_max]


def simplify_json_polygons(
    data: dict,
    min_aspect: float = 0.0,
    export_line: bool = False,
    target_classes: list | None = None,
) -> dict:
    """
    對整個標註 JSON 做 polygon ➜ 旋轉矩形 簡化。

    - target_classes:
        * None 或空：所有物件都處理（原本行為）。
        * 非空：只處理 class_name / class_id 在此清單中的物件，其餘物件完全不變。

      例如：
        target_classes = ["white_line", "rectangle"]
        target_classes = ["0", "5"] 或 [0, 5]

    - export_line = False:
        polygons 會被替換為 4 點旋轉矩形（或原 polygon）。
    - export_line = True:
        若 polygon 成功被簡化為長條矩形，則 polygons 直接被
        「長軸線段兩端點」取代（[[x1,y1],[x2,y2]]），
        並在 line_vectors 裡存放對應向量 [vx, vy]。
        不符合 min_aspect 的 polygon 則保持原樣。

    回傳新的 data，不會修改原物件。
    """
    out = deepcopy(data)

    # 整理 target class 的名稱與 id
    name_set = None
    id_set = None
    if target_classes:
        name_set = set()
        id_set = set()
        for c in target_classes:
            # 允許傳 int 或 str
            if isinstance(c, (int, np.integer)):
                id_set.add(int(c))
            else:
                s = str(c)
                if s.isdigit():
                    id_set.add(int(s))
                else:
                    name_set.add(s)

    for obj in out.get("objects", []):
        # 若有指定 target_classes，先判斷是否需要處理
        if name_set is not None or id_set is not None:
            cid = obj.get("class_id")
            cname = obj.get("class_name")
            hit = False
            if name_set is not None and cname in name_set:
                hit = True
            if id_set is not None and cid in id_set:
                hit = True
            if not hit:
                # 不在指定類別裡，完全跳過（保持原 polygon / bbox）
                continue

        polys = obj.get("polygons", [])
        new_polys = []

        line_vectors = [] if export_line else None

        for poly in polys:
            if not poly:
                new_polys.append(poly)
                if export_line:
                    line_vectors.append(None)
                continue

            rect_poly, line_seg, line_vec = polygon_to_min_rect_and_line(
                poly, min_aspect=min_aspect
            )

            if export_line:
                # 若成功簡化為長條矩形，用長軸線段替代 polygon
                if line_seg is not None:
                    new_polys.append(line_seg)   # 2 點 line segment
                    line_vectors.append(line_vec)
                else:
                    # 不符合 min_aspect 或無法簡化，保留原 polygon
                    new_polys.append(poly)
                    line_vectors.append(None)
            else:
                # 原本行為：polygons 換成 4 點旋轉矩形 / 或原 polygon
                new_polys.append(rect_poly)

        obj["polygons"] = new_polys

        if export_line:
            obj["line_vectors"] = line_vectors

        # 這裡假設每個 object 只有一個 polygon 要簡化，
        # 若有多個，可考慮 merge or 分別算 bbox。
        if new_polys:
            # 先把所有 polygon 點合併後算 bbox
            all_pts = [p for poly in new_polys for p in poly]
            if all_pts:  # 避免全空
                obj["bbox_xyxy"] = update_bbox_xyxy_from_polygon(all_pts)

    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "將 JSON 中的 polygons 簡化成旋轉長條矩形；"
            "啟用 --export-line 時，直接用長軸線段取代 polygon"
        )
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="輸入 polygon JSON 路徑",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="輸出簡化後 JSON 路徑",
    )
    parser.add_argument(
        "--min-aspect",
        type=float,
        default=0.0,
        help=(
            "最小長寬比 (長邊/短邊)。"
            "若設 >0，只有長寬比 >= 此值 的 polygon 才會被簡化，"
            "否則保留原 polygon。預設 0 表示全部都簡化。"
        ),
    )
    parser.add_argument(
        "--export-line",
        action="store_true",
        help=(
            "啟用後：對符合 min-aspect 的 polygon，"
            "直接以長軸線段兩端點取代 polygons，"
            "並輸出對應線段向量到 line_vectors。"
        ),
    )
    parser.add_argument(
        "-c", "--class",
        dest="classes",
        nargs="+",
        help=(
            "只處理指定類別，可輸入多個。"
            "可接受 class_name（例如：white_line rectangle）"
            "或 class_id（例如：0 5）。未指定則處理所有類別。"
        ),
    )

    args = parser.parse_args()

    data = load_json(args.input)
    simplified = simplify_json_polygons(
        data,
        min_aspect=args.min_aspect,
        export_line=args.export_line,
        target_classes=args.classes,  # 新增：只處理指定類別
    )
    save_json(simplified, args.output)

    print(
        f"已將 {args.input} 的 polygons 簡化，"
        f"輸出到 {args.output}（export_line={args.export_line}，"
        f"classes={args.classes}）"
    )


if __name__ == "__main__":
    main()
