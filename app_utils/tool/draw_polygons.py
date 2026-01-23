import argparse
import json
import os

import cv2
import numpy as np
import hashlib


def load_annotations(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def color_for_class(class_name: str):
    """
    根據 class_name 做一個穩定的 BGR 顏色（hash 出來）。
    """
    h = int(hashlib.md5(class_name.encode("utf-8")).hexdigest()[:6], 16)
    b = h & 0xFF
    g = (h >> 8) & 0xFF
    r = (h >> 16) & 0xFF
    return (b, g, r)


def create_base_image(meta: dict, background_path: str | None):
    width = int(meta["width"])
    height = int(meta["height"])

    if background_path:
        img = cv2.imread(background_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"無法讀取底圖：{background_path}")
        img = cv2.resize(img, (width, height))
    else:
        img = np.zeros((height, width, 3), dtype=np.uint8)

    return img


def draw_polygons_on_image(
    img: np.ndarray,
    data: dict,
    alpha: float = 0.4,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    draw_labels: bool = True,
    fill_polygons: bool = False,         # ✅ 預設改為不填色（只畫邊線）
    line_alpha: float = 1.0,             # ✅ 線條透明度（1.0=不透明（建議））
):
    """
    在 img 上畫出 JSON 裡的 polygons + 標籤。
    - polygon >= 3：預設只畫邊線（避免凹多邊形填色造成誤差）；若 fill_polygons=True 才填色
    - polygon == 2：畫線段
    回傳一張新的影像（不直接改動原圖）。
    """
    # 用兩層 overlay：一層專門畫線/填色，最後再以 alpha 或 line_alpha 疊回原圖
    overlay = img.copy()          # 用於填色（若需要）
    line_layer = img.copy()       # 用於線段/邊線（可獨立控制透明度）

    objects = data.get("objects", [])
    for obj in objects:
        class_name = obj.get("class_name", str(obj.get("class_id", "")))
        confidence = obj.get("confidence", None)
        polygons = obj.get("polygons", [])

        if confidence is not None:
            label = f"{class_name} {confidence:.2f}"
        else:
            label = str(class_name)

        color = color_for_class(class_name)

        for poly in polygons:
            if not poly:
                continue

            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            num_pts = pts.shape[0]

            if num_pts >= 3:
                pts_poly = pts.reshape((-1, 1, 2))

                # ✅ 只畫邊線（輪廓），避免凹多邊形/自交造成填色誤差
                cv2.polylines(
                    line_layer,
                    [pts_poly],
                    isClosed=True,
                    color=color,
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )

                # （可選）需要時才填色
                if fill_polygons:
                    cv2.fillPoly(overlay, [pts_poly], color)

                if not draw_labels:
                    continue

                # ✅ 標籤位置：用外接矩形中心，比平均值更不容易落到多邊形外
                x, y, w, h = cv2.boundingRect(pts_poly)
                cx = x + w // 2
                cy = y + h // 2

            elif num_pts == 2:
                p1 = tuple(pts[0])
                p2 = tuple(pts[1])

                cv2.line(
                    line_layer,
                    p1,
                    p2,
                    color=color,
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )

                if not draw_labels:
                    continue

                cx = int((pts[0, 0] + pts[1, 0]) * 0.5)
                cy = int((pts[0, 1] + pts[1, 1]) * 0.5)
            else:
                continue

            # ---- 畫標籤（共用）----
            cx = max(0, min(cx, img.shape[1] - 1))
            cy = max(0, min(cy, img.shape[0] - 1))

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            text_bg_tl = (cx, max(0, cy - th - baseline - 2))
            text_bg_br = (min(img.shape[1] - 1, cx + tw + 2), min(img.shape[0] - 1, cy + baseline + 2))

            # 標籤背景/文字建議畫在 line_layer 上（確保在邊線之上）
            # --- 標籤背景使用 class 顏色 ---
            cv2.rectangle(line_layer, text_bg_tl, text_bg_tl := text_bg_br, color, thickness=-1)

            # --- 根據背景亮度自動選字色 ---
            b, g, r = color
            luminance = 0.299*r + 0.587*g + 0.114*b
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)

            cv2.putText(
                line_layer,
                label,
                (cx + 1, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )


    # ---- 合成：先把（可選的）填色用 alpha 疊回，再把線條用 line_alpha 疊回 ----
    out = img.copy()

    if fill_polygons:
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    # 線條通常建議 1.0 不透明；若你想要半透明，可調 line_alpha < 1
    if line_alpha >= 1.0:
        out = line_layer
    else:
        out = cv2.addWeighted(line_layer, line_alpha, out, 1 - line_alpha, 0)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="使用 OpenCV 將 JSON 標註中的 polygons/線段 & 標籤畫在圖上"
    )
    parser.add_argument(
        "-j", "--json",
        required=True,
        help="標註 JSON 路徑（包含 image / objects / polygons 結構）",
    )
    parser.add_argument(
        "-bkg", "--background",
        help="(可選) 底圖路徑，若不給則用黑底畫布，尺寸依 JSON image.width/height",
    )
    parser.add_argument(
        "-o", "--output",
        default="output.png",
        help="輸出影像路徑（預設：output.png）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="加上此參數時，畫完會用視窗顯示結果",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="加上此參數時，不繪製標籤文字",
    )

    # ✅ 新增：若你仍想保留「填色」能力，可用參數打開
    parser.add_argument(
        "--fill",
        action="store_true",
        help="加上此參數時，對 polygon 進行填色（預設只畫邊線以避免凹多邊形誤差）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="填色透明度（僅在 --fill 時生效，預設 0.4）",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=1.0,
        help="邊線/線段透明度（預設 1.0 不透明）",
    )

    args = parser.parse_args()

    data = load_annotations(args.json)

    image_meta = data.get("image", {})
    if not image_meta:
        raise ValueError("JSON 中缺少 'image' 欄位 (width/height)")

    base_img = create_base_image(image_meta, args.background)
    result = draw_polygons_on_image(
        base_img,
        data,
        alpha=args.alpha,
        draw_labels=not args.no_label,
        fill_polygons=args.fill,
        line_alpha=args.line_alpha,
    )

    cv2.imwrite(args.output, result)
    print(f"已輸出結果到: {args.output}")

    if args.show:
        cv2.imshow("polygons", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
