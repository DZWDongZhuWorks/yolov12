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
        # 依照標註的 image size resize
        img = cv2.resize(img, (width, height))
    else:
        # 黑底畫布
        img = np.zeros((height, width, 3), dtype=np.uint8)

    return img


def draw_polygons_on_image(
    img: np.ndarray,
    data: dict,
    alpha: float = 0.4,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    draw_labels: bool = True,
):
    """
    在 img 上畫出 JSON 裡的 polygons + 標籤。
    - 若 polygon 有 3 點以上：當作多邊形填色 + 描邊
    - 若 polygon 只有 2 點：當作線段畫線
    會回傳一張新的影像（不直接改動原圖）。
    """
    overlay = img.copy()

    objects = data.get("objects", [])
    for obj in objects:
        class_name = obj.get("class_name", str(obj.get("class_id", "")))
        confidence = obj.get("confidence", None)
        polygons = obj.get("polygons", [])

        # 標籤內容：class_name + 信心值（如果有）
        if confidence is not None:
            label = f"{class_name} {confidence:.2f}"
        else:
            label = str(class_name)

        color = color_for_class(class_name)

        for poly in polygons:
            # poly: [[x1,y1], [x2,y2], ...]
            if not poly:
                continue

            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            num_pts = pts.shape[0]

            if num_pts >= 3:
                # ------- 多邊形模式 -------
                pts_poly = pts.reshape((-1, 1, 2))

                # 填色 + 描邊
                cv2.fillPoly(overlay, [pts_poly], color)
                cv2.polylines(
                    overlay, [pts_poly],
                    isClosed=True,
                    color=color,
                    thickness=line_thickness
                )

                # 若不畫標籤就跳過
                if not draw_labels:
                    continue

                # 以多邊形中心點放標籤
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))

            elif num_pts == 2:
                # ------- 線段模式（支援 export-line 後的 2 點線段） -------
                p1 = tuple(pts[0])
                p2 = tuple(pts[1])

                cv2.line(
                    overlay,
                    p1,
                    p2,
                    color=color,
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )

                if not draw_labels:
                    continue

                # 以線段中點放標籤
                cx = int((pts[0, 0] + pts[1, 0]) * 0.5)
                cy = int((pts[0, 1] + pts[1, 1]) * 0.5)
            else:
                # 只有一點 / 奇怪的資料，略過
                continue

            # ---- 以下是共用的畫標籤部分 ----
            # 避免文字剛好超出畫面
            cx = max(0, min(cx, img.shape[1] - 1))
            cy = max(0, min(cy, img.shape[0] - 1))

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            text_bg_tl = (cx, max(0, cy - th - baseline - 2))
            text_bg_br = (cx + tw + 2, cy + baseline + 2)

            cv2.rectangle(overlay, text_bg_tl, text_bg_br, (0, 0, 0), thickness=-1)
            cv2.putText(
                overlay,
                label,
                (cx + 1, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # 疊加半透明顏色
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return blended


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

    args = parser.parse_args()

    data = load_annotations(args.json)

    image_meta = data.get("image", {})
    if not image_meta:
        raise ValueError("JSON 中缺少 'image' 欄位 (width/height)")

    base_img = create_base_image(image_meta, args.background)
    result = draw_polygons_on_image(
        base_img,
        data,
        draw_labels=not args.no_label,
    )

    cv2.imwrite(args.output, result)
    print(f"已輸出結果到: {args.output}")

    if args.show:
        cv2.imshow("polygons", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
