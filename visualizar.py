"""
================================================================================
  ALPR - Result Visualizer
  Generates annotated images with bounding boxes and detected plates.
  Author : Matheus Paixão da Silva
  GitHub : https://github.com/matheuspaixaodasilva-lab
================================================================================
"""

import cv2
import numpy as np
import os
import glob
import re
import easyocr
from datetime import datetime
from ultralytics import YOLO

from projeto_placa import (
    detect_all_plates,
    crop_plate,
    preprocess_plate,
    run_ocr,
    clean_plate_text_multi,
)

MODEL_PATH    = "license_plate_detector.pt"
IMAGES_FOLDER = "imagens_teste"
OUTPUT_FOLDER = f"resultados_visuais_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def draw_result(image, bbox, plate, candidates, conf_yolo, conf_ocr):
    """
    Draws bounding box, plate text, confidence scores and alternatives on the image.
    Box color reflects OCR confidence: green (>=50%), orange (>=30%), red (<30%).
    """
    img = image.copy()
    x1, y1, x2, y2, _ = bbox
    h_img, w_img = img.shape[:2]

    # Color by OCR confidence
    if conf_ocr >= 0.5:
        color = (0, 200, 0)      # green
    elif conf_ocr >= 0.3:
        color = (0, 165, 255)    # orange
    else:
        color = (0, 0, 220)      # red

    # Bounding box
    thickness = max(2, int(min(w_img, h_img) * 0.004))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Corner accents
    corner = int((x2 - x1) * 0.15)
    t = thickness + 1
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (px, py), (px + dx * corner, py), color, t)
        cv2.line(img, (px, py), (px, py + dy * corner), color, t)

    # Plate text label
    font   = cv2.FONT_HERSHEY_DUPLEX
    scale  = max(0.8, min(2.0, (x2 - x1) / 180))
    thick  = max(1, int(scale * 1.5))
    label  = plate if plate else "?"
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

    margin = 8
    lx = x1
    ly = y1 - th - margin * 2
    if ly < 0:
        ly = y2 + margin

    cv2.rectangle(img, (lx, ly), (lx + tw + margin*2, ly + th + margin*2), color, -1)
    cv2.putText(img, label, (lx + margin, ly + th + margin),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    # Confidence scores
    conf_text = f"YOLO:{conf_yolo:.0%}  OCR:{conf_ocr:.0%}"
    cs = max(0.4, scale * 0.55)
    (cw, ch), _ = cv2.getTextSize(conf_text, font, cs, 1)
    cy = y2 + ch + margin
    if cy > h_img:
        cy = y1 - margin
    cv2.rectangle(img, (x1, cy - ch - 4), (x1 + cw + 8, cy + 4), (30, 30, 30), -1)
    cv2.putText(img, conf_text, (x1 + 4, cy),
                font, cs, (200, 200, 200), 1, cv2.LINE_AA)

    # Alternative candidates
    alt = [c for c in candidates
           if c != plate and len(re.sub(r"[^A-Z0-9]", "", c)) >= 6]
    if alt:
        alt_text = "Alt: " + " | ".join(alt[:3])
        als = max(0.35, scale * 0.45)
        (aw, ah), _ = cv2.getTextSize(alt_text, font, als, 1)
        ay = cy + ah + margin
        if ay < h_img:
            cv2.rectangle(img, (x1, ay - ah - 4), (x1 + aw + 8, ay + 4), (50, 50, 50), -1)
            cv2.putText(img, alt_text, (x1 + 4, ay),
                        font, als, (180, 180, 180), 1, cv2.LINE_AA)

    return img


def process_and_visualize(model, reader, image_path, output_folder):
    """Processes a single image and saves the annotated result."""
    filename     = os.path.basename(image_path)
    name_no_ext  = os.path.splitext(filename)[0]

    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERROR] Could not read: {filename}")
        return

    print(f"\n[{filename}]")
    bboxes = detect_all_plates(model, image)

    if not bboxes:
        print(f"  No plate detected.")
        img_out = image.copy()
        cv2.putText(img_out, "No plate detected",
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (0, 0, 220), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_folder, f"{name_no_ext}_result.jpg"), img_out)
        return

    annotated = image.copy()
    print(f"  {len(bboxes)} plate(s) found.")

    for i, bbox in enumerate(bboxes):
        crop = crop_plate(image, bbox)
        if crop.size == 0:
            continue

        # Aspect ratio filter
        h_crop, w_crop = crop.shape[:2]
        if h_crop == 0:
            continue
        ratio = w_crop / h_crop
        if ratio < 0.8 or ratio > 8.0:
            continue

        processed        = preprocess_plate(crop)
        raw_text, conf   = run_ocr(reader, processed)
        plate, candidates = clean_plate_text_multi(raw_text)

        if not plate or len(re.sub(r"[^A-Z0-9]", "", plate)) < 5:
            continue

        print(f"  Plate {i+1}: {plate}  (raw: {raw_text}, conf: {conf:.1%})")

        annotated = draw_result(annotated, bbox, plate, candidates, bbox[4], conf)

        # Save individual plate crop
        cv2.imwrite(
            os.path.join(output_folder, f"{name_no_ext}_plate{i+1}.jpg"),
            crop
        )

    out_path = os.path.join(output_folder, f"{name_no_ext}_result.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"  Saved: {out_path}")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: '{MODEL_PATH}'")
        return
    if not os.path.isdir(IMAGES_FOLDER):
        print(f"[ERROR] Folder not found: '{IMAGES_FOLDER}/'")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False)
    print("Ready.")

    extensions  = ["*.jpg", "*.jpeg", "*.png", "*.bmp",
                   "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = sorted(set(
        p for ext in extensions
        for p in glob.glob(os.path.join(IMAGES_FOLDER, ext))
    ))

    if not image_paths:
        print(f"[WARNING] No images found in '{IMAGES_FOLDER}/'")
        return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Generating visualizations for {len(image_paths)} image(s)")
    print(sep)

    for path in image_paths:
        process_and_visualize(model, reader, path, OUTPUT_FOLDER)

    print(f"\n{sep}")
    print(f"  Results saved to: {OUTPUT_FOLDER}/")
    print(sep)


if __name__ == "__main__":
    main()
