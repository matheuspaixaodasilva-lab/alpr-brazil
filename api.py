"""
================================================================================
  ALPR - REST API
  Author : Matheus Paixão da Silva
  GitHub : https://github.com/matheuspaixaodasilva-lab
  Stack  : FastAPI + YOLOv8 + OpenCV + EasyOCR

  Endpoints:
    POST /detect  — receives an image, returns detected plates
    GET  /health  — returns API status
================================================================================
"""

import cv2
import numpy as np
import easyocr
import re
import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from projeto_placa import (
    detect_all_plates,
    crop_plate,
    preprocess_plate,
    run_ocr,
    clean_plate_text_multi,
)

# ──────────────────────────────────────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "license_plate_detector.pt"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found: '{MODEL_PATH}'")

print("Loading YOLO model...")
yolo_model = YOLO(MODEL_PATH)

print("Loading EasyOCR...")
ocr_reader = easyocr.Reader(["en"], gpu=False)

print("API ready.")

app = FastAPI(
    title="ALPR Brazil",
    description="Automatic License Plate Recognition for Brazilian plates.",
    version="1.0.0",
)


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Returns API status and loaded models."""
    return {
        "status": "ok",
        "model":  MODEL_PATH,
        "ocr":    "easyocr-en",
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    multiscale: bool = False,
):
    """
    Receives an image file and returns all detected plates.

    Parameters:
      - file: image file (JPEG, PNG, etc.)
      - multiscale: if True, runs YOLO in 3 passes to detect distant plates.
                    Slower (~3x) but catches small/distant plates.
                    Default: False (single pass, faster).

    Returns:
      - plates: list of detected plates with candidates and confidence scores
      - processing_time_ms: total processing time in milliseconds
      - image_size: width x height of the input image
      - mode: detection mode used ("multiscale" or "standard")
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.content_type}'. Must be an image."
        )

    start = time.time()

    # Read image from upload
    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    image    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    h_img, w_img = image.shape[:2]

    # Run detection pipeline
    if multiscale:
        # 3 passes: original + 2x upscaled + 2x2 tiles
        bboxes = detect_all_plates(yolo_model, image)
        mode = "multiscale"
    else:
        # Single pass: faster, best for close-range plates
        from projeto_placa import _yolo_scan, _nms
        bboxes = _nms(_yolo_scan(yolo_model, image, scale=1.0, offset=(0, 0)))
        mode = "standard"

    plates = []
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
        raw_text, conf   = run_ocr(ocr_reader, processed)
        plate, candidates = clean_plate_text_multi(raw_text)

        cleaned = re.sub(r"[^A-Z0-9]", "", plate)
        if not plate or len(cleaned) < 5 or conf < 0.10:
            continue

        plates.append({
            "plate":       plate,
            "candidates":  candidates,
            "ocr_raw":     raw_text,
            "confidence": {
                "yolo": round(float(bbox[4]), 4),
                "ocr":  round(float(conf), 4),
            },
            "bbox": {
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3]),
            }
        })

    elapsed_ms = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "plates":             plates,
        "total_detected":     len(plates),
        "processing_time_ms": elapsed_ms,
        "mode":               mode,
        "image_size": {
            "width":  w_img,
            "height": h_img,
        }
    })
