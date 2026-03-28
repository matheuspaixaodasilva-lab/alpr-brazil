"""
================================================================================
  ALPR - Automatic License Plate Recognition System
  Author : Matheus Paixão da Silva
  GitHub : https://github.com/matheuspaixaodasilva-lab
  Stack  : YOLOv8 + OpenCV + EasyOCR
================================================================================
"""

import cv2
import numpy as np
import easyocr
import re
import os
import glob
from itertools import combinations, product
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH      = "license_plate_detector.pt"
IMAGES_FOLDER   = "imagens_teste"
YOLO_CONFIDENCE = 0.10  # Low threshold to capture distant/partially visible plates

# Brazilian plate regex patterns
PLATE_PATTERNS = [
    r"[A-Z]{3}[0-9][A-Z][0-9]{2}",   # Mercosul (2018+): AAA0A00
    r"[A-Z]{3}[0-9]{4}",             # Old standard:     AAA0000
    r"[A-Z]{3}-[0-9]{4}",            # Old with hyphen:  AAA-0000
]

# Characters allowed by EasyOCR — full alphanumeric set
PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Position-based correction maps
# Applied when OCR reads a letter where a digit is expected, and vice versa
CHAR_FIX_NUMERO = {
    "O": "0", "I": "1", "B": "8", "S": "5",
    "G": "6", "D": "0", "A": "4", "Z": "2",
    "P": "9", "J": "1", "T": "7", "R": "9", "Q": "0"
}
CHAR_FIX_LETRA = {
    "0": "O", "1": "I", "8": "B", "5": "S", "6": "G"
}

# Characters corrected at letter positions (V is not used in Brazilian plates)
CORR_LETRA_OCR = {
    "V": "U",
}

# Visually similar character pairs used to generate plate candidates
# Each key maps to a list of possible substitutions
SIMILARES_MULTI = {
    "O": ["U", "Q"],
    "Q": ["U", "O"],
    "U": ["O", "Q"],
    "B": ["8"], "8": ["B"],
    "S": ["5"], "5": ["S"],
    "Z": ["2"], "2": ["Z", "9"], "9": ["2"],
    "T": ["Z"],
    "G": ["6"], "6": ["G"],
    "D": ["0"],
    "I": ["1"], "1": ["I"],
    "0": ["Q"],
}


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-SCALE DETECTION
# Runs YOLO in 3 passes to handle plates at varying distances:
#   1) Full image
#   2) 2x upscaled image
#   3) 2x2 tiles with 20% overlap, each upscaled 2x
# Results are merged and deduplicated via Non-Maximum Suppression (NMS).
# ──────────────────────────────────────────────────────────────────────────────

def detect_all_plates(model, image_bgr):
    h, w = image_bgr.shape[:2]
    deteccoes = []

    # Pass 1: original resolution
    deteccoes += _yolo_scan(model, image_bgr, scale=1.0, offset=(0, 0))

    # Pass 2: full image upscaled 2x
    img2x = cv2.resize(image_bgr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    deteccoes += _yolo_scan(model, img2x, scale=2.0, offset=(0, 0))

    # Pass 3: 2x2 tiles with 20% overlap, each upscaled 2x
    tile_w, tile_h = w // 2, h // 2
    ox = int(tile_w * 0.2)
    oy = int(tile_h * 0.2)
    for row in range(2):
        for col in range(2):
            x1 = max(0, col * tile_w - ox)
            y1 = max(0, row * tile_h - oy)
            x2 = min(w, x1 + tile_w + ox * 2)
            y2 = min(h, y1 + tile_h + oy * 2)
            tile = image_bgr[y1:y2, x1:x2]
            tile2x = cv2.resize(tile, (tile.shape[1]*2, tile.shape[0]*2),
                                interpolation=cv2.INTER_CUBIC)
            deteccoes += _yolo_scan(model, tile2x, scale=2.0, offset=(x1, y1))

    return _nms(deteccoes)


def _yolo_scan(model, img, scale, offset):
    """Runs YOLO on a single image and converts bboxes back to original coordinates."""
    results = model(img, conf=YOLO_CONFIDENCE, verbose=False)
    deteccoes = []
    ox, oy = offset
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            deteccoes.append((
                int(x1 / scale) + ox,
                int(y1 / scale) + oy,
                int(x2 / scale) + ox,
                int(y2 / scale) + oy,
                conf
            ))
    return deteccoes


def _iou(a, b):
    """Calculates Intersection over Union between two bounding boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _nms(deteccoes, iou_thresh=0.4):
    """Non-Maximum Suppression: removes duplicate detections from overlapping passes."""
    if not deteccoes:
        return []
    deteccoes = sorted(deteccoes, key=lambda x: x[4], reverse=True)
    mantidos = []
    for det in deteccoes:
        if all(_iou(det, m) < iou_thresh for m in mantidos):
            mantidos.append(det)
    return mantidos


# ──────────────────────────────────────────────────────────────────────────────
# CROP AND PRE-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def crop_plate(image_bgr, bbox):
    """Crops the plate region with a small padding around the bounding box."""
    x1, y1, x2, y2, _ = bbox
    h_img, w_img = image_bgr.shape[:2]
    pad_x = int((x2 - x1) * 0.04)
    pad_y = int((y2 - y1) * 0.04)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w_img, x2 + pad_x); y2 = min(h_img, y2 + pad_y)
    return image_bgr[y1:y2, x1:x2]


def preprocess_plate(crop_bgr):
    """
    Prepares the plate crop for OCR:
      1. Removes the Mercosul blue band (top 20%)
      2. Upscales to at least 400px wide
      3. Applies sharpening kernel
      4. Boosts contrast slightly
    """
    h, w = crop_bgr.shape[:2]

    # Remove Mercosul top band
    crop_bgr = crop_bgr[int(h * 0.20):h, :]

    # Upscale to minimum readable resolution
    th, tw = crop_bgr.shape[:2]
    scale = max(400 / tw, 120 / th, 1.0)
    crop_bgr = cv2.resize(crop_bgr,
                          (int(tw * scale), int(th * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # Sharpening kernel
    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]], dtype=np.float32)
    crop_bgr = cv2.filter2D(crop_bgr, -1, kernel)

    # Mild contrast boost
    crop_bgr = cv2.convertScaleAbs(crop_bgr, alpha=1.4, beta=10)

    return crop_bgr


# ──────────────────────────────────────────────────────────────────────────────
# OCR
# Runs EasyOCR on 4 pre-processing variants and returns the highest-confidence result.
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_variante(reader, img, allowlist):
    """Runs EasyOCR on a single image variant."""
    try:
        res = reader.readtext(
            img,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
            batch_size=1,
        )
        if not res:
            return "", 0.0
        melhor = max(res, key=lambda x: x[2])
        return melhor[1].upper().strip(), melhor[2]
    except Exception:
        return "", 0.0


def run_ocr(reader, processed_img):
    """
    Tests 4 image variants and returns the result with highest OCR confidence:
      v1 - Otsu binarization       (best for sharp, high-contrast plates)
      v2 - Original color image    (best for well-lit plates)
      v3 - CLAHE + Otsu            (best for shadowed or uneven lighting)
      v4 - High contrast           (best for faded plates)
    """
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY) \
           if len(processed_img.shape) == 3 else processed_img

    # v1: Otsu binarization
    _, v1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if v1.mean() < 127:
        v1 = cv2.bitwise_not(v1)

    # v2: original color
    v2 = processed_img

    # v3: CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_cl = clahe.apply(gray)
    _, v3 = cv2.threshold(gray_cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if v3.mean() < 127:
        v3 = cv2.bitwise_not(v3)

    # v4: high contrast
    v4 = cv2.convertScaleAbs(processed_img, alpha=2.0, beta=0)

    variantes = [
        ("Otsu",          v1),
        ("Color",         v2),
        ("CLAHE+Otsu",    v3),
        ("HighContrast",  v4),
    ]

    melhor_texto, melhor_conf, melhor_nome = "", 0.0, ""
    for nome, v in variantes:
        texto, conf = _ocr_variante(reader, v, PLATE_CHARS)
        if conf > melhor_conf:
            melhor_conf  = conf
            melhor_texto = texto
            melhor_nome  = nome

    print(f"  -> EasyOCR ({melhor_nome}): [{melhor_texto}] conf={melhor_conf:.1%}")
    return melhor_texto, melhor_conf


# ──────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING AND VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def corrigir_por_posicao(texto):
    """
    Applies position-based correction using Brazilian plate structure:
      Mercosul : L L L N L N N  (L=letter, N=number)
      Old      : L L L N N N N

    Corrects characters that are in the wrong position type,
    e.g. 'B' where a digit is expected → '8'
    """
    t = re.sub(r"[^A-Z0-9]", "", texto)
    if len(t) != 7:
        return t
    eh_mercosul = t[4].isalpha() or t[4] in CHAR_FIX_LETRA
    posicoes = ["L","L","L","N","L","N","N"] if eh_mercosul else ["L","L","L","N","N","N","N"]
    resultado = []
    for ch, tipo in zip(t, posicoes):
        if tipo == "N" and ch.isalpha():
            resultado.append(CHAR_FIX_NUMERO.get(ch, ch))
        elif tipo == "L" and ch.isdigit():
            resultado.append(CHAR_FIX_LETRA.get(ch, ch))
        elif tipo == "L" and ch in CORR_LETRA_OCR:
            resultado.append(CORR_LETRA_OCR[ch])
        else:
            resultado.append(ch)
    return "".join(resultado)


def _tentar_padroes(texto):
    """Tries to match text against all known Brazilian plate patterns."""
    for p in PLATE_PATTERNS:
        m = re.search(p, texto)
        if m:
            return m.group(0)
    return None


def clean_plate_text_multi(raw_text):
    """
    Validates and corrects OCR output. Returns (best_candidate, all_candidates).

    Strategy:
      1. Apply position-based correction
      2. Try to match plate pattern directly
      3. If no match, generate candidates by substituting visually similar chars
         (up to 3 simultaneous substitutions)
      4. If only 6 chars detected, attempt to complete the 7th
    """
    cleaned = re.sub(r"[^A-Z0-9-]", "", raw_text)
    candidatos = []

    if len(cleaned) >= 7:
        corrigido = corrigir_por_posicao(cleaned[:7])
        posicoes_trocaveis = [(i, ch) for i, ch in enumerate(corrigido)
                              if ch in SIMILARES_MULTI]

        for n in range(0, min(4, len(posicoes_trocaveis) + 1)):
            if n == 0:
                r = _tentar_padroes(corrigido)
                if r and r not in candidatos:
                    candidatos.append(r)
            else:
                for combo in combinations(posicoes_trocaveis, n):
                    cands_pos = [SIMILARES_MULTI[ch] for _, ch in combo]
                    for subs in product(*cands_pos):
                        v = list(corrigido)
                        for (i, _), s in zip(combo, subs):
                            v[i] = s
                        r = _tentar_padroes(corrigir_por_posicao("".join(v)))
                        if r and r not in candidatos:
                            candidatos.append(r)

    r = _tentar_padroes(cleaned)
    if r and r not in candidatos:
        candidatos.append(r)

    if candidatos:
        if len(candidatos) > 1:
            def pontuacao(cand):
                """Scores candidates by similarity to the original OCR output."""
                score = 0
                c = re.sub(r"[^A-Z0-9]", "", cleaned)
                for a, b in zip(cand, c[:len(cand)]):
                    if a == b:
                        score += 3
                    elif a in SIMILARES_MULTI.get(b, []):
                        score += 1
                return score
            candidatos.sort(key=pontuacao, reverse=True)
        return candidatos[0], candidatos

    # Fallback: try to complete a partial 6-char read
    if len(cleaned) == 6:
        base = corrigir_por_posicao(cleaned + "0")[:6]
        for suffix in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            r = _tentar_padroes(corrigir_por_posicao(base + suffix))
            if r:
                return r + "?", [r + "?"]

    return cleaned if cleaned else "", [cleaned] if cleaned else []


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def process_image(model, reader, image_path):
    """Runs the full ALPR pipeline on a single image."""
    filename = os.path.basename(image_path)
    result   = {"arquivo": filename, "placas": [], "status": "OK"}

    image = cv2.imread(image_path)
    if image is None:
        result["status"] = "ERROR: could not read image"
        return result

    print(f"  -> Multi-scale detection...")
    bboxes = detect_all_plates(model, image)

    if not bboxes:
        result["status"] = "ERROR: no plate detected"
        return result

    print(f"  -> {len(bboxes)} plate(s) found.")

    for i, bbox in enumerate(bboxes):
        crop = crop_plate(image, bbox)
        if crop.size == 0:
            continue

        # Aspect ratio filter: Brazilian plates are roughly 2:1 to 5:1 (w:h)
        h_crop, w_crop = crop.shape[:2]
        if h_crop == 0:
            continue
        ratio = w_crop / h_crop
        if ratio < 0.8 or ratio > 8.0:
            continue

        processed = preprocess_plate(crop)
        raw_text, conf_ocr = run_ocr(reader, processed)
        final, todos = clean_plate_text_multi(raw_text)

        cleaned_final = re.sub(r"[^A-Z0-9]", "", final)
        if final and len(cleaned_final) >= 5 and conf_ocr >= 0.10:
            result["placas"].append({
                "placa":      final,
                "candidatos": todos,
                "ocr_bruto":  raw_text,
                "conf_yolo":  bbox[4],
                "conf_ocr":   conf_ocr,
            })

    if not result["placas"]:
        result["status"] = "WARNING: detected but OCR returned no result"
    return result


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: '{MODEL_PATH}'")
        return
    if not os.path.isdir(IMAGES_FOLDER):
        print(f"[ERROR] Folder not found: '{IMAGES_FOLDER}/'")
        return

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False)
    print("Ready.\n")

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
    print(sep)
    print(f"  ALPR — Processing {len(image_paths)} image(s)")
    print(sep)

    all_results = []
    for path in image_paths:
        print(f"\n[{os.path.basename(path)}]")
        res = process_image(model, reader, path)
        all_results.append(res)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  FINAL SUMMARY")
    print(sep)

    for i, res in enumerate(all_results, 1):
        status_icon = "OK" if res["placas"] else "ERROR"
        print(f"\n  [{status_icon}] Image {i}: {res['arquivo']}")
        print(f"  {'-' * 40}")

        if res["placas"]:
            for j, p in enumerate(res["placas"], 1):
                cands = p.get("candidatos", [])
                alt   = [c for c in cands
                         if c != p["placa"]
                         and len(re.sub(r"[^A-Z0-9]", "", c)) >= 6]

                print(f"    Plate {j}:")
                if alt:
                    todos = [p["placa"]] + alt[:3]
                    print(f"      Candidates  : {' | '.join(todos)}")
                    print(f"      Note        : U/O/Q ambiguity — correct plate is in the list")
                else:
                    print(f"      Reading     : {p['placa']}")
                print(f"      OCR raw     : {p['ocr_bruto']}")
                print(f"      YOLO conf.  : {p['conf_yolo']:.1%}")
                print(f"      OCR conf.   : {p['conf_ocr']:.1%}")
                break  # one result per plate bbox
        else:
            print(f"    No plate identified")
            print(f"    Status: {res['status']}")

    print(f"\n{sep}")
    total  = len(all_results)
    lidas  = sum(1 for r in all_results if r["placas"])
    falhas = total - lidas
    print(f"  Total images    : {total}")
    print(f"  Plates read     : {lidas}")
    print(f"  No result       : {falhas}")
    print(sep)


if __name__ == "__main__":
    main()
