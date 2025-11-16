import base64
import json
import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import asyncio

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import init_model, run_inference, draw_boxes
from .pdf_utils import is_pdf, pdf_bytes_to_images

executor = ThreadPoolExecutor(max_workers=4)


APP_TITLE = "Digital Inspector API"
APP_VERSION = "0.1.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


def _get_cors_origins() -> List[str]:
    origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    return [o.strip() for o in origins_env.split(",") if o.strip()]

def _get_cors_origin_regex() -> str | None:
    return os.getenv("CORS_ORIGIN_REGEX", "https://.*\\.vercel\\.app")


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_origin_regex=_get_cors_origin_regex(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    
    default_stamp_path = os.path.join(root_dir, "best.pt")
    stamp_model_path = os.getenv("STAMP_MODEL_PATH", default_stamp_path)
    
    default_qr_path = os.path.join(root_dir, "best_qr.pt")
    qr_model_path = os.getenv("QR_MODEL_PATH", default_qr_path)
    
    env_signature_path = os.getenv("SIGNATURE_MODEL_PATH")
    if env_signature_path and env_signature_path.strip():
        signature_model_path = env_signature_path
    else:
        sign_auth_path = os.path.join(root_dir, "sign-auth.pt")
        best_sign_path = os.path.join(root_dir, "best_sign.pt")
        signature_model_path = sign_auth_path if os.path.exists(sign_auth_path) else best_sign_path
    
    try:
        init_model(stamp_model_path, qr_model_path, signature_model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize YOLO models: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


_ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
ANNOTATIONS_OUTPUT_DIR = os.getenv(
    "ANNOTATIONS_OUTPUT_DIR",
    os.path.join(_ROOT_DIR, "selected_output"),
)

_MASK_LABEL_MAP = {
    "stamp": "label_1",
    "signature": "label_2",
    "qr": "label_3",
    "auth": "label_4",
    "signauth": "label_5",
}


def _ensure_output_dir() -> None:
    os.makedirs(ANNOTATIONS_OUTPUT_DIR, exist_ok=True)


def _safe_load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _build_selected_annotations_for_file(file_name: str, pages: List[Dict]) -> Dict:
    file_block: Dict[str, Dict] = {}
    ann_counter = 1
    for page in pages:
        page_key = f"page_{page['page_index'] + 1}"
        page_block = {
            "annotations": [],
            "page_size": {"width": page["width"], "height": page["height"]},
        }
        for obj in page.get("objects", []):
            bbox = obj.get("bbox", [0, 0, 0, 0])
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            area = float(w) * float(h)
            entry = {
                f"annotation_{ann_counter}": {
                    "category": str(obj.get("class", "unknown")),
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "area": area,
                }
            }
            page_block["annotations"].append(entry)
            ann_counter += 1
        file_block[page_key] = page_block
    return {file_name: file_block}


def _build_masked_annotations_for_file(file_name: str, pages: List[Dict]) -> Dict:
    file_block: Dict[str, Dict] = {}
    ann_counter = 1
    for page in pages:
        page_key = f"page_{page['page_index'] + 1}"
        page_block = {
            "page_size": {"width": page["width"], "height": page["height"]},
            "annotations": [],
        }
        for obj in page.get("objects", []):
            cls = str(obj.get("class", "unknown")).lower()
            masked_label = _MASK_LABEL_MAP.get(cls, "label_99")
            bbox = obj.get("bbox", [0, 0, 0, 0])
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            area = float(w) * float(h)
            entry = {
                f"annotation_{ann_counter}": {
                    "category": masked_label,
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "area": area,
                }
            }
            page_block["annotations"].append(entry)
            ann_counter += 1
        file_block[page_key] = page_block
    return {file_name: file_block}


def _write_annotations_json(file_name: str, pages: List[Dict]) -> None:
    _ensure_output_dir()

    selected_path = os.path.join(ANNOTATIONS_OUTPUT_DIR, "selected_annotations.json")
    masked_path = os.path.join(ANNOTATIONS_OUTPUT_DIR, "masked_annotations.json")

    selected_payload = _build_selected_annotations_for_file(file_name, pages)
    masked_payload = _build_masked_annotations_for_file(file_name, pages)

    selected_all = _safe_load_json(selected_path)
    masked_all = _safe_load_json(masked_path)

    selected_all.update(selected_payload)
    masked_all.update(masked_payload)

    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selected_all, f, ensure_ascii=False, indent=2)
    with open(masked_path, "w", encoding="utf-8") as f:
        json.dump(masked_all, f, ensure_ascii=False, indent=2)


def _write_batch_annotations_json(file_results: List[Dict]) -> None:
    _ensure_output_dir()

    selected_batch: Dict[str, Dict] = {}
    masked_batch: Dict[str, Dict] = {}
    for fr in file_results:
        fname = fr.get("file_name")
        pages = fr.get("pages", [])
        if not fname:
            continue
        selected_batch.update(_build_selected_annotations_for_file(fname, pages))
        masked_batch.update(_build_masked_annotations_for_file(fname, pages))

    selected_path = os.path.join(ANNOTATIONS_OUTPUT_DIR, "selected_annotations_batch.json")
    masked_path = os.path.join(ANNOTATIONS_OUTPUT_DIR, "masked_annotations_batch.json")

    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selected_batch, f, ensure_ascii=False, indent=2)
    with open(masked_path, "w", encoding="utf-8") as f:
        json.dump(masked_batch, f, ensure_ascii=False, indent=2)


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _encode_image_to_base64(image_bgr: np.ndarray, quality: int = 70, max_dimension: int = 1600) -> str:
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(".jpg", image_bgr, encode_params)
    if not ok:
        raise ValueError("Failed to encode image to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    try:
        results = []
        
        for file in files:
            data = await file.read()
            pages_imgs: List[np.ndarray]
            if is_pdf(file.filename, file.content_type):
                pages_imgs = pdf_bytes_to_images(data, dpi=300, as_jpeg=True)
            else:
                pages_imgs = [_decode_image_bytes(data)]

            pages_out: List[Dict] = []
            conf_thr = float(os.getenv("DETECT_CONF", 0.25))
            for idx, img_bgr in enumerate(pages_imgs):
                objects = run_inference(img_bgr, conf=conf_thr)
                h, w = img_bgr.shape[:2]
                pages_out.append({
                    "page_index": idx,
                    "width": w,
                    "height": h,
                    "objects": objects,
                })
            
            _write_annotations_json(file.filename, pages_out)

            results.append({
                "file_name": file.filename,
                "pages": pages_out,
            })
        if results:
            _write_batch_annotations_json(results)
        
        return {"files": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _process_single_page(idx: int, img_bgr: np.ndarray) -> Dict:
    conf_thr = float(os.getenv("DETECT_CONF", 0.25))
    objects = run_inference(img_bgr, conf=conf_thr)
    annotated = draw_boxes(img_bgr, objects)
    b64 = _encode_image_to_base64(annotated, quality=60, max_dimension=1400)
    h, w = img_bgr.shape[:2]
    return {
        "page_index": idx,
        "width": w,
        "height": h,
        "objects": objects,
        "image_base64": b64,
    }


@app.post("/api/analyze/annotated")
async def analyze_annotated(files: List[UploadFile] = File(...)):
    try:
        results = []
        loop = asyncio.get_event_loop()
        
        for file in files:
            data = await file.read()
            pages_imgs: List[np.ndarray]
            if is_pdf(file.filename, file.content_type):
                pages_imgs = pdf_bytes_to_images(data, dpi=200, as_jpeg=True)
            else:
                pages_imgs = [_decode_image_bytes(data)]

            futures = []
            for idx, img_bgr in enumerate(pages_imgs):
                future = loop.run_in_executor(executor, _process_single_page, idx, img_bgr)
                futures.append(future)
            
            pages_out = await asyncio.gather(*futures)

            _write_annotations_json(file.filename, list(pages_out))
            
            results.append({
                "file_name": file.filename,
                "pages": list(pages_out),
            })
        if results:
            _write_batch_annotations_json(results)
        
        return {"files": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))