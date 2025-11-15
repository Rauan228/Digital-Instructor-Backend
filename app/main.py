import base64
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

# Пул потоков для параллельной обработки
executor = ThreadPoolExecutor(max_workers=4)


APP_TITLE = "Digital Inspector API"
APP_VERSION = "0.1.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


def _get_cors_origins() -> List[str]:
    origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    return [o.strip() for o in origins_env.split(",") if o.strip()]

def _get_cors_origin_regex() -> str | None:
    # Позволяет разрешить домены вида *.vercel.app без явного перечисления
    return os.getenv("CORS_ORIGIN_REGEX")


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
    # Определяем пути к моделям относительно директории backend/app
    root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    
    # Модель для печатей (обязательная)
    default_stamp_path = os.path.join(root_dir, "best.pt")
    stamp_model_path = os.getenv("STAMP_MODEL_PATH", default_stamp_path)
    
    # Модель для QR-кодов (опциональная)
    default_qr_path = os.path.join(root_dir, "best_qr.pt")
    qr_model_path = os.getenv("QR_MODEL_PATH", default_qr_path)
    
    # Модель для подписей (опциональная)
    default_signature_path = os.path.join(root_dir, "best_sign.pt")
    signature_model_path = os.getenv("SIGNATURE_MODEL_PATH", default_signature_path)
    
    try:
        init_model(stamp_model_path, qr_model_path, signature_model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize YOLO models: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _encode_image_to_base64(image_bgr: np.ndarray, quality: int = 85, max_dimension: int = 1920) -> str:
    """
    Кодирует изображение в base64 с оптимизацией размера.
    
    Args:
        image_bgr: Изображение BGR
        quality: Качество JPEG (1-100), меньше = меньше размер
        max_dimension: Максимальная ширина/высота для ресайза
    """
    # Ресайзим если изображение слишком большое
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Используем JPEG вместо PNG для меньшего размера
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(".jpg", image_bgr, encode_params)
    if not ok:
        raise ValueError("Failed to encode image to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    """
    Анализ одного или нескольких файлов.
    Возвращает результаты для каждого файла.
    """
    try:
        results = []
        
        for file in files:
            data = await file.read()
            pages_imgs: List[np.ndarray]
            if is_pdf(file.filename, file.content_type):
                # Приводим PDF к JPEG, DPI=300 — ближе к тренировочному домену
                pages_imgs = pdf_bytes_to_images(data, dpi=300, as_jpeg=True)
            else:
                pages_imgs = [_decode_image_bytes(data)]

            pages_out: List[Dict] = []
            for idx, img_bgr in enumerate(pages_imgs):
                # ПРОСТО model(img) - как в tsue.py и Google Colab!
                objects = run_inference(img_bgr)
                h, w = img_bgr.shape[:2]
                pages_out.append({
                    "page_index": idx,
                    "width": w,
                    "height": h,
                    "objects": objects,
                })
            
            results.append({
                "file_name": file.filename,
                "pages": pages_out,
            })
        
        return {"files": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _process_single_page(idx: int, img_bgr: np.ndarray) -> Dict:
    """
    Обрабатывает одну страницу (для параллельной обработки).
    """
    objects = run_inference(img_bgr)
    annotated = draw_boxes(img_bgr, objects)
    # Более агрессивное сжатие для превью
    b64 = _encode_image_to_base64(annotated, quality=75, max_dimension=1600)
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
    """
    Анализ одного или нескольких файлов с аннотированными изображениями.
    Возвращает результаты с base64-изображениями для каждого файла.
    Оптимизировано для обработки множества файлов.
    """
    try:
        results = []
        loop = asyncio.get_event_loop()
        
        for file in files:
            data = await file.read()
            pages_imgs: List[np.ndarray]
            if is_pdf(file.filename, file.content_type):
                # Используем DPI=200 вместо 300 для ускорения
                pages_imgs = pdf_bytes_to_images(data, dpi=200, as_jpeg=True)
            else:
                pages_imgs = [_decode_image_bytes(data)]

            # Параллельная обработка страниц
            futures = []
            for idx, img_bgr in enumerate(pages_imgs):
                future = loop.run_in_executor(executor, _process_single_page, idx, img_bgr)
                futures.append(future)
            
            # Ждем завершения всех страниц
            pages_out = await asyncio.gather(*futures)
            
            results.append({
                "file_name": file.filename,
                "pages": list(pages_out),
            })
        
        return {"files": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))