import os
from typing import List, Dict, Optional
import requests

import cv2
import numpy as np
from ultralytics import YOLO
import numpy as np
import cv2


STAMP_MODEL: Optional[YOLO] = None
STAMP_CLASS_ID: Optional[int] = None

QR_MODEL: Optional[YOLO] = None
QR_CLASS_ID: Optional[int] = None

SIGNATURE_MODEL: Optional[YOLO] = None
SIGNATURE_CLASS_ID: Optional[int] = None
AUTHOR_CLASS_ID: Optional[int] = None
SIGNAUTH_CLASS_ID: Optional[int] = None

SIGNATURE_ROBOFLOW_CFG: Optional[Dict] = None


def _warmup_yolo_model(model: YOLO) -> None:
    try:
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        model.predict(dummy, conf=0.01, verbose=False)
    except Exception as e:
        print(f"⚠️ Ошибка прогрева модели {getattr(model, 'model', 'YOLO')}: {type(e).__name__}: {e}")


def init_model(stamp_model_path: str, qr_model_path: Optional[str] = None, signature_model_path: Optional[str] = None) -> None:
    global STAMP_MODEL, STAMP_CLASS_ID, QR_MODEL, QR_CLASS_ID, SIGNATURE_MODEL, SIGNATURE_CLASS_ID, AUTHOR_CLASS_ID, SIGNAUTH_CLASS_ID
    
    if not os.path.exists(stamp_model_path):
        raise FileNotFoundError(f"Модель печатей не найдена: {stamp_model_path}")
    
    print(f"Загрузка модели ПЕЧАТЕЙ из {stamp_model_path}...")
    STAMP_MODEL = YOLO(stamp_model_path)
    print("✓ Модель печатей загружена")
    _warmup_yolo_model(STAMP_MODEL)
    
    STAMP_CLASS_ID = None
    for k, v in STAMP_MODEL.names.items():
        if v.lower() == "stamp":
            STAMP_CLASS_ID = k
            break
    
    if STAMP_CLASS_ID is None:
        print("⚠️ Класс 'stamp' не найден!")
        print(f"Доступные классы: {list(STAMP_MODEL.names.values())}")
    else:
        print(f"✓ Класс 'stamp' найден с ID: {STAMP_CLASS_ID}")
    
    if qr_model_path and os.path.exists(qr_model_path):
        print(f"\nЗагрузка модели QR-КОДОВ из {qr_model_path}...")
        QR_MODEL = YOLO(qr_model_path)
        print("✓ Модель QR-кодов загружена")
        _warmup_yolo_model(QR_MODEL)
        
        QR_CLASS_ID = None
        for k, v in QR_MODEL.names.items():
            v_lower = v.lower()
            if 'qr' in v_lower:
                QR_CLASS_ID = k
                break
        
        if QR_CLASS_ID is None:
            print("⚠️ Класс QR не найден!")
            print(f"Доступные классы: {list(QR_MODEL.names.values())}")
        else:
            print(f"✓ Класс QR найден с ID: {QR_CLASS_ID} ('{QR_MODEL.names[QR_CLASS_ID]}')")
    else:
        print("\n⚠️ Модель QR-кодов не указана или не найдена.")
    
    rf_model_id = os.getenv("ROBOFLOW_MODEL_ID") or os.getenv("SIGNATURE_ROBOFLOW_MODEL")
    rf_api_key = os.getenv("ROBOFLOW_API_KEY")
    rf_endpoint = os.getenv("SIGNATURE_ROBOFLOW_ENDPOINT")
    rf_conf = os.getenv("ROBOFLOW_CONF") or os.getenv("SIGNATURE_ROBOFLOW_CONF") or "0.25"
    rf_overlap = os.getenv("ROBOFLOW_OVERLAP") or "30"
    rf_format = os.getenv("ROBOFLOW_FORMAT") or "json"
    if rf_model_id and rf_api_key:
        endpoint = rf_endpoint or (
            f"https://detect.roboflow.com/{rf_model_id}?api_key={rf_api_key}"
            f"&format={rf_format}&confidence={rf_conf}&overlap={rf_overlap}"
        )
        SIGNATURE_ROBOFLOW_CFG = {
            "endpoint": endpoint,
            "model_id": rf_model_id,
            "api_key": rf_api_key,
            "confidence": rf_conf,
            "overlap": rf_overlap,
            "format": rf_format,
        }
        print(f"\n✓ Включён Roboflow-инференс для ПОДПИСЕЙ: {rf_model_id}")
        print(f"Endpoint: {endpoint}")
        SIGNATURE_MODEL = None
        SIGNATURE_CLASS_ID = 0
    else:
        if signature_model_path and os.path.exists(signature_model_path):
            print(f"\nЗагрузка модели ПОДПИСЕЙ из {signature_model_path}...")
            SIGNATURE_MODEL = YOLO(signature_model_path)
            print("✓ Модель подписей загружена")
            _warmup_yolo_model(SIGNATURE_MODEL)
            
            SIGNATURE_CLASS_ID = None
            AUTHOR_CLASS_ID = None
            SIGNAUTH_CLASS_ID = None
            for k, v in SIGNATURE_MODEL.names.items():
                v_lower = v.lower()
                if 'sign' in v_lower and ('signature' in v_lower or v_lower == 'sign'):
                    SIGNATURE_CLASS_ID = k
                if 'author' in v_lower or v_lower == 'auth':
                    AUTHOR_CLASS_ID = k
                if v_lower in ('signauth', 'sign-auth', 'sign_auth'):
                    SIGNAUTH_CLASS_ID = k
            
            if SIGNATURE_CLASS_ID is None:
                print("⚠️ Класс подписи не найден!")
                print(f"Доступные классы: {list(SIGNATURE_MODEL.names.values())}")
            else:
                print(f"✓ Класс подписи найден с ID: {SIGNATURE_CLASS_ID} ('{SIGNATURE_MODEL.names[SIGNATURE_CLASS_ID]}')")
            if AUTHOR_CLASS_ID is None:
                print("⚠️ Класс автора не найден!")
            else:
                print(f"✓ Класс автора найден с ID: {AUTHOR_CLASS_ID} ('{SIGNATURE_MODEL.names[AUTHOR_CLASS_ID]}')")
            if SIGNAUTH_CLASS_ID is None:
                print("ℹ️ Общий бокс 'signauth' не найден (необязателен)")
            else:
                print(f"✓ Общий бокс 'signauth' найден с ID: {SIGNAUTH_CLASS_ID} ('{SIGNATURE_MODEL.names[SIGNAUTH_CLASS_ID]}')")
        else:
            print("\n⚠️ Модель подписей не указана или не найдена.")


def run_inference(image_bgr: np.ndarray, **kwargs) -> List[Dict]:
    all_detections = []
    try:
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError(f"image_bgr is not numpy array, got {type(image_bgr)}")
        if image_bgr.dtype != np.uint8:
            image_bgr = image_bgr.astype(np.uint8)
        image_bgr = np.ascontiguousarray(image_bgr)
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"Ошибка подготовки изображения: {type(e).__name__}: {e}")
    
    conf_thr = float(kwargs.get("conf", os.getenv("DETECT_CONF", 0.25)))

    if STAMP_MODEL is not None and STAMP_CLASS_ID is not None:
        try:
            stamp_results = STAMP_MODEL(image_bgr, conf=conf_thr)
            if not stamp_results or stamp_results[0] is None:
                raise RuntimeError("Stamp model returned empty result")
            boxes = stamp_results[0].boxes.xyxy
            confidences = stamp_results[0].boxes.conf
            classes = stamp_results[0].boxes.cls
            
            if boxes is None or confidences is None or classes is None:
                raise RuntimeError("Stamp boxes/conf/classes are None")
            
            for i, cls in enumerate(classes):
                if int(cls) == STAMP_CLASS_ID:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    conf = float(confidences[i])
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    all_detections.append({
                        "class": "stamp",
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, w, h]
                    })
        except Exception as e:
            import traceback
            print(f"Ошибка детекции печатей: {type(e).__name__}: {e}")
            traceback.print_exc()
    
    if QR_MODEL is not None and QR_CLASS_ID is not None:
        try:
            qr_results = QR_MODEL(image_bgr, conf=conf_thr)
            
            boxes = qr_results[0].boxes.xyxy
            confidences = qr_results[0].boxes.conf
            classes = qr_results[0].boxes.cls
            
            for i, cls in enumerate(classes):
                if int(cls) == QR_CLASS_ID:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    conf = float(confidences[i])
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    all_detections.append({
                        "class": "qr",
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, w, h]
                    })
        except Exception as e:
            import traceback
            print(f"Ошибка детекции QR-кодов: {type(e).__name__}: {e}")
            traceback.print_exc()
    
    if SIGNATURE_MODEL is not None and (SIGNATURE_CLASS_ID is not None or AUTHOR_CLASS_ID is not None or SIGNAUTH_CLASS_ID is not None):
        try:
            sign_results = SIGNATURE_MODEL(image_bgr, conf=conf_thr)
            boxes = sign_results[0].boxes.xyxy
            confidences = sign_results[0].boxes.conf
            classes = sign_results[0].boxes.cls
            for i, cls in enumerate(classes):
                icls = int(cls)
                if SIGNATURE_CLASS_ID is not None and icls == SIGNATURE_CLASS_ID:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    conf = float(confidences[i])
                    w = x2 - x1
                    h = y2 - y1
                    all_detections.append({
                        "class": "signature",
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, w, h]
                    })
                elif AUTHOR_CLASS_ID is not None and icls == AUTHOR_CLASS_ID:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    conf = float(confidences[i])
                    w = x2 - x1
                    h = y2 - y1
                    all_detections.append({
                        "class": "auth",
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, w, h]
                    })
                elif SIGNAUTH_CLASS_ID is not None and icls == SIGNAUTH_CLASS_ID:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    conf = float(confidences[i])
                    w = x2 - x1
                    h = y2 - y1
                    all_detections.append({
                        "class": "signauth",
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, w, h]
                    })
        except Exception as e:
            import traceback
            print(f"Ошибка детекции подписи/автора (YOLO): {type(e).__name__}: {e}")
            traceback.print_exc()
    elif SIGNATURE_ROBOFLOW_CFG is not None:
        try:
            ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("Failed to encode image to JPEG for Roboflow")
            files = {"file": ("image.jpg", buf.tobytes(), "image/jpeg")}
            response = requests.post(SIGNATURE_ROBOFLOW_CFG["endpoint"], files=files, timeout=20)
            response.raise_for_status()
            data = response.json()
            preds = data.get("predictions", [])
            for p in preds:
                cx = float(p.get("x", 0))
                cy = float(p.get("y", 0))
                w = float(p.get("width", 0))
                h = float(p.get("height", 0))
                conf = float(p.get("confidence", 0))
                cls_name = str(p.get("class", "signature")).lower()
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                all_detections.append({
                    "class": "auth" if cls_name in ("author", "auth") else "signature",
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, int(w), int(h)]
                })
        except Exception as e:
            import traceback
            print(f"Ошибка детекции подписей (Roboflow): {type(e).__name__}: {e}")
            traceback.print_exc()
    
    return all_detections


def draw_boxes(image_bgr: np.ndarray, objects: List[Dict]) -> np.ndarray:
    boxed_img = image_bgr.copy()
    
    if len(boxed_img.shape) == 2:
        boxed_img = cv2.cvtColor(boxed_img, cv2.COLOR_GRAY2BGR)
    
    colors = {
        "stamp": (0, 255, 0),
        "qr": (255, 0, 255),
        "signature": (0, 255, 0),
        "auth": (0, 0, 255),
        "signauth": (255, 0, 0),
    }
    
    for i, obj in enumerate(objects):
        cls = obj.get("class", "unknown")
        if str(cls).lower() in ("signauth", "sign-auth", "sign_auth"):
            continue
        bbox = obj.get("bbox", [0, 0, 0, 0])
        conf = obj.get("confidence", 0.0)
        
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        img_h, img_w = boxed_img.shape[:2]
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        
        color = colors.get(cls, (200, 200, 200))
        
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), color, 8)
        
        label = f'{cls.upper()} ({conf:.2f})'
        cv2.putText(boxed_img, label, (x1, y1 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    return boxed_img
