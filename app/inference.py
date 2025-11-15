"""
Inference модуль с поддержкой ТРЁХ моделей:
1. best.pt - для печатей (stamp)
2. best_qr.pt - для QR-кодов (qr)
3. best_sign.pt - для подписей (signature)

Логика как в tsue.py - просто model(img) для каждой модели.
"""
import os
from typing import List, Dict, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# Глобальные модели
STAMP_MODEL: Optional[YOLO] = None
STAMP_CLASS_ID: Optional[int] = None

QR_MODEL: Optional[YOLO] = None
QR_CLASS_ID: Optional[int] = None

SIGNATURE_MODEL: Optional[YOLO] = None
SIGNATURE_CLASS_ID: Optional[int] = None


def init_model(stamp_model_path: str, qr_model_path: Optional[str] = None, signature_model_path: Optional[str] = None) -> None:
    """
    Загружает модели YOLO для детекции печатей, QR-кодов и подписей.
    
    Args:
        stamp_model_path: Путь к модели для печатей (best.pt)
        qr_model_path: Путь к модели для QR-кодов (best_qr.pt), опционально
        signature_model_path: Путь к модели для подписей (best_sign.pt), опционально
    """
    global STAMP_MODEL, STAMP_CLASS_ID, QR_MODEL, QR_CLASS_ID, SIGNATURE_MODEL, SIGNATURE_CLASS_ID
    
    # ========== Модель для печатей ==========
    if not os.path.exists(stamp_model_path):
        raise FileNotFoundError(f"Модель печатей не найдена: {stamp_model_path}")
    
    print(f"Загрузка модели ПЕЧАТЕЙ из {stamp_model_path}...")
    STAMP_MODEL = YOLO(stamp_model_path)
    print("✓ Модель печатей загружена")
    
    # Находим класс 'stamp'
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
    
    # ========== Модель для QR-кодов ==========
    if qr_model_path and os.path.exists(qr_model_path):
        print(f"\nЗагрузка модели QR-КОДОВ из {qr_model_path}...")
        QR_MODEL = YOLO(qr_model_path)
        print("✓ Модель QR-кодов загружена")
        
        # Находим класс 'qr' или 'qr_code'
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
    
    # ========== Модель для подписей ==========
    if signature_model_path and os.path.exists(signature_model_path):
        print(f"\nЗагрузка модели ПОДПИСЕЙ из {signature_model_path}...")
        SIGNATURE_MODEL = YOLO(signature_model_path)
        print("✓ Модель подписей загружена")
        
        # Находим класс 'signature' или 'sign'
        SIGNATURE_CLASS_ID = None
        for k, v in SIGNATURE_MODEL.names.items():
            v_lower = v.lower()
            if 'sign' in v_lower:
                SIGNATURE_CLASS_ID = k
                break
        
        if SIGNATURE_CLASS_ID is None:
            print("⚠️ Класс подписи не найден!")
            print(f"Доступные классы: {list(SIGNATURE_MODEL.names.values())}")
        else:
            print(f"✓ Класс подписи найден с ID: {SIGNATURE_CLASS_ID} ('{SIGNATURE_MODEL.names[SIGNATURE_CLASS_ID]}')")
    else:
        print("\n⚠️ Модель подписей не указана или не найдена.")


def run_inference(image_bgr: np.ndarray, **kwargs) -> List[Dict]:
    """
    Обнаруживает печати, QR-коды и подписи на изображении.
    Запускает все три модели и объединяет результаты.
    
    Args:
        image_bgr: Изображение в формате BGR (OpenCV)
        **kwargs: Игнорируются (для совместимости)
    
    Returns:
        Список словарей с детекциями:
        {
            "class": "stamp", "qr" или "signature",
            "confidence": float,
            "bbox": [x, y, w, h]
        }
    """
    all_detections = []
    
    # ========== Детекция ПЕЧАТЕЙ ==========
    if STAMP_MODEL is not None and STAMP_CLASS_ID is not None:
        try:
            stamp_results = STAMP_MODEL(image_bgr)
            
            boxes = stamp_results[0].boxes.xyxy
            confidences = stamp_results[0].boxes.conf
            classes = stamp_results[0].boxes.cls
            
            for i, cls in enumerate(classes):
                if int(cls) == STAMP_CLASS_ID:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    conf = float(confidences[i])
                    
                    # Преобразуем xyxy -> xywh
                    w = x2 - x1
                    h = y2 - y1
                    
                    all_detections.append({
                        "class": "stamp",
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, w, h]
                    })
        except Exception as e:
            print(f"Ошибка детекции печатей: {e}")
    
    # ========== Детекция QR-КОДОВ ==========
    if QR_MODEL is not None and QR_CLASS_ID is not None:
        try:
            qr_results = QR_MODEL(image_bgr)
            
            boxes = qr_results[0].boxes.xyxy
            confidences = qr_results[0].boxes.conf
            classes = qr_results[0].boxes.cls
            
            for i, cls in enumerate(classes):
                if int(cls) == QR_CLASS_ID:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    conf = float(confidences[i])
                    
                    # Преобразуем xyxy -> xywh
                    w = x2 - x1
                    h = y2 - y1
                    
                    all_detections.append({
                        "class": "qr",
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, w, h]
                    })
        except Exception as e:
            print(f"Ошибка детекции QR-кодов: {e}")
    
    # ========== Детекция ПОДПИСЕЙ ==========
    if SIGNATURE_MODEL is not None and SIGNATURE_CLASS_ID is not None:
        try:
            sign_results = SIGNATURE_MODEL(image_bgr)
            
            boxes = sign_results[0].boxes.xyxy
            confidences = sign_results[0].boxes.conf
            classes = sign_results[0].boxes.cls
            
            for i, cls in enumerate(classes):
                if int(cls) == SIGNATURE_CLASS_ID:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    conf = float(confidences[i])
                    
                    # Преобразуем xyxy -> xywh
                    w = x2 - x1
                    h = y2 - y1
                    
                    all_detections.append({
                        "class": "signature",
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, w, h]
                    })
        except Exception as e:
            print(f"Ошибка детекции подписей: {e}")
    
    return all_detections


def draw_boxes(image_bgr: np.ndarray, objects: List[Dict]) -> np.ndarray:
    """
    Рисует боксы на изображении.
    Разные цвета для печатей, QR-кодов и подписей.
    """
    boxed_img = image_bgr.copy()
    
    # Убеждаемся, что изображение в правильном формате
    if len(boxed_img.shape) == 2:
        boxed_img = cv2.cvtColor(boxed_img, cv2.COLOR_GRAY2BGR)
    
    # Цвета для разных классов (BGR)
    colors = {
        "stamp": (0, 255, 0),       # Зеленый для печатей
        "qr": (255, 0, 255),        # Фиолетовый для QR-кодов
        "signature": (0, 255, 255), # Желтый для подписей
    }
    
    for i, obj in enumerate(objects):
        cls = obj.get("class", "unknown")
        bbox = obj.get("bbox", [0, 0, 0, 0])
        conf = obj.get("confidence", 0.0)
        
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        
        # Убеждаемся, что координаты валидны
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Координаты в пределах изображения
        img_h, img_w = boxed_img.shape[:2]
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        
        # Выбираем цвет
        color = colors.get(cls, (200, 200, 200))
        
        # Рисуем толстый прямоугольник (толщина линии 8)
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), color, 8)
        
        # Добавляем метку с крупным шрифтом
        label = f'{cls.upper()} ({conf:.2f})'
        cv2.putText(boxed_img, label, (x1, y1 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    return boxed_img
