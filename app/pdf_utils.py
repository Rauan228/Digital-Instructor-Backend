from typing import List, Optional

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:  # подсказка, если не установлено
    fitz = None


def is_pdf(filename: str, content_type: Optional[str]) -> bool:
    if filename and filename.lower().endswith(".pdf"):
        return True
    if content_type and content_type.lower() == "application/pdf":
        return True
    return False


def pdf_bytes_to_images(
    pdf_bytes: bytes,
    dpi: int = 300,
    as_jpeg: bool = True,
) -> List[np.ndarray]:
    """
    Конвертировать PDF-байты в список изображений (BGR, для OpenCV).

    По умолчанию рендерим страницы как JPEG (качество 90) и без альфы,
    чтобы привести формат к тренировочному домену и уменьшить артефакты.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (pymupdf) is not installed. Please add 'pymupdf' to requirements.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[np.ndarray] = []
    for page in doc:
        # Без альфы, чтобы избежать прозрачностей поверх объектов
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        if as_jpeg:
            # PyMuPDF tobytes("jpeg") не принимает качество в текущей версии
            img_bytes = pix.tobytes("jpeg")
        else:
            img_bytes = pix.tobytes("png")
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        images.append(img)
    doc.close()
    return images