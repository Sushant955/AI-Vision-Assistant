"""Module 5: Smart OCR with EasyOCR (pip only, no extra program needed)."""
import cv2
import numpy as np

LANGUAGES = {
    "English": ["en"],
    "English + Telugu": ["te", "en"],
    "English + Hindi": ["hi", "en"],
}

_readers = {}


def _reader(lang_key):
    import easyocr  # imported lazily because it is heavy
    if lang_key not in _readers:
        _readers[lang_key] = easyocr.Reader(LANGUAGES[lang_key], gpu=False)
    return _readers[lang_key]


def read_text(image_bgr, lang_key="English"):
    """Returns (annotated RGB image, list of {'text','confidence'})."""
    results = _reader(lang_key).readtext(image_bgr)
    annotated = image_bgr.copy()
    items = []
    for box, text, conf in results:
        pts = np.array([(int(x), int(y)) for x, y in box])
        cv2.polylines(annotated, [pts], True, (255, 184, 0), 2)
        items.append({"text": text, "confidence": round(float(conf) * 100, 1)})
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), items
