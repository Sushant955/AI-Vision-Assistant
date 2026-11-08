"""Module 1: Object detection with selectable models (different datasets)."""
import glob
import os

import cv2
from ultralytics import YOLO

from modules import settings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_DIR = os.path.join(ROOT, "models")

BUILTIN = {
    "COCO 80 objects - Nano (fastest)": "yolov8n.pt",
    "COCO 80 objects - Small": "yolov8s.pt",
    "COCO 80 objects - Medium": "yolov8m.pt",
    "COCO 80 objects - XLarge (most accurate, slow)": "yolov8x.pt",
    "Open Images 600 objects - Small": "yolov8s-oiv7.pt",
    "Open Images 600 objects - Medium": "yolov8m-oiv7.pt",
    "Open Images 600 objects - XLarge (most accurate, slow)": "yolov8x-oiv7.pt",
    "Any object you type - YOLO-World": "world:yolov8s-worldv2.pt",
}
DEFAULT_WORLD = ("person", "laptop", "mobile phone", "bottle", "book", "backpack", "chair", "cup")

_cache = {}


def available_models():
    """label -> weights file. Includes any .pt files in the models/ folder."""
    models = dict(BUILTIN)
    for path in sorted(glob.glob(os.path.join(CUSTOM_DIR, "*.pt"))):
        models["Custom: " + os.path.basename(path)] = path
    return models


def label_for(weights, models=None):
    for label, w in (models or available_models()).items():
        if w == weights:
            return label
    return None


def get_model(name=None, classes_text="", fresh=False):
    name = name or settings.load()["model"]
    if name.startswith("world:"):
        weights = name.split(":", 1)[1]
        classes = tuple(c.strip() for c in classes_text.split(",") if c.strip()) or DEFAULT_WORLD
        key = ("world", weights, classes)
        if fresh or key not in _cache:
            from ultralytics import YOLOWorld  # needs CLIP; ultralytics installs it (Git required)
            model = YOLOWorld(weights)
            model.set_classes(list(classes))
            if fresh:
                return model
            _cache[key] = model
        return _cache[key]
    key = (name,)
    if fresh:
        return YOLO(name)
    if key not in _cache:
        _cache[key] = YOLO(name)  # downloads automatically on first use
    return _cache[key]


def load_model(name=None):  # kept for older code
    return get_model(name)


def detect_objects(image_bgr, conf=0.25, model_name=None, imgsz=640, augment=False,
                   classes_text="", iou=0.5):
    """Returns (annotated RGB image, list of {'label','confidence'})."""
    name = model_name or settings.load()["model"]
    model = get_model(name, classes_text)
    kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    if augment and not name.startswith("world:"):
        kwargs["augment"] = True  # test-time augmentation: slower, more accurate
    results = model.predict(image_bgr, **kwargs)[0]

    detections = []
    for box in results.boxes:
        detections.append({
            "label": results.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]) * 100, 1),
        })
    return cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB), detections
