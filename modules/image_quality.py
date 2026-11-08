"""Module 7: Measure image quality and enhance images."""
import cv2
import numpy as np


def analyze(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())     # low = blurry
    brightness = float(gray.mean())                                 # 0-255
    noise = float(np.abs(gray.astype(np.float32) -
                         cv2.medianBlur(gray, 3).astype(np.float32)).mean())
    contrast = float(gray.std())

    issues = []
    if blur_score < 100:
        issues.append("Image looks blurry")
    if brightness < 70:
        issues.append("Image is too dark")
    elif brightness > 190:
        issues.append("Image is too bright")
    if noise > 6:
        issues.append("Image looks noisy")
    if contrast < 35:
        issues.append("Low contrast")

    return {"blur_score": round(blur_score, 1), "brightness": round(brightness, 1),
            "noise": round(noise, 2), "contrast": round(contrast, 1), "issues": issues}


def auto_params(q):
    """Choose enhancement settings from the quality report (always a visible change)."""
    brightness = 0.0
    if q["brightness"] < 95:
        brightness = min(100.0, (118 - q["brightness"]) * 1.4)
    elif q["brightness"] > 175:
        brightness = -min(50.0, (q["brightness"] - 160) * 0.8)
    contrast = 1.0 + min(0.4, (55 - q["contrast"]) / 70) if q["contrast"] < 50 else 1.08
    noisy = q["noise"] > 5
    return {
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 2),
        "saturation": 1.1,
        # sharpening amplifies noise, so sharpen less when the photo is noisy
        "sharpness": 0.3 if noisy else (1.2 if q["blur_score"] < 100 else 0.6),
        "denoise": int(min(15, 5 + q["noise"] * 0.5)) if noisy else 0,
        "clahe": q["contrast"] < 45 or q["brightness"] < 80,
        "upscale": 1,
    }


def apply_adjustments(img, brightness=0, contrast=1.0, saturation=1.0, sharpness=0.0,
                      denoise=0, clahe=False, upscale=1):
    out = img.copy()
    if denoise and denoise > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None, float(denoise), float(denoise), 7, 21)
    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, bb = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, bb)), cv2.COLOR_LAB2BGR)
    if contrast != 1.0 or brightness != 0:
        # contrast is applied around mid-grey (128) so it does not shift overall brightness
        out = cv2.convertScaleAbs(out, alpha=float(contrast),
                                  beta=float(brightness) + 128.0 * (1.0 - float(contrast)))
    if saturation != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if upscale and upscale > 1:
        out = cv2.resize(out, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    if sharpness and sharpness > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = cv2.addWeighted(out, 1 + sharpness, blur, -sharpness, 0)
    return out
