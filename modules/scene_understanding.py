"""Modules 3 & 4: Scene description and Ask AI (cloud AI with offline fallbacks)."""
import base64
import io
import time
from collections import Counter

import requests

# Tried after the model chosen in Settings. "-latest" names are aliases Google keeps current.
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
RETRY_CODES = (429, 500, 502, 503, 504)
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def describe_from_detections(detections):
    """Offline fallback: build a sentence from YOLO results."""
    if not detections:
        return "I could not detect any known objects in this image."
    counts = Counter(d["label"] for d in detections)
    parts = [f"{n} {label}{'s' if n > 1 else ''}" for label, n in counts.most_common()]
    listing = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"This image contains {listing}."


def _to_jpeg_b64(pil_image, max_side=900):
    img = pil_image.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def list_gemini_models(key):
    """Returns (list of model names your key can use, error)."""
    try:
        r = requests.get(GEMINI_BASE, params={"key": key, "pageSize": 200}, timeout=30)
        data = r.json()
        if r.status_code != 200:
            return None, data.get("error", {}).get("message", f"HTTP {r.status_code}")
        names = [m["name"].replace("models/", "") for m in data.get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
        names = [n for n in names if "gemini" in n and
                 not any(x in n for x in ("embedding", "tts", "image", "live", "audio"))]
        return sorted(names), None
    except (requests.RequestException, ValueError) as e:
        return None, f"Could not reach Google: {e}"


def test_gemini(key, model):
    """Returns (ok, message) for a tiny text request."""
    try:
        r = requests.post(f"{GEMINI_BASE}/{model}:generateContent", params={"key": key},
                          json={"contents": [{"parts": [{"text": "Reply with the word OK."}]}]},
                          timeout=45)
        data = r.json()
        if r.status_code == 200:
            return True, data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return False, f"HTTP {r.status_code}: " + data.get("error", {}).get("message", "")
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        return False, f"Failed: {e}"


def _gemini(b64, question, key, first_model):
    models = [first_model] + [m for m in FALLBACK_MODELS if m != first_model]
    last_err = "Unknown error"
    for model in models:
        payload = {"contents": [{"parts": [
            {"text": question},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}]}
        for attempt in range(3):
            try:
                r = requests.post(f"{GEMINI_BASE}/{model}:generateContent",
                                  params={"key": key}, json=payload, timeout=45)
            except requests.RequestException as e:
                last_err = f"Network error: {e}"
                time.sleep(2)
                continue
            try:
                data = r.json()
            except ValueError:
                data = {}
            if r.status_code == 200:
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"], None
                except (KeyError, IndexError):
                    last_err = f"{model}: no text in the answer"
                    break
            msg = data.get("error", {}).get("message", f"HTTP {r.status_code}")
            last_err = f"{model}: {msg}"
            if r.status_code in RETRY_CODES:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (400, 401, 403):
                return None, msg
            break
    return None, last_err


def ask_about_image(pil_image, question, cfg):
    """Cloud AI. Returns (answer_text, error_text)."""
    provider = cfg.get("provider", "None")
    key = cfg.get("api_key", "").strip()
    if provider not in ("Gemini", "Claude") or not key:
        return None, "No cloud AI key set."
    b64 = _to_jpeg_b64(pil_image)
    try:
        if provider == "Gemini":
            return _gemini(b64, question, key, cfg["gemini_model"])
        payload = {"model": cfg["claude_model"], "max_tokens": 800,
                   "messages": [{"role": "user", "content": [
                       {"type": "image", "source": {"type": "base64",
                                                    "media_type": "image/jpeg", "data": b64}},
                       {"type": "text", "text": question}]}]}
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers,
                          json=payload, timeout=60)
        data = r.json()
        if r.status_code != 200:
            return None, data.get("error", {}).get("message", f"HTTP {r.status_code}")
        return data["content"][0]["text"], None
    except requests.RequestException as e:
        return None, f"Network error: {e}"
    except (KeyError, IndexError, ValueError) as e:
        return None, f"Unexpected response from the AI service: {e}"


def smart_answer(pil_image, question, cfg):
    """Cloud AI first, then the offline BLIP model. Returns (answer, source, notes)."""
    notes = []
    if cfg.get("provider") in ("Gemini", "Claude"):
        ans, err = ask_about_image(pil_image, question, cfg)
        if ans:
            return ans, cfg["provider"], notes
        notes.append(f"{cfg['provider']}: {err}")
    try:
        from modules import local_vision
        return local_vision.answer(pil_image, question), "Local AI (BLIP, offline)", notes
    except ImportError:
        notes.append("Offline AI is not installed. Run:  pip install transformers")
    except Exception as e:
        notes.append(f"Offline AI failed: {e}")
    return None, None, notes
