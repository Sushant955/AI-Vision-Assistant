"""Simple settings stored in settings.json (kept out of GitHub by .gitignore)."""
import json
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")

DEFAULTS = {
    "model": "yolov8n.pt",
    "confidence": 0.30,
    "provider": "None",          # None | Gemini | Claude
    "api_key": "",
    "gemini_model": "gemini-2.5-flash",
    "claude_model": "claude-sonnet-5-5",
}


def load():
    data = dict(DEFAULTS)
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            data.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return data


def save(data):
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
