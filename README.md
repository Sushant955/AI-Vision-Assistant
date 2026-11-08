# AI Vision Assistant

An AI application that sees images and video, detects and tracks objects, reads text,
answers questions about images, speaks results aloud and keeps an analytics history.

## Features
| Page | What it does | Tech |
|---|---|---|
| Image Analysis | Detect objects, boxes, confidence, quality report, download result | YOLOv8, OpenCV |
| Live Camera | Real-time detection from webcam | YOLOv8, OpenCV |
| Video Analysis | Object tracking with unique IDs | ByteTrack |
| Ask AI | Questions about an image (Gemini or Claude API, offline fallback) | Vision-language model |
| Text Scanner | OCR in English, Telugu, Hindi | EasyOCR |
| Image Enhancement | Detect and fix dark, noisy, low-contrast, soft images | OpenCV |
| Detection History | Charts and CSV reports | SQLite, Pandas |
| Settings | Model, confidence, AI provider | JSON |

## Setup (Windows)
```
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
app.py            Home dashboard
pages/            One file per page (Streamlit multipage)
modules/          Reusable logic (detection, tracking, OCR, quality, voice, database)
database/         SQLite history (created automatically)
outputs/          Tracked videos
```

## Notes
- First run downloads YOLO weights; first OCR run downloads OCR models (internet needed).
- Ask AI needs an API key (Settings page). Without it, a basic offline description is used.
- Never commit `settings.json` (it holds your API key). It is already in `.gitignore`.
