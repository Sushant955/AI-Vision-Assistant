"""Module 6: Text-to-speech (offline, using pyttsx3). Returns WAV bytes for st.audio."""
import os
import re
import tempfile


def speak_to_wav(text):
    """Returns (wav_bytes, error). Works offline on Windows."""
    text = re.sub(r"[*#_`>]+", "", text or "")
    if not text.strip():
        return None, "Nothing to speak."
    try:
        import pyttsx3
        path = os.path.join(tempfile.gettempdir(), "vision_assistant_tts.wav")
        if os.path.exists(path):
            os.remove(path)
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.save_to_file(text, path)
        engine.runAndWait()
        engine.stop()
        with open(path, "rb") as f:
            return f.read(), None
    except Exception as e:
        return None, f"Text-to-speech failed: {e}"
