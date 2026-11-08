import cv2
import numpy as np
import streamlit as st

from modules.ui import apply_style, hero
from PIL import Image

from modules.ocr import LANGUAGES, read_text
from modules.voice_assistant import speak_to_wav

st.set_page_config(page_title="Text Scanner", page_icon="📄", layout="wide")
apply_style()
hero("Text Scanner", "Read documents, labels and signboards in English, Telugu or Hindi.", "📄")
st.caption("Reads documents, labels and signboards. The first run downloads the OCR model (needs internet).")

lang = st.sidebar.selectbox("Language", list(LANGUAGES.keys()))
source = st.radio("Source", ["Upload image", "Use camera"], horizontal=True)

file = (st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if source == "Upload image" else st.camera_input("Take a photo"))

if file:
    image = Image.open(file).convert("RGB")
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Reading text..."):
        try:
            annotated, items = read_text(bgr, lang)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()

    c1, c2 = st.columns(2)
    c1.image(image, caption="Original", use_container_width=True)
    c2.image(annotated, caption="Text found", use_container_width=True)

    if items:
        full_text = "\n".join(i["text"] for i in items)
        st.subheader("Extracted text")
        st.text_area("Text", full_text, height=200)
        st.dataframe(items, use_container_width=True)
        st.download_button("Download text", full_text, file_name="scanned_text.txt")
        if st.button("🔊 Read aloud"):
            audio, err = speak_to_wav(full_text)
            if err:
                st.error(err)
            else:
                st.audio(audio, format="audio/wav")
    else:
        st.info("No text found. Try a clearer, closer photo.")
