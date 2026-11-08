import io
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from modules import database as db
from modules import image_quality
from modules.detection import detect_objects
from modules.model_picker import sidebar_controls
from modules.scene_understanding import describe_from_detections
from modules.ui import apply_style, hero
from modules.voice_assistant import speak_to_wav

st.set_page_config(page_title="Image Analysis", page_icon="🖼️", layout="wide")
apply_style()
hero("Image Analysis", "Detect objects with the model and dataset you choose, check quality, hear a summary.", "🖼️")

opts = sidebar_controls()
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    try:
        with st.spinner("Loading the model and detecting (the first use of a model downloads it)..."):
            annotated, detections = detect_objects(image_bgr, **opts)
    except Exception as e:
        st.error(f"Detection failed: {e}")
        st.info("If you chose YOLO-World, it needs Git installed and internet. "
                "Try 'Open Images 600 objects' instead.")
        st.stop()

    key = f"{uploaded.name}-{uploaded.size}-{opts}"
    if st.session_state.get("last_logged") != key:
        db.log_detections("image", detections)
        st.session_state["last_logged"] = key

    col1, col2 = st.columns(2)
    col1.subheader("Original")
    col1.image(image, use_container_width=True)
    col2.subheader("Detected")
    col2.image(annotated, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Total objects detected", len(detections))
    c2.metric("Different object types", len({d["label"] for d in detections}))

    summary = describe_from_detections(detections)
    st.write(summary)

    if detections:
        counts = Counter(d["label"] for d in detections)
        table = [{"object": k, "count": v,
                  "avg confidence %": round(np.mean([d["confidence"] for d in detections
                                                      if d["label"] == k]), 1)}
                 for k, v in counts.most_common()]
        st.dataframe(table, use_container_width=True)

    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button("Download processed image", buf.getvalue(),
                       file_name="detected.png", mime="image/png")

    if st.button("🔊 Speak the summary"):
        audio, err = speak_to_wav(summary)
        if err:
            st.error(err)
        else:
            st.audio(audio, format="audio/wav")

    with st.expander("Image quality report"):
        q = image_quality.analyze(image_bgr)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sharpness", q["blur_score"])
        m2.metric("Brightness", q["brightness"])
        m3.metric("Noise", q["noise"])
        m4.metric("Contrast", q["contrast"])
        for issue in q["issues"]:
            st.warning(issue)
        if not q["issues"]:
            st.success("Image quality looks good.")
