import time
from collections import Counter

import cv2
import streamlit as st

from modules import database as db
from modules import detection
from modules.detection import detect_objects
from modules.model_picker import sidebar_controls
from modules.ui import apply_style, hero

st.set_page_config(page_title="Live Camera", page_icon="📷", layout="wide")
apply_style()
hero("Live Camera", "Real-time detection with the dataset you choose in the sidebar.", "📷")

opts = sidebar_controls(live=True)
cam_index = st.sidebar.number_input("Camera index", 0, 5, 0)
run = st.checkbox("Start camera")

frame_box = st.empty()
info_box = st.empty()

if run:
    try:
        with st.spinner("Loading the model (the first use downloads it)..."):
            detection.get_model(opts["model_name"], opts["classes_text"])
    except Exception as e:
        st.error(f"Could not load the model: {e}")
        st.stop()

    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Could not open the camera. Close other apps using it and try again.")
    else:
        last_log = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    st.warning("Camera stopped sending frames.")
                    break
                annotated, detections = detect_objects(frame, **opts)
                frame_box.image(annotated, use_container_width=True)
                counts = Counter(d["label"] for d in detections)
                info_box.markdown("**In view:** " + (", ".join(f"{n} {l}" for l, n in counts.most_common())
                                                      or "nothing detected"))
                if time.time() - last_log > 10:
                    db.log_detections("camera", detections)
                    last_log = time.time()
        finally:
            cap.release()
else:
    st.info("Tick **Start camera** to begin. Untick it to stop.")
