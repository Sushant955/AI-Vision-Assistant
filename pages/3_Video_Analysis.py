import os
import tempfile

import streamlit as st

from modules import database as db
from modules.model_picker import sidebar_controls
from modules.tracking import analyze_video
from modules.ui import apply_style, hero

st.set_page_config(page_title="Video Analysis", page_icon="🎞️", layout="wide")
apply_style()
hero("Video Analysis & Tracking", "Each object keeps the same ID across frames, so unique objects can be counted.", "🎞️")

opts = sidebar_controls()
skip = st.sidebar.slider("Process every Nth frame", 1, 6, 2, help="Higher = faster, less smooth")
max_frames = st.sidebar.slider("Max frames to process", 50, 1500, 400, 50)

video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if video and st.button("Analyze video", type="primary"):
    tmp_in = os.path.join(tempfile.gettempdir(), "va_input_" + video.name)
    with open(tmp_in, "wb") as f:
        f.write(video.getbuffer())
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "outputs", "tracked_video.mp4")

    bar = st.progress(0.0, text="Tracking objects...")
    try:
        result = analyze_video(tmp_in, out_path, opts["conf"], skip, max_frames,
                               max_width=max(640, opts["imgsz"]),
                               progress_cb=lambda p: bar.progress(p, text="Tracking objects..."),
                               model_name=opts["model_name"], imgsz=opts["imgsz"],
                               classes_text=opts["classes_text"])
    except Exception as e:
        st.error(f"Video analysis failed: {e}")
        st.stop()
    bar.empty()

    c1, c2, c3 = st.columns(3)
    c1.metric("Frames processed", result["frames_processed"])
    c2.metric("Time (s)", result["seconds"])
    c3.metric("Speed (FPS)", result["fps"])

    st.subheader("Unique objects tracked")
    if result["unique_objects"]:
        st.bar_chart(result["unique_objects"])
        db.log_detections("video", [{"label": k, "confidence": 100.0}
                                    for k, n in result["unique_objects"].items() for _ in range(n)])
    else:
        st.info("No objects were tracked.")

    if result["samples"]:
        st.subheader("Sample frames")
        cols = st.columns(3)
        for i, img in enumerate(result["samples"]):
            cols[i % 3].image(img, use_container_width=True)

    if result["output_exists"]:
        with open(out_path, "rb") as f:
            st.download_button("Download tracked video", f.read(),
                               file_name="tracked_video.mp4", mime="video/mp4")
        st.caption("If the video does not play in your browser, open it with VLC or Windows Media Player.")
