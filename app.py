import streamlit as st

from modules import database as db
from modules.ui import apply_style, card, donut_chart, hero, timeline_chart

st.set_page_config(page_title="AI Vision Assistant", page_icon="👁️", layout="wide")
apply_style()

hero("AI Vision Assistant",
     "See images and video, detect and track objects, read text, ask questions and hear the answers.",
     "👁️")

stats = db.totals()
c1, c2, c3 = st.columns(3)
c1.metric("Total detections", stats["total"])
c2.metric("Different object types", stats["labels"])
c3.metric("Average confidence", f"{stats['avg_conf']}%")

st.write("")
counts = db.label_counts()
history = db.get_history()

left, right = st.columns([1, 1.3])
with left:
    st.subheader("Detected objects")
    if counts.empty:
        st.info("No data yet. Analyze an image to fill this chart.")
    else:
        st.altair_chart(donut_chart(counts), use_container_width=True)
with right:
    st.subheader("Activity over time")
    if history.empty:
        st.info("No activity yet.")
    else:
        st.altair_chart(timeline_chart(history), use_container_width=True)

st.subheader("Modules (open them from the sidebar)")
modules = [
    ("🖼️", "Image Analysis", "Upload a photo, detect objects and check image quality."),
    ("📷", "Live Camera", "Real-time detection from your webcam."),
    ("🎞️", "Video Analysis", "Track objects across a video with unique IDs."),
    ("💬", "Ask AI", "Ask questions about an image and hear the answer."),
    ("📄", "Text Scanner", "Read documents, labels and signboards."),
    ("✨", "Image Enhancement", "Fix blur, noise and poor lighting."),
    ("📊", "Detection History", "Charts and reports from the database."),
    ("⚙️", "Settings", "Model, confidence and AI provider."),
]
cols = st.columns(4)
for i, (icon, title, text) in enumerate(modules):
    cols[i % 4].markdown(card(icon, title, text), unsafe_allow_html=True)

