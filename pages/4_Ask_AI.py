import cv2
import numpy as np
import streamlit as st
from PIL import Image

from modules import settings
from modules.detection import detect_objects
from modules.scene_understanding import describe_from_detections, smart_answer
from modules.ui import apply_style, hero
from modules.voice_assistant import speak_to_wav

st.set_page_config(page_title="Ask AI", page_icon="💬", layout="wide")
apply_style()
hero("Ask AI", "Ask questions about an image and listen to the answer.", "💬")

cfg = settings.load()
st.sidebar.write(f"**Provider:** {cfg['provider']}")
st.sidebar.caption("Order of attempts: cloud AI (retries automatically), then offline AI, "
                   "then object detection only. Change this on the Settings page.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, width=450)
    question = st.text_input("Your question", value="Describe what is happening in this image.")

    if st.button("Ask", type="primary"):
        with st.spinner("Thinking... this can take up to 30 seconds if the cloud AI is busy"):
            answer, source, notes = smart_answer(image, question, cfg)
        if answer is None:
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            _, detections = detect_objects(bgr, cfg["confidence"])
            answer = describe_from_detections(detections)
            source = "Object detection only (no AI answer)"
        st.session_state["ai_result"] = {"answer": answer, "source": source, "notes": notes}

    res = st.session_state.get("ai_result")
    if res:
        st.subheader("Answer")
        st.write(res["answer"])
        st.caption(f"Answered by: {res['source']}")
        for n in res["notes"]:
            st.warning(n)
        if st.button("🔊 Speak the answer"):
            audio, err = speak_to_wav(res["answer"])
            if err:
                st.error(err)
            else:
                st.audio(audio, format="audio/wav")
