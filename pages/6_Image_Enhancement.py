import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from modules import image_quality as iq
from modules.ui import apply_style, hero

st.set_page_config(page_title="Image Enhancement", page_icon="✨", layout="wide")
apply_style()
hero("Image Enhancement", "Auto-fix or fine-tune brightness, contrast, sharpness, noise and size.", "✨")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image.thumbnail((1600, 1600))          # keeps denoising fast
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    before = iq.analyze(bgr)

    mode = st.sidebar.radio("Mode", ["Auto enhance", "Manual"])
    if mode == "Auto enhance":
        params = iq.auto_params(before)
        st.sidebar.caption("Settings chosen from your image: " +
                           ", ".join(f"{k}={v}" for k, v in params.items() if v not in (0, 1.0, False)))
    else:
        params = {
            "brightness": st.sidebar.slider("Brightness", -80, 80, 0),
            "contrast": st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.05),
            "saturation": st.sidebar.slider("Colour", 0.0, 2.0, 1.0, 0.05),
            "sharpness": st.sidebar.slider("Sharpness", 0.0, 3.0, 0.0, 0.1),
            "denoise": st.sidebar.slider("Noise reduction", 0, 20, 0),
            "clahe": st.sidebar.checkbox("Local contrast (CLAHE)", value=False),
            "upscale": 1,
        }
    params["upscale"] = 2 if st.sidebar.checkbox("Make image 2x bigger", value=False) else 1

    try:
        with st.spinner("Enhancing..."):
            result = iq.apply_adjustments(bgr, **params)
    except Exception as e:
        st.error(f"Enhancement failed: {e}")
        st.stop()
    after = iq.analyze(result)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sharpness", after["blur_score"], f"{after['blur_score'] - before['blur_score']:+.1f}")
    m2.metric("Brightness", after["brightness"], f"{after['brightness'] - before['brightness']:+.1f}")
    m3.metric("Noise", after["noise"], f"{after['noise'] - before['noise']:+.2f}", delta_color="inverse")
    m4.metric("Contrast", after["contrast"], f"{after['contrast'] - before['contrast']:+.1f}")
    for issue in before["issues"]:
        st.warning("Found: " + issue)

    c1, c2 = st.columns(2)
    c1.image(image, caption="Before", use_container_width=True)
    c2.image(result_rgb, caption="After", use_container_width=True)
    st.caption("Classic enhancement improves edges and lighting. It cannot rebuild detail "
               "that a badly blurred photo never captured.")

    buf = io.BytesIO()
    Image.fromarray(result_rgb).save(buf, format="PNG")
    st.download_button("Download enhanced image", buf.getvalue(),
                       file_name="enhanced.png", mime="image/png")
