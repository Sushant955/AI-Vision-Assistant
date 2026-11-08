"""Sidebar controls shared by Image Analysis, Live Camera and Video Analysis."""
import streamlit as st

from modules import detection, settings

PRECISION = {
    "Fast (640)": (640, False),
    "Balanced (960)": (960, False),
    "High precision (1280 + TTA)": (1280, True),
}


def sidebar_controls(live=False):
    cfg = settings.load()
    models = detection.available_models()
    labels = list(models.keys())
    default = detection.label_for(cfg["model"], models) or labels[0]

    st.sidebar.header("Detection settings")
    choice = st.sidebar.selectbox("Detection model (dataset)", labels,
                                  index=labels.index(default), key="mp_model")
    weights = models[choice]

    classes_text = ""
    if weights.startswith("world:"):
        classes_text = st.sidebar.text_area(
            "Objects to find (separate with commas)",
            value=", ".join(detection.DEFAULT_WORLD), key="mp_words",
            help="Type ANY object names, for example: helmet, fire extinguisher, red car")
        st.sidebar.caption("Needs Git installed and internet on the first run.")
    elif "Open Images" in choice:
        st.sidebar.caption("Knows about 600 object types (Open Images dataset).")
    elif "COCO" in choice:
        st.sidebar.caption("Knows 80 everyday object types (COCO dataset).")
    else:
        st.sidebar.caption("Your own trained model.")

    options = [k for k, v in PRECISION.items() if not (live and v[1])]
    prec = st.sidebar.selectbox("Accuracy vs speed", options, key="mp_prec")
    imgsz, augment = PRECISION[prec]
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, float(cfg["confidence"]), 0.05,
                             key="mp_conf")
    if "XLarge" in choice or imgsz >= 960:
        st.sidebar.warning("Big models and high precision are slow on a CPU laptop.")
    return {"conf": conf, "model_name": weights, "imgsz": imgsz,
            "augment": augment, "classes_text": classes_text}
