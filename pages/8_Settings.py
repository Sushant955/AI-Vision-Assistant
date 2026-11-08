import streamlit as st

from modules import detection, settings
from modules.scene_understanding import list_gemini_models, test_gemini
from modules.ui import apply_style, hero

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
apply_style()
hero("Settings", "Choose the detection model, default confidence and AI provider.", "⚙️")

cfg = settings.load()

st.subheader("Detection")
models = detection.available_models()
labels = list(models.keys())
current = detection.label_for(cfg["model"], models) or labels[0]
choice = st.selectbox("Default detection model (dataset)", labels, index=labels.index(current))
cfg["model"] = models[choice]
cfg["confidence"] = st.slider("Default confidence threshold", 0.1, 0.9, float(cfg["confidence"]), 0.05)
st.caption("Train a model on any dataset, put the .pt file in the project's 'models' folder, "
           "and restart the app. It will appear in the list.")

st.subheader("Ask AI provider")
providers = ["None", "Gemini", "Claude", "Local (offline)"]
cfg["provider"] = st.selectbox("Provider", providers,
                               index=providers.index(cfg["provider"]) if cfg["provider"] in providers else 0,
                               help="'Local (offline)' needs: pip install transformers")
cfg["api_key"] = st.text_input("API key (Gemini or Claude)", value=cfg["api_key"], type="password",
                               help="Stored only on this computer in settings.json")

fetched = st.session_state.get("gemini_models")
if fetched:
    cfg["gemini_model"] = st.selectbox(
        "Gemini model (the list your key can really use)", fetched,
        index=fetched.index(cfg["gemini_model"]) if cfg["gemini_model"] in fetched else 0)
else:
    cfg["gemini_model"] = st.text_input("Gemini model name", value=cfg["gemini_model"])
cfg["claude_model"] = st.text_input("Claude model name", value=cfg["claude_model"])

b1, b2 = st.columns(2)
if b1.button("Fetch my available Gemini models"):
    names, err = list_gemini_models(cfg["api_key"].strip())
    if err:
        st.error(err)
    else:
        st.session_state["gemini_models"] = names
        st.rerun()
if b2.button("Test Gemini connection"):
    with st.spinner("Testing..."):
        ok, msg = test_gemini(cfg["api_key"].strip(), cfg["gemini_model"])
    if ok:
        st.success("Working. Reply: " + msg)
    else:
        st.error("Not working. " + msg)

if st.button("Save settings", type="primary"):
    settings.save(cfg)
    st.success("Saved.")
