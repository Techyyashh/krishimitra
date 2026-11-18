"""KrishiMitra – Streamlit demo (quickest way to get a shareable link).

Run:  streamlit run streamlit_app/app.py
It uses the same modules as the API, in-process (no separate backend needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.inference.image_utils import InvalidImageError, load_image_bytes  # noqa: E402
from src.inference.leaf_validator import is_probably_leaf  # noqa: E402
from src.inference.predictor import load_predictor  # noqa: E402
from src.llm.advisor import Advisor  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.llm.knowledge import KnowledgeBase  # noqa: E402
from src.speech.tts import TTSError, synthesize  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

st.set_page_config(page_title="KrishiMitra", page_icon="🌾", layout="centered")
cfg = load_config()


@st.cache_resource
def get_services():
    icfg = cfg["inference"]
    kb = KnowledgeBase(resolve_path(cfg["knowledge_base"]["path"]))
    predictor = load_predictor(resolve_path(icfg["onnx_path"]), resolve_path(icfg["classes_path"]),
                               icfg["top_k"], icfg["confidence_threshold"])
    llm = LLMClient(cfg["llm"]["model"], cfg["llm"]["max_tokens"], cfg["llm"]["effort"])
    return kb, predictor, Advisor(kb, llm)


kb, predictor, advisor = get_services()

st.title("🌾 KrishiMitra")
st.caption("Click a leaf, save your crop. · पत्ती की फ़ोटो लें, फसल बचाएँ।")
lang = st.radio("Language / भाषा", ["en", "hi"], format_func=lambda x: "English" if x == "en" else "हिंदी", horizontal=True)

if predictor is None:
    st.error("Model not found. Train and export it first (README → Train the model), "
             "or set HF_MODEL_REPO to download it.")
    st.stop()

source = st.radio("Photo", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")
photo = st.camera_input("Take a photo of one leaf") if source == "Camera" else st.file_uploader(
    "Upload a leaf photo", type=["jpg", "jpeg", "png", "webp"])

if photo is not None:
    try:
        image = load_image_bytes(photo.getvalue())
    except InvalidImageError:
        st.error("This file is not a valid image.")
        st.stop()
    st.image(image, use_container_width=True)

    ok, score = is_probably_leaf(image, cfg["inference"]["leaf_min_ratio"])
    if not ok:
        st.warning(f"This does not look like a leaf (score {score:.2f}). Take a close photo of ONE leaf in daylight.")
        st.stop()

    result = predictor.predict(image)
    best = result["best"]
    info = kb.get(best["label"])
    name = info["name_hi"] if lang == "hi" else info["name_en"]
    crop = info["crop_hi"] if lang == "hi" else info["crop_en"]

    if not result["confident"]:
        st.warning("Not sure – please retake the photo (one leaf, daylight, not blurry).")
    st.subheader(f"{crop}: {name}")
    st.progress(best["confidence"], text=f"{best['confidence']:.0%} sure · {result['inference_ms']} ms")
    with st.expander("Other possibilities"):
        for item in result["top"][1:]:
            st.write(f"- {item['crop']} – {item['disease']} ({item['confidence']:.0%})")

    if st.button("Get advice / सलाह पाएँ", type="primary"):
        with st.spinner("Preparing advice…"):
            advice = advisor.advise(best["label"], lang, best["confidence"])
        st.session_state["advice"] = advice["text"]
        st.caption(f"Source: {advice['source']}")

    if st.session_state.get("advice"):
        st.markdown(st.session_state["advice"].replace("\n", "  \n"))
        if st.button("🔊 Listen / सुनें"):
            try:
                st.audio(synthesize(st.session_state["advice"], lang), format="audio/mp3")
            except TTSError as err:
                st.error(str(err))

    st.info(kb.disclaimer(lang))
