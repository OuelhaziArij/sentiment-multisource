"""Mini-application web de test (Option B) : interroge l'API FastAPI en arrière-plan.

Lancer :  streamlit run streamlit_app.py      (API sur http://127.0.0.1:8000, modifiable via API_URL)
"""
import os

import httpx
import streamlit as st

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")
STYLE = {
    "positif": ("😀", "#1a9850", "Positif"),
    "negatif": ("😡", "#d73027", "Négatif"),
    "neutre": ("😐", "#7f7f7f", "Neutre"),
}

st.set_page_config(page_title="Sentiment multi-source", page_icon="💬", layout="centered")
st.title("💬 Classifieur de sentiment multi-source")
st.caption("E-commerce Customer Voice — texte, image ou audio ")

with st.sidebar:
    api_url = st.text_input("URL de l'API", DEFAULT_API)
    try:
        h = httpx.get(f"{api_url}/health", timeout=3).json()
        st.success("API en ligne")
        st.write({"modèle chargé": h["model_loaded"], "Tesseract": h["tesseract_available"],
                  "Whisper installé": h["whisper_installed"], "taille Whisper": h["whisper_model"]})
    except Exception:
        st.error("API injoignable. Lancez : uvicorn main:app --reload")
    st.markdown(f"[Swagger (/docs)]({api_url}/docs) · [ReDoc]({api_url}/redoc)")


def call_api(route, **kwargs):
    try:
        r = httpx.post(f"{api_url}{route}", timeout=180, **kwargs)
    except httpx.HTTPError as e:
        st.error(f"Impossible de joindre l'API : {e}")
        return
    if r.status_code != 200:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        if isinstance(detail, list):
            detail = " ; ".join(d.get("msg", str(d)) for d in detail)
        st.error(f"Erreur {r.status_code} : {detail}")
        return
    show(r.json())


def show(res):
    emoji, color, name = STYLE[res["sentiment_predit"]]
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{color}22;border-left:6px solid {color}'>"
        f"<span style='font-size:2rem'>{emoji}</span> <b style='font-size:1.5rem;color:{color}'>{name}</b></div>",
        unsafe_allow_html=True)
    st.write("")
    st.progress(min(max(res["confiance"], 0.0), 1.0), text=f"Confiance : {res['confiance']:.0%}")
    st.text_area("Texte extrait", res["texte_extrait"], height=110, disabled=True, key=f"txt-{id(res)}")
    with st.expander("Réponse JSON brute"):
        st.json(res)


tab_text, tab_img, tab_audio = st.tabs(["📝 Texte", "🖼️ Image", "🎙️ Audio"])

with tab_text:
    txt = st.text_area("Avis client", placeholder="Ex. : Livraison rapide, produit parfait !", height=120)
    if st.button("Analyser le texte", type="primary"):
        call_api("/predict/text", json={"texte": txt})

with tab_img:
    img = st.file_uploader("Capture d'écran (png, jpg, webp, bmp)", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if img:
        st.image(img, use_container_width=True)
        if st.button("Analyser l'image", type="primary"):
            call_api("/predict/image", files={"file": (img.name,
                     img.getvalue(), img.type or "application/octet-stream")})

with tab_audio:
    aud = st.file_uploader("Message vocal (wav, mp3)", type=["wav", "mp3"])
    if aud:
        st.audio(aud)
        if st.button("Analyser l'audio", type="primary"):
            with st.spinner("Transcription en cours…"):
                call_api("/predict/audio", files={"file": (aud.name,
                         aud.getvalue(), aud.type or "application/octet-stream")})
