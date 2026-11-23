"""Couche métier : cascade source -> texte -> classifieur (réponse unifiée)."""
import os
import subprocess
import tempfile

from fastapi import HTTPException
from PIL import Image

from . import asr, classifier, ocr
from .schemas import PredictionResponse


def _respond(source: str, text: str) -> PredictionResponse:
    label, conf = classifier.predict_sentiment(text)
    return PredictionResponse(source=source, texte_extrait=text, sentiment_predit=label, confiance=conf)


def predict_from_text(text: str) -> PredictionResponse:
    return _respond("texte", text)


def predict_from_image(img: Image.Image) -> PredictionResponse:
    try:
        text = ocr.extract_text(img)
    except ocr.OCRUnavailableError as e:
        raise HTTPException(503, f"Service OCR indisponible : {e}")
    except Exception as e:
        raise HTTPException(422, f"Image non exploitable par l'OCR : {e}")
    if not text:
        raise HTTPException(422, "Aucun texte détecté dans l'image.")
    return _respond("image", text)


def predict_from_audio(data: bytes, suffix: str) -> PredictionResponse:
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            text = asr.transcribe(path)
        except asr.ASRUnavailableError as e:
            raise HTTPException(503, f"Service de transcription indisponible : {e}")
        except (RuntimeError, subprocess.SubprocessError, OSError) as e:
            raise HTTPException(422, f"Audio illisible (décodage ffmpeg impossible) : {e}")
    finally:
        if os.path.exists(path):
            os.remove(path)
    if not text:
        raise HTTPException(422, "Aucune parole détectée dans l'audio.")
    return _respond("audio", text)
