"""Module audio : transcription avec openai-whisper (chargement paresseux, .wav et .mp3 via ffmpeg)."""
import os
import threading

from . import config

_model = None
_model_name = None
_lock = threading.Lock()


class ASRUnavailableError(RuntimeError):
    """Whisper (ou ses poids / ffmpeg) indisponible dans l'environnement."""


def get_model():
    """Charge le modèle Whisper à la première utilisation (taille : variable WHISPER_MODEL)."""
    global _model, _model_name
    name = os.getenv("WHISPER_MODEL", config.WHISPER_MODEL)  # relu à chaque appel
    with _lock:
        if _model is None or _model_name != name:
            try:
                import whisper
            except ImportError as e:
                raise ASRUnavailableError("openai-whisper n'est pas installé (pip install openai-whisper).") from e
            try:
                _model = whisper.load_model(name)
            except Exception as e:  # poids non téléchargeables, réseau, mémoire...
                raise ASRUnavailableError(f"Impossible de charger le modèle Whisper '{name}' : {e}") from e
            _model_name = name
        return _model


def reset_model():
    global _model, _model_name
    with _lock:
        _model, _model_name = None, None


def transcribe(path: str) -> str:
    """Transcrit un fichier .wav/.mp3 en français."""
    model = get_model()
    result = model.transcribe(path, language=config.WHISPER_LANGUAGE, fp16=False)
    return (result.get("text") or "").strip()
