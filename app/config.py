import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.pkl"))
VECTORIZER_PATH = Path(os.getenv("VECTORIZER_PATH", MODELS_DIR / "vectorizer.pkl"))
DATA_DIR = ROOT / "data_test_v2"

SEED = 42
LABELS = ("positif", "negatif", "neutre")

# --- Limites de validation ---
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", 5000))
MAX_IMAGE_BYTES = int(float(os.getenv("MAX_IMAGE_MB", 5)) * 1024 * 1024)
MAX_AUDIO_BYTES = int(float(os.getenv("MAX_AUDIO_MB", 25)) * 1024 * 1024)

ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
ALLOWED_AUDIO_MIME = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "audio/mpeg", "audio/mp3", "audio/x-mpeg", "audio/mpeg3",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".mp3"}

# --- OCR / ASR ---
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "fra")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny | base | small | medium | large
WHISPER_LANGUAGE = "fr"

# Texte parasite ajouté par les captures du jeu de données (bannière avec le nom du fichier).
PARASITE_PATTERNS = [
    r"capture\s+avis\s+client",
    r"\b\w{2,5}\s*\.\s*(png|jpe?g|webp|bmp)\b",
]
