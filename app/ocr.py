"""Module image : prétraitement Pillow + pytesseract (lang=fra) + suppression du texte parasite."""
import io
import re
from typing import Union

from PIL import Image, ImageFilter, ImageOps

from . import config


class OCRUnavailableError(RuntimeError):
    pass


def preprocess_image(img: Image.Image, scale: int = 2) -> Image.Image:
    """Niveaux de gris -> contraste (autocontrast) -> agrandissement x2 -> légère netteté."""
    img = ImageOps.exif_transpose(img)
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.resize((gray.width * scale, gray.height * scale), Image.LANCZOS)
    return gray.filter(ImageFilter.SHARPEN)


def remove_parasite_text(text: str, patterns=None) -> str:
    """Supprime les lignes parasites (bannière « CAPTURE AVIS CLIENT — Ixx.png »)
    puis recolle les lignes restantes en un seul texte."""
    regs = [re.compile(p, re.IGNORECASE) for p in (patterns or config.PARASITE_PATTERNS)]
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if line and not any(r.search(line) for r in regs):
            kept.append(line)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def extract_text(image: Union[str, bytes, Image.Image], remove_parasite: bool = True) -> str:
    try:
        import pytesseract
    except ImportError as e:  # pragma: no cover
        raise OCRUnavailableError("pytesseract n'est pas installé.") from e
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, str):
        image = Image.open(image)
    prepared = preprocess_image(image)
    try:
        raw = pytesseract.image_to_string(prepared, lang=config.TESSERACT_LANG, config="--psm 6")
    except pytesseract.TesseractNotFoundError as e:
        raise OCRUnavailableError("Le binaire Tesseract est introuvable (apt install tesseract-ocr tesseract-ocr-fra).") from e
    return remove_parasite_text(raw) if remove_parasite else raw.strip()
