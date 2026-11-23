"""Validation des fichiers entrants : MIME (repli sur l'extension), taille, vide, corrompu."""
import io
import os
from typing import Tuple

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from . import config

CHUNK = 64 * 1024


def read_limited(upload: UploadFile, max_bytes: int, kind: str) -> bytes:
    """Lit le fichier par blocs et s'arrête dès que la limite est dépassée (413)."""
    data = bytearray()
    while True:
        chunk = upload.file.read(CHUNK)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(413, f"Fichier {kind} trop volumineux : limite {max_bytes / 1024 / 1024:.0f} Mo.")
    if not data:
        raise HTTPException(422, f"Fichier {kind} vide.")
    return bytes(data)


def _ext(filename: str | None) -> str:
    return os.path.splitext(filename or "")[1].lower()


def sniff_kind(data: bytes) -> str | None:
    """Détecte le type réel par signature binaire (« magic bytes »)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff") or data[:2] == b"BM" \
            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
        return "image"
    if (data[:4] == b"RIFF" and data[8:12] == b"WAVE") or data.startswith(b"ID3") \
            or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio"
    return None


def check_declared_type(upload: UploadFile, kind: str) -> None:
    """Contrôle le type MIME déclaré ; s'il est absent/générique, repli sur l'extension."""
    allowed_mime = config.ALLOWED_IMAGE_MIME if kind == "image" else config.ALLOWED_AUDIO_MIME
    allowed_ext = config.IMAGE_EXT if kind == "image" else config.AUDIO_EXT
    mime = (upload.content_type or "").split(";")[0].strip().lower()
    if mime and mime != "application/octet-stream":
        if mime not in allowed_mime:
            raise HTTPException(422, f"Type MIME '{mime}' non accepté pour une {kind}. Acceptés : {', '.join(sorted(allowed_mime))}.")
    elif _ext(upload.filename) not in allowed_ext:
        raise HTTPException(422, f"Type de fichier non reconnu (MIME absent et extension '{_ext(upload.filename)}' non acceptée). "
                                 f"Extensions acceptées : {', '.join(sorted(allowed_ext))}.")


def load_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise HTTPException(422, "Image corrompue ou illisible.") from e


def check_audio_bytes(data: bytes) -> str:
    """Vérifie la signature audio ; retourne l'extension à utiliser (.wav / .mp3)."""
    if sniff_kind(data) != "audio":
        raise HTTPException(422, "Fichier audio corrompu ou format non reconnu (attendu : WAV ou MP3).")
    return ".wav" if data[:4] == b"RIFF" else ".mp3"


def detect_upload_kind(upload: UploadFile, data: bytes) -> Tuple[str, str]:
    """Pour /predict : type via MIME, sinon extension, sinon signature binaire."""
    mime = (upload.content_type or "").split(";")[0].strip().lower()
    if mime in config.ALLOWED_IMAGE_MIME:
        return "image", "mime"
    if mime in config.ALLOWED_AUDIO_MIME:
        return "audio", "mime"
    ext = _ext(upload.filename)
    if ext in config.IMAGE_EXT:
        return "image", "extension"
    if ext in config.AUDIO_EXT:
        return "audio", "extension"
    kind = sniff_kind(data)
    if kind:
        return kind, "signature"
    raise HTTPException(422, f"Impossible de déterminer le type du fichier (MIME '{mime or 'absent'}', extension '{ext or 'absente'}'). "
                             "Envoyez une image (png/jpg/webp/bmp) ou un audio (wav/mp3), ou un champ 'texte'.")
