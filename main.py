
import json
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from app import asr, classifier, config, service, validation
from app.schemas import ErrorResponse, HealthResponse, PredictionResponse, TextRequest

ERRORS = {
    413: {"model": ErrorResponse, "description": "Fichier trop volumineux"},
    422: {"model": ErrorResponse, "description": "Charge utile invalide (vide, corrompue, type non accepté...)"},
    503: {"model": ErrorResponse, "description": "Module OCR/ASR indisponible"},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        classifier.load_artifacts()  # échoue tôt si les .pkl manquent ; Whisper reste chargé à la demande
    except classifier.ModelNotFoundError as e:
        print(f"[WARN] {e}")
    yield


app = FastAPI(
    title="Classifieur de sentiment multi-source",
    description="Cascade **OCR / ASR -> texte -> classifieur unique** (TF-IDF + Régression Logistique). "
                "Réponse unifiée : `source`, `texte_extrait`, `sentiment_predit`, `confiance`.",
    version="1.0.0", lifespan=lifespan,
)


@app.exception_handler(classifier.ModelNotFoundError)
async def model_missing(_, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse, tags=["Système"])
def health():
    import importlib.util
    return HealthResponse(
        status="ok", model_loaded=classifier.is_ready(), tesseract_available=shutil.which("tesseract") is not None,
        whisper_installed=importlib.util.find_spec("whisper") is not None, whisper_model=config.WHISPER_MODEL)


@app.post("/predict/text", response_model=PredictionResponse, responses=ERRORS, tags=["Prédiction"])
def predict_text(body: TextRequest):
    """Sentiment d'un avis saisi en texte (JSON `{"texte": "..."}`)."""
    return service.predict_from_text(body.texte)


def _handle_image(upload: UploadFile) -> PredictionResponse:
    validation.check_declared_type(upload, "image")
    data = validation.read_limited(upload, config.MAX_IMAGE_BYTES, "image")
    return service.predict_from_image(validation.load_image(data))


def _handle_audio(upload: UploadFile) -> PredictionResponse:
    validation.check_declared_type(upload, "audio")
    data = validation.read_limited(upload, config.MAX_AUDIO_BYTES, "audio")
    suffix = validation.check_audio_bytes(data)
    return service.predict_from_audio(data, suffix)


@app.post("/predict/image", response_model=PredictionResponse, responses=ERRORS, tags=["Prédiction"])
def predict_image(file: UploadFile = File(..., description="Capture d'écran (png, jpg, webp, bmp) - max 5 Mo")):
    """Image -> OCR (pytesseract, fra) -> classifieur."""
    return _handle_image(file)


@app.post("/predict/audio", response_model=PredictionResponse, responses=ERRORS, tags=["Prédiction"])
def predict_audio(file: UploadFile = File(..., description="Message vocal (wav, mp3) - max 25 Mo")):
    """Audio -> Whisper (fr) -> classifieur."""
    return _handle_audio(file)


_AUTO_DOC = {"requestBody": {"required": True, "content": {
    "multipart/form-data": {"schema": {"type": "object", "properties": {
        "file": {"type": "string", "format": "binary", "description": "Image ou audio (type détecté automatiquement)"},
        "texte": {"type": "string", "description": "Alternative à `file` : avis en texte"}}}},
    "application/json": {"schema": {"$ref": "#/components/schemas/TextRequest"}}}}}


@app.post("/predict", response_model=PredictionResponse, responses=ERRORS, tags=["Prédiction"], openapi_extra=_AUTO_DOC)
async def predict_auto(request: Request):
    """Détection automatique : JSON `{"texte"}`, formulaire `texte`, ou fichier `file` (image/audio via MIME > extension > signature)."""
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        try:
            body = TextRequest.model_validate(await request.json())
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise HTTPException(422, f"Corps JSON invalide : {_short(e)}")
        return service.predict_from_text(body.texte)
    if ctype.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        form = await request.form()
        upload = form.get("file")
        texte: Optional[str] = form.get("texte")  # type: ignore[assignment]
        if isinstance(upload, StarletteUploadFile):
            from starlette.concurrency import run_in_threadpool
            return await run_in_threadpool(_auto_file, upload)
        if isinstance(texte, str):
            try:
                body = TextRequest(texte=texte)
            except ValidationError as e:
                raise HTTPException(422, f"Champ 'texte' invalide : {_short(e)}")
            return service.predict_from_text(body.texte)
        raise HTTPException(422, "Fournissez un fichier 'file' (image/audio) ou un champ 'texte'.")
    raise HTTPException(422, "Content-Type non pris en charge : utilisez application/json, multipart/form-data ou application/x-www-form-urlencoded.")


def _auto_file(upload: UploadFile) -> PredictionResponse:
    # Lecture bornée par la plus grande limite, puis contrôle selon le type détecté
    data = validation.read_limited(upload, max(config.MAX_IMAGE_BYTES, config.MAX_AUDIO_BYTES), "envoyé")
    kind, _ = validation.detect_upload_kind(upload, data)
    if kind == "image":
        if len(data) > config.MAX_IMAGE_BYTES:
            raise HTTPException(413, f"Fichier image trop volumineux : limite {config.MAX_IMAGE_BYTES / 1024 / 1024:.0f} Mo.")
        return service.predict_from_image(validation.load_image(data))
    if len(data) > config.MAX_AUDIO_BYTES:
        raise HTTPException(413, f"Fichier audio trop volumineux : limite {config.MAX_AUDIO_BYTES / 1024 / 1024:.0f} Mo.")
    return service.predict_from_audio(data, validation.check_audio_bytes(data))


def _short(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
    return str(e)
