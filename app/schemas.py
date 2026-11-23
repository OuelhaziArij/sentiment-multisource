"""Schémas Pydantic (entrées / sorties de l'API)."""
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from . import config

Sentiment = Literal["positif", "negatif", "neutre"]


class TextRequest(BaseModel):
    """Corps JSON de POST /predict/text."""
    model_config = ConfigDict(json_schema_extra={"example": {"texte": "Livraison rapide, produit parfait !"}})

    texte: str = Field(..., validation_alias=AliasChoices("texte", "text"), max_length=config.MAX_TEXT_CHARS,
                       description=f"Avis à analyser (1 à {config.MAX_TEXT_CHARS} caractères, non vide).")

    @field_validator("texte")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le texte est vide ou ne contient que des espaces.")
        return v


class PredictionResponse(BaseModel):
    """Réponse JSON unifiée pour toutes les sources."""
    source: Literal["texte", "image", "audio"]
    texte_extrait: str
    sentiment_predit: Sentiment
    confiance: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(json_schema_extra={"example": {
        "source": "texte", "texte_extrait": "Livraison rapide, produit parfait !",
        "sentiment_predit": "positif", "confiance": 0.89}})


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    tesseract_available: bool
    whisper_installed: bool
    whisper_model: str
