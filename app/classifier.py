"""Classifieur de sentiment textuel : chargement des artefacts (.pkl) et prédiction."""
import threading
from typing import Tuple

import joblib
import numpy as np

from . import config
from .preprocessing import clean_text

_lock = threading.Lock()
_cache: dict = {}


class ModelNotFoundError(RuntimeError):
    pass


def load_artifacts(force: bool = False):
    """Charge (une seule fois) vectorizer.pkl et model.pkl."""
    with _lock:
        if force or "model" not in _cache:
            if not config.MODEL_PATH.exists() or not config.VECTORIZER_PATH.exists():
                raise ModelNotFoundError(
                    "Artefacts introuvables : lancez `python scripts/train_model.py` pour générer "
                    f"{config.MODEL_PATH.name} et {config.VECTORIZER_PATH.name}."
                )
            _cache["vectorizer"] = joblib.load(config.VECTORIZER_PATH)
            _cache["model"] = joblib.load(config.MODEL_PATH)
        return _cache["vectorizer"], _cache["model"]


def is_ready() -> bool:
    try:
        load_artifacts()
        return True
    except ModelNotFoundError:
        return False


def predict_sentiment(text: str) -> Tuple[str, float]:
    """Retourne (label, confiance) où confiance = max(predict_proba)."""
    vectorizer, model = load_artifacts()
    X = vectorizer.transform([clean_text(text)])
    proba = model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    return str(model.classes_[idx]), round(float(proba[idx]), 4)
