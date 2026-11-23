import io
import shutil
import sys
import types

import pytest
from PIL import Image

import main
from app import asr, classifier, config, ocr
from app.preprocessing import STOPWORDS, clean_text, normalize_for_similarity

LABELS = set(config.LABELS)
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract absent")


def check_prediction(body, source):
    assert set(body) == {"source", "texte_extrait", "sentiment_predit", "confiance"}
    assert body["source"] == source
    assert isinstance(body["texte_extrait"], str) and body["texte_extrait"]
    assert body["sentiment_predit"] in LABELS
    assert 0.0 <= body["confiance"] <= 1.0


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True



def test_predict_text_ok(client):
    r = client.post("/predict/text", json={"texte": "Livraison rapide, produit parfait !"})
    assert r.status_code == 200
    check_prediction(r.json(), "texte")
    assert r.json()["sentiment_predit"] == "positif"


def test_predict_text_alias_text(client):
    assert client.post("/predict/text", json={"text": "Colis reçu cassé, très déçu"}).status_code == 200


@pytest.mark.parametrize("payload", [{"texte": ""}, {"texte": "   \n\t "}, {}, {"texte": 123}, {"texte": None},
                                     {"texte": "x" * (config.MAX_TEXT_CHARS + 1)}])
def test_predict_text_invalid_422(client, payload):
    r = client.post("/predict/text", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_predict_text_not_json_422(client):
    assert client.post("/predict/text", content=b"pas du json", headers={"content-type": "application/json"}).status_code == 422


# ---------------------------------------------------------------- /predict/image
@needs_tesseract
def test_predict_image_dataset_ok(client, dataset_image):
    with open(dataset_image, "rb") as f:
        r = client.post("/predict/image", files={"file": ("I04.png", f, "image/png")})
    assert r.status_code == 200
    body = r.json()
    check_prediction(body, "image")
    assert "capture avis client" not in body["texte_extrait"].lower()  # texte parasite retiré
    assert "qualité" in body["texte_extrait"].lower()


def test_predict_image_mocked_ocr(client, png_bytes, monkeypatch):
    monkeypatch.setattr(ocr, "extract_text", lambda img, **k: "Service client injoignable, inadmissible !")
    r = client.post("/predict/image", files={"file": ("a.png", png_bytes, "image/png")})
    assert r.status_code == 200
    check_prediction(r.json(), "image")


def test_predict_image_extension_fallback(client, png_bytes, monkeypatch):
    monkeypatch.setattr(ocr, "extract_text", lambda img, **k: "très bien")
    r = client.post("/predict/image", files={"file": ("a.png", png_bytes, "application/octet-stream")})
    assert r.status_code == 200


def test_predict_image_wrong_mime_422(client):
    r = client.post("/predict/image", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 422 and "MIME" in r.json()["detail"]


def test_predict_image_unknown_ext_without_mime_422(client):
    r = client.post("/predict/image", files={"file": ("a.xyz", b"hello", "application/octet-stream")})
    assert r.status_code == 422


def test_predict_image_empty_422(client):
    r = client.post("/predict/image", files={"file": ("a.png", b"", "image/png")})
    assert r.status_code == 422 and "vide" in r.json()["detail"]


def test_predict_image_corrupted_422(client):
    r = client.post("/predict/image", files={"file": ("a.png", b"\x89PNG\r\n\x1a\ngarbage", "image/png")})
    assert r.status_code == 422 and "corrompue" in r.json()["detail"]


def test_predict_image_too_large_413(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", 1024)
    r = client.post("/predict/image", files={"file": ("a.png", b"\x89PNG" + b"0" * 5000, "image/png")})
    assert r.status_code == 413


def test_predict_image_no_text_422(client, monkeypatch):
    monkeypatch.setattr(ocr, "extract_text", lambda img, **k: "")
    buf = io.BytesIO(); Image.new("RGB", (50, 50), "white").save(buf, "PNG")
    r = client.post("/predict/image", files={"file": ("blank.png", buf.getvalue(), "image/png")})
    assert r.status_code == 422 and "Aucun texte" in r.json()["detail"]


def test_predict_image_missing_file_422(client):
    assert client.post("/predict/image").status_code == 422


# ---------------------------------------------------------------- /predict/audio (Whisper simulé)
def test_predict_audio_wav_ok(client, wav_bytes, fake_whisper):
    r = client.post("/predict/audio", files={"file": ("v.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 200
    body = r.json(); check_prediction(body, "audio")
    assert body["texte_extrait"] == fake_whisper.text and body["sentiment_predit"] == "positif"
    assert fake_whisper.calls[0][1]["language"] == "fr"
    assert fake_whisper.calls[0][0].endswith(".wav")


def test_predict_audio_mp3_dataset_ok(client, dataset_mp3, fake_whisper):
    with open(dataset_mp3, "rb") as f:
        r = client.post("/predict/audio", files={"file": ("A01.mp3", f, "audio/mpeg")})
    assert r.status_code == 200
    check_prediction(r.json(), "audio")
    assert fake_whisper.calls[0][0].endswith(".mp3")


def test_predict_audio_negative_text(client, wav_bytes, fake_whisper):
    fake_whisper.text = "Le produit ne fonctionne pas du tout, je demande le remboursement."
    r = client.post("/predict/audio", files={"file": ("v.wav", wav_bytes, "audio/x-wav")})
    assert r.json()["sentiment_predit"] == "negatif"


def test_predict_audio_wrong_mime_422(client, fake_whisper):
    assert client.post("/predict/audio", files={"file": ("a.png", b"\x89PNG", "image/png")}).status_code == 422


def test_predict_audio_empty_422(client, fake_whisper):
    assert client.post("/predict/audio", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 422


def test_predict_audio_corrupted_422(client, fake_whisper):
    r = client.post("/predict/audio", files={"file": ("a.wav", b"ceci n'est pas de l'audio", "audio/wav")})
    assert r.status_code == 422 and "corrompu" in r.json()["detail"]


def test_predict_audio_too_large_413(client, wav_bytes, fake_whisper, monkeypatch):
    monkeypatch.setattr(config, "MAX_AUDIO_BYTES", 1000)
    r = client.post("/predict/audio", files={"file": ("a.wav", wav_bytes + b"0" * 5000, "audio/wav")})
    assert r.status_code == 413


def test_predict_audio_silence_422(client, wav_bytes, fake_whisper):
    fake_whisper.text = ""
    r = client.post("/predict/audio", files={"file": ("a.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 422 and "parole" in r.json()["detail"]


def test_predict_audio_whisper_unavailable_503(client, wav_bytes, monkeypatch):
    def boom():
        raise asr.ASRUnavailableError("poids absents")
    monkeypatch.setattr(asr, "get_model", boom)
    r = client.post("/predict/audio", files={"file": ("a.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 503 and "poids absents" in r.json()["detail"]


# ---------------------------------------------------------------- /predict (détection automatique)
def test_auto_json_text(client):
    r = client.post("/predict", json={"texte": "Article cassé, très déçu"})
    assert r.status_code == 200; check_prediction(r.json(), "texte")


def test_auto_form_text(client):
    r = client.post("/predict", data={"texte": "Article cassé, très déçu"})
    assert r.status_code == 200; check_prediction(r.json(), "texte")


def test_auto_image_by_mime(client, png_bytes, monkeypatch):
    monkeypatch.setattr(ocr, "extract_text", lambda img, **k: "parfait, merci")
    r = client.post("/predict", files={"file": ("x", png_bytes, "image/png")})
    assert r.status_code == 200 and r.json()["source"] == "image"


def test_auto_audio_by_extension(client, wav_bytes, fake_whisper):
    r = client.post("/predict", files={"file": ("msg.wav", wav_bytes, "application/octet-stream")})
    assert r.status_code == 200 and r.json()["source"] == "audio"


def test_auto_type_by_signature(client, wav_bytes, fake_whisper):
    r = client.post("/predict", files={"file": ("sans_extension", wav_bytes, "application/octet-stream")})
    assert r.status_code == 200 and r.json()["source"] == "audio"


def test_auto_unknown_422(client):
    r = client.post("/predict", files={"file": ("a.pdf", b"%PDF-1.4 ...", "application/octet-stream")})
    assert r.status_code == 422


def test_auto_nothing_422(client):
    assert client.post("/predict", data={"autre": "x"}, files={"zzz": ("a", b"x")}).status_code == 422


def test_auto_blank_text_422(client):
    assert client.post("/predict", json={"texte": "  "}).status_code == 422
    assert client.post("/predict", data={"texte": "  "}).status_code == 422


def test_auto_too_large_413(client, monkeypatch, png_bytes):
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", 100)
    r = client.post("/predict", files={"file": ("a.png", png_bytes, "image/png")})
    assert r.status_code == 413


def test_docs_and_openapi(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/predict/text", "/predict/image", "/predict/audio", "/health"} <= set(paths)


# ---------------------------------------------------------------- unitaires : prétraitement
def test_clean_text_basics():
    assert clean_text("Très DÉÇU !!!") == "tres decu"
    assert clean_text(None) == ""
    assert clean_text("Commande #7783 : 24h") == "commande h"  # chiffres retirés


def test_clean_text_keeps_negations():
    out = clean_text("Je n'ai jamais rien reçu, ce n'est pas bien et sans emballage")
    for w in ("n", "jamais", "rien", "pas", "sans"):
        assert w in out.split()


def test_stopwords_removed_but_not_negations():
    assert {"le", "la", "de", "et"} <= STOPWORDS
    assert not ({"ne", "pas", "sans", "jamais", "rien", "plus", "non", "aucun"} & STOPWORDS)


def test_normalize_for_similarity_keeps_stopwords():
    assert normalize_for_similarity("Le colis, ok!") == "le colis ok"


# ---------------------------------------------------------------- unitaires : classifieur
@pytest.mark.parametrize("text,expected", [
    ("Excellente qualité, très satisfait, je recommande vivement !", "positif"),
    ("Article cassé, service client injoignable, très déçu.", "negatif"),
    ("Colis conforme à la description, livraison standard.", "neutre"),
])
def test_classifier_known_examples(text, expected):
    label, conf = classifier.predict_sentiment(text)
    assert label == expected and 0.0 < conf <= 1.0


def test_classifier_probabilities_sum_to_one():
    vec, model = classifier.load_artifacts()
    proba = model.predict_proba(vec.transform([clean_text("bon produit")]))[0]
    assert abs(proba.sum() - 1.0) < 1e-9 and set(model.classes_) == LABELS


def test_classifier_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "nope.pkl")
    with pytest.raises(classifier.ModelNotFoundError):
        classifier.load_artifacts(force=True)
    monkeypatch.undo()
    classifier.load_artifacts(force=True)


# ---------------------------------------------------------------- unitaires : OCR
def test_preprocess_image_grayscale_and_upscaled():
    img = Image.new("RGB", (100, 40), "white")
    out = ocr.preprocess_image(img)
    assert out.mode == "L" and out.size == (200, 80)


def test_remove_parasite_text():
    raw = "CAPTURE AVIS CLIENT — 102.png\nRÉCLAMATION COMMANDE #9921\nColis ouvert.\n"
    assert ocr.remove_parasite_text(raw) == "RÉCLAMATION COMMANDE #9921 Colis ouvert."
    assert ocr.remove_parasite_text("capture avis client — I05.png\nOK") == "OK"


@needs_tesseract
def test_ocr_real_image_matches_reference():
    import pandas as pd
    ref = pd.read_csv(config.DATA_DIR / "images.csv").set_index("fichier").texte_attendu_ocr
    for name in ("I02.png", "I06.png"):
        assert ocr.extract_text(str(config.DATA_DIR / "images" / name)) == ref[name]


# ---------------------------------------------------------------- unitaires : ASR (chargement paresseux)
def test_whisper_lazy_loading_and_env_size(monkeypatch):
    loaded = []
    fake = types.ModuleType("whisper")
    fake.load_model = lambda name: loaded.append(name) or object()
    monkeypatch.setitem(sys.modules, "whisper", fake)
    asr.reset_model()
    assert loaded == []                       # rien chargé à l'import
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    asr.get_model(); asr.get_model()
    assert loaded == ["tiny"]                 # chargé une seule fois, taille lue dans l'environnement
    asr.reset_model()


def test_whisper_missing_raises_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper", None)  # force ImportError
    asr.reset_model()
    with pytest.raises(asr.ASRUnavailableError):
        asr.get_model()
