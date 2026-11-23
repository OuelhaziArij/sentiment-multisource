import io
import struct
import sys
import types
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from app import asr  # noqa: E402

DATA = ROOT / "data_test_v2"


class FakeWhisperModel:
    """Remplace Whisper : renvoie un texte fixe (rapide et déterministe)."""
    text = "Excellent produit, je suis très satisfait et je recommande."
    calls = []

    def transcribe(self, path, **kwargs):
        FakeWhisperModel.calls.append((path, kwargs))
        return {"text": f" {FakeWhisperModel.text} "}


@pytest.fixture
def fake_whisper(monkeypatch):
    FakeWhisperModel.calls = []
    FakeWhisperModel.text = "Excellent produit, je suis très satisfait et je recommande."
    monkeypatch.setattr(asr, "get_model", lambda: FakeWhisperModel())
    return FakeWhisperModel


@pytest.fixture(scope="session")
def client():
    with TestClient(main.app) as c:
        yield c


def make_png(text="Livraison rapide, produit parfait !", size=(820, 200)) -> bytes:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((20, 80), text, fill="black")
    buf = io.BytesIO(); img.save(buf, "PNG")
    return buf.getvalue()


def make_wav(seconds=0.2, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(rate * seconds))
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return make_png()


@pytest.fixture
def wav_bytes():
    return make_wav()


@pytest.fixture
def dataset_image():
    return DATA / "images" / "I04.png"


@pytest.fixture
def dataset_mp3():
    return DATA / "audios" / "A01.mp3"
