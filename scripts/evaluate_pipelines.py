"""Étape 2 : évaluation des modules OCR et ASR contre images.csv / audios.csv.

Ne simule JAMAIS un résultat : si Whisper (poids) est indisponible, l'évaluation audio est
marquée « non exécutée » et seul le décodage ffmpeg + la classification des transcriptions
de référence sont rapportés.
"""
import difflib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app import asr, classifier, config, ocr
from app.preprocessing import normalize_for_similarity

REPORTS = config.ROOT / "reports"


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def metrics(ref: str, hyp: str):
    r, h = normalize_for_similarity(ref), normalize_for_similarity(hyp)
    cer = levenshtein(r, h) / max(len(r), 1)
    wer = levenshtein_words(r.split(), h.split()) / max(len(r.split()), 1)
    return round(difflib.SequenceMatcher(None, r, h).ratio(), 3), round(cer, 3), round(wer, 3)


def levenshtein_words(a, b):
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def eval_images():
    df = pd.read_csv(config.DATA_DIR / "images.csv")
    rows = []
    for _, r in df.iterrows():
        raw = ocr.extract_text(str(config.DATA_DIR / "images" / r.fichier), remove_parasite=False)
        txt = ocr.remove_parasite_text(raw)
        sim, cer, wer = metrics(r.texte_attendu_ocr, txt)
        label, conf = classifier.predict_sentiment(txt)
        rows.append({"fichier": r.fichier, "texte_ocr_brut": raw, "texte_ocr_nettoye": txt, "similarite": sim,
                     "CER": cer, "WER": wer, "sentiment_ref": r.sentiment_ref, "sentiment_predit": label,
                     "confiance": conf, "correct": label == r.sentiment_ref})
    return pd.DataFrame(rows)


def eval_audio():
    df = pd.read_csv(config.DATA_DIR / "audios.csv")
    out = {"fichiers": [], "whisper": None}
    for _, r in df.iterrows():
        path = config.DATA_DIR / "audios" / r.fichier
        probe = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "wav", "-"],
                               capture_output=True)
        label_ref, conf_ref = classifier.predict_sentiment(r.transcription_reference)
        out["fichiers"].append({"fichier": r.fichier, "decodage_ffmpeg_ok": probe.returncode == 0 and len(probe.stdout) > 1000,
                                "sentiment_ref": r.sentiment_ref, "predit_sur_transcription_reference": label_ref,
                                "confiance": conf_ref})
    try:
        asr.get_model()
    except asr.ASRUnavailableError as e:
        out["whisper"] = {"execute": False, "raison": str(e)}
        return out, None
    rows = []
    for _, r in df.iterrows():
        txt = asr.transcribe(str(config.DATA_DIR / "audios" / r.fichier))
        sim, cer, wer = metrics(r.transcription_reference, txt)
        label, conf = classifier.predict_sentiment(txt)
        rows.append({"fichier": r.fichier, "transcription": txt, "similarite": sim, "CER": cer, "WER": wer,
                     "sentiment_ref": r.sentiment_ref, "sentiment_predit": label, "confiance": conf, "correct": label == r.sentiment_ref})
    out["whisper"] = {"execute": True, "modele": config.WHISPER_MODEL}
    return out, pd.DataFrame(rows)


def main():
    REPORTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    img = eval_images()
    img.to_csv(REPORTS / "eval_ocr.csv", index=False)
    print("== OCR ==")
    print(img[["fichier", "texte_ocr_nettoye", "similarite", "CER", "sentiment_ref", "sentiment_predit", "confiance"]].to_string(index=False))
    print(f"\nSimilarité moyenne {img.similarite.mean():.3f} | CER moyen {img.CER.mean():.3f} | WER moyen {img.WER.mean():.3f} | "
          f"accuracy sentiment (OCR réel) {img.correct.mean():.3f} ({int(img.correct.sum())}/{len(img)})")

    aud, aud_df = eval_audio()
    print("\n== AUDIO ==")
    print(json.dumps(aud["whisper"], ensure_ascii=False))
    ok = sum(f["decodage_ffmpeg_ok"] for f in aud["fichiers"])
    ref_ok = sum(f["sentiment_ref"] == f["predit_sur_transcription_reference"] for f in aud["fichiers"])
    print(f"Décodage ffmpeg mp3 -> wav 16 kHz mono : {ok}/{len(aud['fichiers'])} fichiers OK")
    print(f"Classifieur sur transcriptions de RÉFÉRENCE (pas de l'ASR) : {ref_ok}/{len(aud['fichiers'])} corrects")
    if aud_df is not None:
        aud_df.to_csv(REPORTS / "eval_asr.csv", index=False)
        print(aud_df.to_string(index=False))
        print(f"Similarité {aud_df.similarite.mean():.3f} | WER {aud_df.WER.mean():.3f} | accuracy {aud_df.correct.mean():.3f}")
    (REPORTS / "eval_audio.json").write_text(json.dumps(aud, ensure_ascii=False, indent=2))
    summary = {"ocr": {"similarite_moy": float(img.similarite.mean()), "CER_moy": float(img.CER.mean()),
                       "WER_moy": float(img.WER.mean()), "accuracy": float(img.correct.mean())},
               "audio": {"whisper": aud["whisper"], "ffmpeg_ok": ok, "classif_sur_reference": ref_ok,
                         "n": len(aud["fichiers"])}}
    (REPORTS / "eval_pipelines_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
