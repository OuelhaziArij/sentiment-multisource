
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from app import config
from app.preprocessing import clean_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = config.SEED
LABELS = list(config.LABELS)
REPORTS = config.ROOT / "reports"
N_SPLITS, N_REPEATS = 4, 25


def word(ngram=(1, 1)):
    return TfidfVectorizer(ngram_range=ngram, sublinear_tf=True)


def char(ngram=(2, 5)):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, sublinear_tf=True)


def both():
    return FeatureUnion([("w", word((1, 2))), ("c", char((2, 5)))])


def lr(C):
    return LogisticRegression(C=C, max_iter=2000, random_state=SEED)


CANDIDATES = {
    "Baseline (classe majoritaire)": (word, lambda: DummyClassifier(strategy="most_frequent")),
    "TF-IDF mots (1,1) + LogReg C=10": (lambda: word((1, 1)), lambda: lr(10)),
    "TF-IDF mots (1,2) + LogReg C=10": (lambda: word((1, 2)), lambda: lr(10)),
    "TF-IDF mots (1,2) + LogReg C=100": (lambda: word((1, 2)), lambda: lr(100)),
    "TF-IDF car. (2,5) + LogReg C=10": (lambda: char((2, 5)), lambda: lr(10)),
    "TF-IDF mots+car. + LogReg C=10": (both, lambda: lr(10)),
    "TF-IDF mots+car. + LogReg C=100": (both, lambda: lr(100)),
    "TF-IDF mots (1,2) + LinearSVC C=1": (lambda: word((1, 2)), lambda: LinearSVC(C=1.0, random_state=SEED)),
    "TF-IDF mots+car. + LinearSVC C=1": (both, lambda: LinearSVC(C=1.0, random_state=SEED)),
}


def load_data():
    train = pd.read_csv(config.DATA_DIR / "textes.csv")
    img = pd.read_csv(config.DATA_DIR / "images.csv")
    aud = pd.read_csv(config.DATA_DIR / "audios.csv")
    ext_texts = list(img.texte_attendu_ocr) + list(aud.transcription_reference)
    ext_labels = list(img.sentiment_ref) + list(aud.sentiment_ref)
    return train, ext_texts, ext_labels


def main():
    train, ext_texts, ext_labels = load_data()
    X = np.array([clean_text(t) for t in train.texte])
    y = np.array(train.sentiment_ref)
    Xe = np.array([clean_text(t) for t in ext_texts])
    ye = np.array(ext_labels)
    print(f"Train: {len(X)} textes, répartition {dict(pd.Series(y).value_counts())}; externe: {len(Xe)} textes")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    # CV complémentaire sur les 42 textes poolés (24 + 18 réf.) : estimation moins dégénérée, NON utilisée pour la sélection
    Xp, yp = np.concatenate([X, Xe]), np.concatenate([y, ye])
    cv_pool = RepeatedStratifiedKFold(n_splits=6, n_repeats=N_REPEATS, random_state=SEED)
    rows, oof = [], {}
    for name, (vec_f, clf_f) in CANDIDATES.items():
        accs, f1s = [], []
        preds = []
        for tr, te in cv.split(X, y):
            pipe = Pipeline([("vec", vec_f()), ("clf", clf_f())]).fit(X[tr], y[tr])
            p = pipe.predict(X[te])
            accs.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="macro", labels=LABELS, zero_division=0))
            preds.append((te, p))
        pool_accs = [accuracy_score(yp[te], Pipeline([("vec", vec_f()), ("clf", clf_f())]).fit(Xp[tr], yp[tr]).predict(Xp[te]))
                     for tr, te in cv_pool.split(Xp, yp)]
        full = Pipeline([("vec", vec_f()), ("clf", clf_f())]).fit(X, y)
        pe = full.predict(Xe)
        rows.append({
            "Modèle": name,
            "CV accuracy (moy)": np.mean(accs), "CV accuracy (écart-type)": np.std(accs),
            "CV macro-F1 (moy)": np.mean(f1s), "CV macro-F1 (écart-type)": np.std(f1s),
            "CV poolée 42 textes accuracy (moy)": np.mean(pool_accs),
            "Externe accuracy (18 textes OCR+ASR réf.)": accuracy_score(ye, pe),
            "Externe macro-F1": f1_score(ye, pe, average="macro", labels=LABELS, zero_division=0),
            "predict_proba": hasattr(full.named_steps["clf"], "predict_proba"),
        })
        oof[name] = preds
        print(f"{name:42s} CV acc {np.mean(accs):.3f}±{np.std(accs):.3f}  F1 {np.mean(f1s):.3f}  externe {accuracy_score(ye, pe):.3f}")

    table = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    table.round(3).to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "model_comparison.md").write_text(table.round(3).to_markdown(index=False) if _has_tabulate() else table.round(3).to_string(index=False))

    # --- Sélection : meilleur macro-F1 en CV parmi les modèles dotés de predict_proba (exigé pour « confiance »)
    eligible = table[table.predict_proba & ~table["Modèle"].str.startswith("Baseline")]
    best_name = eligible.sort_values(["CV macro-F1 (moy)", "CV macro-F1 (écart-type)"], ascending=[False, True]).iloc[0]["Modèle"]
    print("\nModèle retenu (CV macro-F1) :", best_name)
    vec_f, clf_f = CANDIDATES[best_name]

    # Matrice de confusion hors-échantillon (agrégée sur les répétitions de la CV)
    yt, yp = [], []
    for te, p in oof[best_name]:
        yt.extend(y[te]); yp.extend(p)
    cm_cv = confusion_matrix(yt, yp, labels=LABELS)
    cr_cv = classification_report(yt, yp, labels=LABELS, output_dict=True, zero_division=0)

    # Modèle final entraîné sur les 24 textes ; évaluation externe
    vec, clf = vec_f(), clf_f()
    clf = clone(clf)
    Xv = vec.fit_transform(X)
    clf.fit(Xv, y)
    pe = clf.predict(vec.transform(Xe))
    cm_ext = confusion_matrix(ye, pe, labels=LABELS)
    cr_ext = classification_report(ye, pe, labels=LABELS, output_dict=True, zero_division=0)
    conf_ext = clf.predict_proba(vec.transform(Xe)).max(axis=1)
    errors = [{"texte": t, "attendu": a, "predit": p, "confiance": round(float(c), 3)}
              for t, a, p, c in zip(ext_texts, ye, pe, conf_ext) if a != p]

    joblib.dump(clf, config.MODELS_DIR / "model.pkl")
    joblib.dump(vec, config.MODELS_DIR / "vectorizer.pkl")

    summary = {
        "modele_retenu": best_name, "seed": SEED, "cv": f"RepeatedStratifiedKFold({N_SPLITS} plis x {N_REPEATS} répétitions)",
        "n_train": int(len(X)), "n_externe": int(len(Xe)),
        "cv_confusion_matrix": cm_cv.tolist(), "cv_classification_report": cr_cv,
        "externe_confusion_matrix": cm_ext.tolist(), "externe_classification_report": cr_ext,
        "externe_confiance_moyenne": float(conf_ext.mean()), "externe_erreurs": errors,
        "labels": LABELS,
    }
    (REPORTS / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, title in zip(axes, [cm_cv, cm_ext], ["Validation croisée (hors-échantillon)", "Test externe (18 textes OCR+ASR réf.)"]):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); ax.set_title(title, fontsize=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrices.png", dpi=150)

    print("\n== Matrice de confusion CV (lignes=réel, colonnes=prédit, ordre", LABELS, ")\n", cm_cv)
    print(classification_report(yt, yp, labels=LABELS, zero_division=0))
    print("== Test externe\n", cm_ext)
    print(classification_report(ye, pe, labels=LABELS, zero_division=0))
    print("Erreurs externes :", json.dumps(errors, ensure_ascii=False, indent=1))
    print("Artefacts sauvegardés dans", config.MODELS_DIR)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
