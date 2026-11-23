"""Nettoyage minimal du texte (étape 1) : minuscules, accents, ponctuation, stop-words.

Les mots de NÉGATION sont conservés volontairement : « pas satisfait » ≠ « satisfait ».
"""
import re
import unicodedata

NEGATIONS = {"ne", "n", "pas", "sans", "jamais", "rien", "non", "aucun", "aucune", "ni", "plus", "personne"}

# Liste de stop-words français courante, écrite à la main (aucune dépendance NLTK / téléchargement).
_RAW_STOPWORDS = """
le la les un une des du de d l au aux et ou a à en dans sur sous par pour avec ce cet cette ces
se sa son ses ma mon mes ta ton tes notre nos votre vos leur leurs qui que quoi dont où
il elle ils elles on nous vous je j tu me m te t y c s qu est sont été etre être ai as avons avez ont
suis es sommes etes êtes ete fait fais faire au aussi mais donc or car si comme
"""


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


STOPWORDS = {strip_accents(w) for w in _RAW_STOPWORDS.split()} - NEGATIONS

_NON_ALNUM = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """minuscules -> sans accents -> ponctuation/chiffres/symboles supprimés -> stop-words retirés."""
    if not isinstance(text, str):
        return ""
    text = strip_accents(text.lower())
    text = text.replace("'", " ").replace("’", " ")  # n'a -> n a (conserve la négation « n »)
    text = _NON_ALNUM.sub(" ", text)
    tokens = _SPACES.sub(" ", text).strip().split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def normalize_for_similarity(text: str) -> str:
    """Normalisation pour comparer OCR/ASR à une référence (sans retirer les stop-words)."""
    return clean_text(text, remove_stopwords=False)
