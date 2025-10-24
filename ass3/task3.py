import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_OUTDIR = Path("outputs_task3_kaggle")
DEFAULT_TOPK = 10
DEFAULT_MIN_PER_GENRE = 10
DEFAULT_NUM_GENRES = 4

LYRICS_FILE = "lyrics-data.csv"  # requires columns: ALink, Lyric, (language optional)
ARTISTS_FILE = "artists-data.csv"  # requires columns: Link, Artist, Genres

TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

EN_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on",
    "that", "the", "to", "was", "were", "will", "with", "i", "you", "me", "my", "we", "our", "your", "they", "them",
    "their", "this", "those", "these", "she", "her", "him", "his", "itself", "yourself", "ourselves", "yours",
    "im", "ive", "id", "ill", "youre", "youve", "youd", "youll", "weve", "wed", "well", "theyre", "theyve", "theyll",
    "cant", "dont", "won", "wont", "doesnt", "isnt", "arent", "wasnt", "werent", "shouldnt", "couldnt", "wouldnt",
    "ooh", "oh", "yeah", "la", "na", "nanana", "uh", "uhh", "woo", "woah", "yo", "m", "re", "ve", "ll", "d", "s", "t"
}
RU_STOP = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да",
    "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей"
}

_SPACY_EN = None
_PYMORPHY_RU = None

def _maybe_load_spacy_en():
    global _SPACY_EN
    if _SPACY_EN is not None:
        return _SPACY_EN
    try:
        import spacy
        try:
            _SPACY_EN = spacy.load("en_core_web_sm")
        except Exception:
            _SPACY_EN = spacy.blank("en")
        return _SPACY_EN
    except Exception:
        return None


def _maybe_load_pymorphy2():
    global _PYMORPHY_RU
    if _PYMORPHY_RU is not None:
        return _PYMORPHY_RU
    try:
        import pymorphy2
        _PYMORPHY_RU = pymorphy2.MorphAnalyzer()
        return _PYMORPHY_RU
    except Exception:
        return None


def simple_lemma_en(tok: str) -> str:
    t = tok
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("sses"):
        return t[:-2]
    for suf in ("ing", "ed", "ly", "iness", "ness", "ment", "ments", "ers", "er", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[:-len(suf)]
    return t


def lemma_en(tokens: List[str]) -> List[str]:
    nlp = _maybe_load_spacy_en()
    if nlp is None or getattr(nlp, "has_pipe", lambda *a, **k: False)("lemmatizer") is False:
        return [simple_lemma_en(t) for t in tokens]
    doc = nlp(" ".join(tokens))
    outs = []
    for tok in doc:
        lem = tok.lemma_.lower().strip()
        if lem in ("-pron-", "") or lem is None:
            lem = tok.text.lower()
        outs.append(lem)
    return outs


def lemma_ru(tokens: List[str]) -> List[str]:
    morph = _maybe_load_pymorphy2()
    outs = []
    if morph is None:
        return tokens
    for t in tokens:
        try:
            p = morph.parse(t)
            outs.append(p[0].normal_form if p else t)
        except Exception:
            outs.append(t)
    return outs


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def preprocess_row(text: str, lang_hint: Optional[str] = None) -> List[str]:
    toks = tokenize(text)
    if lang_hint and str(lang_hint).lower().startswith("ru"):
        toks = [t for t in toks if t not in RU_STOP]
        toks = lemma_ru(toks)
    else:
        toks = [t for t in toks if t not in EN_STOP]
        toks = lemma_en(toks)
    toks = [t for t in toks if t and t not in EN_STOP and t not in RU_STOP]
    return toks


CATEGORY_LEXICONS = {
    "time_of_day": {"morning", "noon", "afternoon", "evening", "night", "midnight", "sunrise", "sunset", "day"},
    "seasons": {"winter", "spring", "summer", "autumn", "fall"},
    "body_parts": {"eye", "eyes", "heart", "hand", "hands", "head", "face", "lips", "mouth", "skin", "body"},
}


def parse_args():
    ap = argparse.ArgumentParser(description="Task 3 — Frequency & category analysis on Kaggle lyrics dataset")
    ap.add_argument("--genres", nargs="*", default=None, help="Explicit list of 4 genres to use (case-insensitive).")
    ap.add_argument("--num_genres", type=int, default=DEFAULT_NUM_GENRES,
                    help="How many genres to pick if not specified (default 4).")
    ap.add_argument("--min_per_genre", type=int, default=DEFAULT_MIN_PER_GENRE,
                    help="Minimum songs per genre (default 10).")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Top-K words per genre (default 10).")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR),
                    help="Output directory (default outputs_task3_kaggle/)")
    return ap.parse_args()


def load_kaggle_lyrics_and_artists(base_dir: Path) -> pd.DataFrame:
    lyr_path = base_dir / LYRICS_FILE
    art_path = base_dir / ARTISTS_FILE
    if not lyr_path.exists() or not art_path.exists():
        raise FileNotFoundError(f"Expected {LYRICS_FILE} and {ARTISTS_FILE} in {base_dir}")
    try:
        lyr = pd.read_csv(lyr_path, on_bad_lines="skip")
    except Exception:
        lyr = pd.read_csv(lyr_path, on_bad_lines="skip", sep=";")
    try:
        art = pd.read_csv(art_path, on_bad_lines="skip")
    except Exception:
        art = pd.read_csv(art_path, on_bad_lines="skip", sep=";")
    needed_lyrics_cols = {"ALink", "Lyric"}
    needed_art_cols = {"Link", "Artist", "Genres"}
    if not needed_lyrics_cols.issubset(set(lyr.columns)) or not needed_art_cols.issubset(set(art.columns)):
        raise ValueError(
            "CSV schema mismatch: lyrics-data.csv must have ALink/Lyric; artists-data.csv must have Link/Artist/Genres")
    merged = lyr.merge(art[["Link", "Artist", "Genres"]], left_on="ALink", right_on="Link", how="left")

    # Label = first listed genre
    def pick_genre(g):
        if pd.isna(g):
            return None
        g = str(g).lower()
        for sep in [",", ";", "/", "|"]:
            if sep in g:
                return g.split(sep)[0].strip()
        return g.strip()

    merged["label"] = merged["Genres"].apply(pick_genre)
    merged = merged.rename(columns={"Lyric": "text"})

    if "language" in merged.columns:
        merged["language"] = merged["language"].astype(str)
    else:
        merged["language"] = None

    merged = merged[["text", "label", "Artist", "language"]]
    merged = merged.dropna(subset=["text", "label"])
    merged["text"] = merged["text"].astype(str)
    merged["label"] = merged["label"].astype(str).str.strip().str.lower()
    merged = merged.drop_duplicates(subset=["text"])
    return merged


def select_genres(df: pd.DataFrame, genres: Optional[Sequence[str]], num_genres: int, min_per_genre: int) -> List[str]:
    counts = df["label"].value_counts()
    if genres:
        wanted = [g.lower() for g in genres]
        ok = [g for g in wanted if counts.get(g, 0) >= min_per_genre]
        if len(ok) < len(wanted):
            missing = [g for g in wanted if counts.get(g, 0) < min_per_genre]
            raise ValueError(f"Requested genres lack enough songs: {missing} (need >= {min_per_genre} each)")
        return ok[:num_genres]
    eligible = counts[counts >= min_per_genre].index.tolist()
    if len(eligible) < num_genres:
        raise ValueError(f"Not enough genres with >= {min_per_genre} songs. Found {len(eligible)}: {eligible[:8]}")
    return eligible[:num_genres]


def build_per_genre_counters(df: pd.DataFrame) -> Dict[str, Counter]:
    per: Dict[str, Counter] = {}
    for lab, grp in df.groupby("label"):
        ctr = Counter()
        for _, row in grp.iterrows():
            lang = row.get("language", None)
            tokens = preprocess_row(row["text"], lang_hint=lang)
            ctr.update(tokens)
        per[lab] = ctr
    return per


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def save_topk(per: Dict[str, Counter], outdir: Path, k: int):
    rows = []
    for lab, ctr in per.items():
        for w, c in ctr.most_common(k):
            rows.append((lab, w, int(c)))
        words, counts = zip(*ctr.most_common(min(k, len(ctr)))) if ctr else ([], [])
        if words:
            plt.figure(figsize=(8, 4), dpi=140)
            plt.bar(range(len(words)), counts)
            plt.xticks(range(len(words)), words, rotation=45, ha="right")
            plt.title(f"Top {len(words)} words — {lab}")
            plt.tight_layout()
            plt.savefig(outdir / f"task3_top{k}_{lab}.png", bbox_inches="tight", dpi=160)
            plt.close()
    pd.DataFrame(rows, columns=["genre", "word", "count"]).to_csv(outdir / "task3_top_words.csv", index=False,
                                                                  encoding="utf-8")


def save_categories(per: Dict[str, Counter], outdir: Path, lexicons: Dict[str, set]):
    rows = []
    for lab, ctr in per.items():
        for cat, vocab in lexicons.items():
            rows.append((lab, cat, int(sum(ctr[w] for w in vocab))))
    df = pd.DataFrame(rows, columns=["genre", "category", "count"])
    df.to_csv(outdir / "task3_categories.csv", index=False, encoding="utf-8")
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat].sort_values("count", ascending=False)
        if sub.empty:
            continue
        plt.figure(figsize=(7, 4), dpi=140)
        plt.bar(range(len(sub)), sub["count"].values)
        plt.xticks(range(len(sub)), sub["genre"].values, rotation=30, ha="right")
        plt.title(f"Category: {cat}")
        plt.tight_layout()
        plt.savefig(outdir / f"task3_category_{cat}.png", bbox_inches="tight", dpi=160)
        plt.close()


def save_rare_words(per: Dict[str, Counter], outdir: Path, max_count: int = 3):
    rows = []
    for lab, ctr in per.items():
        rares = [(w, c) for w, c in ctr.items() if c <= max_count]
        for w, c in sorted(rares, key=lambda x: (x[1], x[0])):
            rows.append((lab, w, int(c)))
    pd.DataFrame(rows, columns=["genre", "word", "count"]).to_csv(outdir / "task3_rare_words.csv", index=False,
                                                                  encoding="utf-8")


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df_all = load_kaggle_lyrics_and_artists(here)
    df_all = df_all.dropna(subset=["label"])

    chosen = select_genres(df_all, args.genres, args.num_genres, args.min_per_genre)
    df = df_all[df_all["label"].isin(chosen)].copy()
    df.to_csv(outdir / "task3_corpus_subset.csv", index=False, encoding="utf-8")

    per = build_per_genre_counters(df)

    save_topk(per, outdir, k=args.topk)
    save_categories(per, outdir, lexicons=CATEGORY_LEXICONS)
    save_rare_words(per, outdir, max_count=3)

    print("Genres selected:", chosen)
    print("Counts per genre:", df["label"].value_counts().to_dict())
    print(f"Done. Artifacts saved under {outdir}/")

if __name__ == "__main__":
    main()
