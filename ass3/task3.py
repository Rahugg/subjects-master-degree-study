import re, random
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

CSV_DIR = "csv"
TOPK = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_LR = True
MIN_DF = 3
MAX_DF = 0.9
NGRAM = (1, 2)
OUTDIR = "outputs_task3"

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
STOP = set(ENGLISH_STOP_WORDS)
TOKEN_RE = re.compile(r"[A-Za-z']+")


def set_font_for_unicode():
    import matplotlib
    from matplotlib import font_manager
    import platform

    if platform.system() == "Darwin":  # macOS
        preferred = [
            "Noto Sans CJK KR", "Noto Sans", "Arial Unicode MS",
            "AppleGothic", "Helvetica Neue", "Arial"
        ]
    elif platform.system() == "Windows":
        preferred = ["Malgun Gothic", "Yu Gothic", "Segoe UI", "Arial Unicode MS", "Arial"]
    else:  # Linux
        preferred = ["Noto Sans CJK KR", "Noto Sans", "DejaVu Sans", "NanumGothic"]

    installed = {f.name for f in font_manager.fontManager.ttflist}

    # Build a fallback list from what's installed
    fallback = [name for name in preferred if name in installed]
    if not fallback:
        # keep DejaVu as a final fallback
        fallback = ["DejaVu Sans"]

    # Set a family + a fallback list
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = fallback

    # Minus signs sometimes render as squares with some CJK fonts
    matplotlib.rcParams["axes.unicode_minus"] = False


def simple_lemma(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4: return tok[:-3] + "y"
    if tok.endswith("sses"): return tok[:-2]
    for suf in ("ing", "ed", "ly", "ness", "ment", "ments", "ers", "er", "s"):
        if tok.endswith(suf) and len(tok) > len(suf) + 2: return tok[:-len(suf)]
    return tok


def analyzer(text: str):
    toks = TOKEN_RE.findall(str(text).lower())
    toks = [t for t in toks if t not in STOP and t not in {"'s", "n't"}]
    return [simple_lemma(t) for t in toks]


def load_csv_folder(csv_dir: str):
    p = Path(csv_dir)
    if not p.exists():
        alt = Path("/mnt/data/csv")
        if alt.exists(): p = alt
    files = sorted(list(p.glob("*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in '{p}'. Put your lyric CSVs in that folder.")
    frames = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                df = pd.read_csv(fh, on_bad_lines="skip")
        except UnicodeDecodeError:
            with open(f, "r", encoding="latin-1") as fh:
                df = pd.read_csv(fh, on_bad_lines="skip")
        cols_lower = {c.lower(): c for c in df.columns}
        text_col = cols_lower.get("lyric") or cols_lower.get("lyrics") or cols_lower.get("text") or cols_lower.get(
            "content") or cols_lower.get("song_lyrics") or cols_lower.get("document")
        label_col = cols_lower.get("artist") or cols_lower.get("genre") or cols_lower.get("label") or cols_lower.get(
            "class") or cols_lower.get("category") or cols_lower.get("style") or cols_lower.get("album")
        if not text_col or not label_col:
            raise SystemExit(
                f"File {f.name} must include a lyric/text column and an artist/genre label column. Found: {list(df.columns)}")
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df, p


def frequency_analysis(df, outdir: Path, topk=TOPK):
    freq = {}
    for lab, grp in df.groupby("label"):
        counter = Counter()
        for t in grp["text"]:
            counter.update(analyzer(t))
        freq[lab] = counter
    rows = []
    for lab, counter in freq.items():
        for w, c in counter.most_common(topk):
            rows.append((lab, w, c))
    topdf = pd.DataFrame(rows, columns=["label", "word", "count"])
    outdir.mkdir(exist_ok=True, parents=True)
    topdf.to_csv(outdir / "task3_top_words.csv", index=False, encoding="utf-8")
    for lab in sorted(freq.keys()):
        items = freq[lab].most_common(topk)
        if not items: continue
        words, counts = zip(*items)
        plt.figure(figsize=(8, 4), dpi=140)
        plt.bar(range(len(words)), counts)
        plt.xticks(range(len(words)), words, rotation=45, ha="right")
        plt.title(f"Top {topk} words — {lab}")
        plt.tight_layout()
        plt.savefig(outdir / f"task3_top_{lab}.png", bbox_inches="tight", dpi=160)
        plt.close()
    return topdf


def train_and_eval(df, outdir: Path):
    if df["label"].nunique() < 2:
        print("Only one label present — skipping classifier training. Provide CSVs for at least two artists/genres.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )
    vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=analyzer,
        preprocessor=None,
        token_pattern=None,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=2000, solver="liblinear") if USE_LR else MultinomialNB()
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"Accuracy={acc:.4f}  F1_macro={f1:.4f}")
    print(classification_report(y_test, pred, digits=3))
    labels = sorted(df["label"].unique().tolist())
    cm = confusion_matrix(y_test, pred, labels=labels)
    plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title("Confusion matrix — Task 3")
    plt.xlabel("pred")
    plt.ylabel("true")
    outdir.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(outdir / "task3_confusion.png", bbox_inches="tight", dpi=160)
    plt.close()
    return vec, clf, labels


def main():
    set_font_for_unicode()
    df, base = load_csv_folder(CSV_DIR)
    print(f"Loaded {len(df)} lyrics across {df['label'].nunique()} labels from {base}.")
    outdir = Path(OUTDIR)
    frequency_analysis(df, outdir, topk=TOPK)
    train_and_eval(df, outdir)

if __name__ == "__main__":
    main()
