#!/usr/bin/env python3
import argparse, re, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

random.seed(7)
np.random.seed(7)

STOPWORDS = set(ENGLISH_STOP_WORDS)
TOKEN_RE = re.compile(r"[a-zA-Z']+")
LEX_POS = {"love": 3, "loved": 3, "like": 2, "liked": 2, "enjoy": 2, "enjoyed": 2, "adore": 3, "adored": 3, "great": 2,
           "good": 1.5,
           "amazing": 3, "awesome": 3, "fantastic": 3, "excellent": 3, "wonderful": 3, "brilliant": 3, "hilarious": 2.5,
           "engaging": 2, "inspiring": 2.5, "masterpiece": 3, "beautiful": 2.5, "superb": 3, "charming": 2,
           "delightful": 2.5,
           "satisfying": 2, "touching": 2, "powerful": 2.5, "thrill": 2}
LEX_NEG = {"hate": -3, "hated": -3, "dislike": -2, "disliked": -2, "boring": -2.5, "awful": -3, "terrible": -3,
           "horrible": -3,
           "worst": -3, "bad": -2, "dull": -2, "mess": -2, "waste": -2.5, "stupid": -2.5, "unfunny": -2,
           "unwatchable": -3,
           "painful": -2.5, "weak": -1.5, "cringe": -2, "forgettable": -2, "ridiculous": -2, "flawed": -1.5,
           "tedious": -2.5,
           "predictable": -1.5, "lazy": -1.5, "cheap": -1.5}
NEGATORS = {"not", "never", "no", "hardly", "barely", "scarcely", "n't"}


def simple_lemma(t):
    if t.endswith("ies") and len(t) > 4: return t[:-3] + "y"
    if t.endswith("sses"): return t[:-2]
    for suf in ("ing", "ed", "ly", "ness", "ment", "ments", "ers", "er", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 2: return t[:-len(suf)]
    return t


def analyzer(text):
    toks = TOKEN_RE.findall(text.lower())
    toks = [t for t in toks if t not in STOPWORDS and t not in {"'s", "n't"}]
    return [simple_lemma(t) for t in toks]


def sentiment_score(text):
    toks = analyzer(text)
    s = 0.0
    flip = False
    for t in toks:
        if t in NEGATORS: flip = True; continue
        v = 0.0
        if t in LEX_POS: v += LEX_POS[t]
        if t in LEX_NEG: v += LEX_NEG[t]
        if flip and v != 0.0: v = -v; flip = False
        s += v
    return s


def tune_lex_threshold(texts, labels):
    vals = np.array([sentiment_score(t) for t in texts])
    y = np.array(labels)
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    best_f1, best_th = -1, 0.0
    for th in grid:
        pred = np.where(vals < -th, "negative", np.where(vals > th, "positive", "neutral"))
        f1 = f1_score(y, pred, average="macro")
        if f1 > best_f1: best_f1, best_th = f1, th
    return best_th


def evaluate(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(title)
    print(f"accuracy={acc:.4f}  f1_macro={f1:.4f}")
    print(classification_report(y_true, y_pred, digits=3))
    labs = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labs)
    plt.figure(figsize=(4, 4), dpi=140)
    ax = plt.gca()
    ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(range(len(labs)))
    ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_yticklabels(labs)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    plt.tight_layout()
    plt.show()


def brief_reason(txt):
    t = txt.lower()
    if any(w in t for w in ["yeah right", "as if"]): return "sarcasm"
    if any(w in t for w in [" not ", "n't", " never", " no "]): return "negation"
    if len(analyzer(txt)) < 4: return "short/ambiguous"
    return "lexical/ambiguous"


def load_imdb(csv_path):
    path = csv_path if csv_path else "IMDB Dataset.csv"
    with open(path, "r", encoding="utf-8") as f:
        df = pd.read_csv(f, on_bad_lines="skip")
    if "text" not in df.columns and "review" in df.columns: df = df.rename(columns={"review": "text"})
    if "label" not in df.columns and "sentiment" in df.columns: df = df.rename(columns={"sentiment": "label"})
    if "text" not in df.columns or "label" not in df.columns:
        raise SystemExit("CSV must have columns: review/sentiment or text/label")
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["label"] = df["label"].str.lower().str.strip()
    return df


def simple_wordcloud(words, weights, title, outfile=None, n=60):
    import random
    idx = np.argsort(-weights)[:n]
    words = words[idx];
    weights = weights[idx]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    plt.figure(figsize=(8, 5), dpi=140)
    ax = plt.gca();
    ax.set_axis_off()
    used = []
    for w, s in zip(words, weights):
        size = 8 + 22 * s
        x, y = random.random(), random.random()
        tries = 0
        while any(abs(x - ux) < 0.08 and abs(y - uy) < 0.06 for ux, uy, _ in used) and tries < 200:
            x, y = random.random(), random.random();
            tries += 1
        ax.text(x, y, w, fontsize=size, transform=ax.transAxes)
        used.append((x, y, size))
    ax.set_title(title, pad=10)
    plt.tight_layout()
    if outfile: plt.savefig(outfile, bbox_inches="tight", dpi=140)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    df = load_imdb(args.csv)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(df["text"], df["label"], test_size=0.3, stratify=df["label"],
                                                random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    tfidf = TfidfVectorizer(analyzer=analyzer, min_df=2, max_df=0.9)
    A_tr = tfidf.fit_transform(X_tr)
    A_val = tfidf.transform(X_val)
    A_te = tfidf.transform(X_te)

    th = tune_lex_threshold(X_val.tolist(), y_val.tolist())
    scores = np.array([sentiment_score(t) for t in X_te.tolist()])
    lex_pred = np.where(scores < -th, "negative", np.where(scores > th, "positive", "neutral"))
    evaluate(y_te, lex_pred, f"Lexicon-based (th={th})")

    lr = LogisticRegression(max_iter=2000, solver="liblinear")
    nb = MultinomialNB()
    lr.fit(A_tr, y_tr)
    nb.fit(A_tr, y_tr)
    best = lr if f1_score(y_val, lr.predict(A_val), average="macro") >= f1_score(y_val, nb.predict(A_val),
                                                                                 average="macro") else nb
    name = "LR" if best is lr else "NB"
    ml_pred = best.predict(A_te)
    evaluate(y_te, ml_pred, f"ML ({name})")

    def mis(X, y_true, y_pred, k=10):
        out = []
        c = 0
        for txt, yt, yp in zip(X, y_true, y_pred):
            if yt != yp and c < k:
                out.append((yt, yp, txt, brief_reason(txt)))
                c += 1
        return pd.DataFrame(out, columns=["true", "pred", "text", "note"])

    print("\nLexicon misclassified examples:")
    print(mis(X_te.tolist(), y_te.tolist(), lex_pred.tolist()).to_string(index=False, max_colwidth=96))
    print("\nML misclassified examples:")
    print(mis(X_te.tolist(), y_te.tolist(), ml_pred.tolist()).to_string(index=False, max_colwidth=96))

    pos_texts = [t for t, y in zip(X_tr.tolist(), y_tr.tolist()) if y == "positive"]
    neg_texts = [t for t, y in zip(X_tr.tolist(), y_tr.tolist()) if y == "negative"]

    tfidf_wc = TfidfVectorizer(analyzer=analyzer, min_df=5)
    Xpos = tfidf_wc.fit_transform(pos_texts)
    vocab = np.array(tfidf_wc.get_feature_names_out())
    pos_scores = np.asarray(Xpos.mean(axis=0)).ravel()

    Xneg = tfidf_wc.transform(neg_texts)
    neg_scores = np.asarray(Xneg.mean(axis=0)).ravel()

    simple_wordcloud(vocab, pos_scores, "Word cloud – positive (train)", "wc_positive.png")
    simple_wordcloud(vocab, neg_scores, "Word cloud – negative (train)", "wc_negative.png")
    print("Saved word clouds: wc_positive.png, wc_negative.png")

if __name__ == "__main__":
    main()
