import os, re, math, heapq, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

GLOB_PATTERN = "article*.txt"
OUTDIR = "summaries"
MAX_SENT = 5
RATIO = 0.2
DO_ABSTRACTIVE = True
HF_MODEL = "sshleifer/distilbart-cnn-12-6"
random.seed(7);
np.random.seed(7)
STOP = set(ENGLISH_STOP_WORDS)
TOKEN_RE = re.compile(r"[A-Za-z']+")


def sent_split(text):
    text = re.sub(r'\s+', ' ', text.strip())
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"(\[])', text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents


def simple_lemma(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4: return tok[:-3] + "y"
    if tok.endswith("sses"): return tok[:-2]
    for suf in ("ing", "ed", "ly", "ness", "ment", "ments", "ers", "er", "s"):
        if tok.endswith(suf) and len(tok) > len(suf) + 2: return tok[:-len(suf)]
    return tok


def normalize(text: str):
    toks = TOKEN_RE.findall(text.lower())
    toks = [t for t in toks if t not in STOP and t not in {"'s", "n't"}]
    toks = [simple_lemma(t) for t in toks]
    return " ".join(toks)


def textrank_summary(text: str, max_sent=5, ratio=0.2):
    sents = sent_split(text)
    if not sents: return ""
    n = len(sents)
    k = max(1, min(max_sent, max(1, int(math.ceil(n * ratio)))))
    norm = [normalize(s) for s in sents]
    vec = TfidfVectorizer(min_df=1, max_df=0.9)
    X = vec.fit_transform(norm)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    d = 0.85
    scores = np.ones(n) / n
    M = sim / (sim.sum(axis=1, keepdims=True) + 1e-9)
    for _ in range(40):
        scores = (1 - d) / n + d * M.T.dot(scores)
    idx = np.argsort(-scores)[:k]
    idx = sorted(idx)
    return " ".join([sents[i] for i in idx])


def abstractive_summary(text: str, model_name=HF_MODEL):
    if not DO_ABSTRACTIVE: return ""
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model=model_name, device=-1)
        # chunk long inputs
        max_chunk = 900
        sents = sent_split(text)
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) < max_chunk:
                cur += (" " if cur else "") + s
            else:
                chunks.append(cur);
                cur = s
        if cur: chunks.append(cur)
        outs = []
        for ch in chunks:
            out = summarizer(ch, max_length=180, min_length=50, truncation=True)
            outs.append(out[0]["summary_text"])
        return " ".join(outs)
    except Exception:
        return ""


def topk_tfidf_words(text, k=20):
    vec = TfidfVectorizer()
    X = vec.fit_transform([text])
    arr = np.asarray(X.todense()).ravel()
    idx = np.argsort(-arr)[:k]
    feats = vec.get_feature_names_out()
    return [feats[i] for i in idx]


def coverage_score(summary, doc, k=20):
    top = set(topk_tfidf_words(doc, k))
    got = 0
    summ_norm = set(normalize(summary).split())
    for w in top:
        if w in summ_norm: got += 1
    return got / max(1, len(top))


def summarize_files(file_paths):
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    rows = []
    for p in file_paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        ext = textrank_summary(text, max_sent=MAX_SENT, ratio=RATIO)
        abs_sum = abstractive_summary(text) if DO_ABSTRACTIVE else ""
        out_ext = Path(OUTDIR) / (Path(p).stem + "_extractive.txt")
        with open(out_ext, "w", encoding="utf-8") as g:
            g.write(ext)
        out_abs = ""
        if abs_sum:
            out_abs = Path(OUTDIR) / (Path(p).stem + "_abstractive.txt")
            with open(out_abs, "w", encoding="utf-8") as g:
                g.write(abs_sum)
        cov_ext = coverage_score(ext, text, k=20) if ext else 0.0
        cov_abs = coverage_score(abs_sum, text, k=20) if abs_sum else 0.0
        rows.append([Path(p).name, len(sent_split(text)), len(sent_split(ext)), cov_ext, str(out_ext),
                     (str(out_abs) if out_abs else ""), cov_abs])
        print(f"Wrote {out_ext}" + (f" and {out_abs}" if out_abs else ""))
    return pd.DataFrame(rows,
                        columns=["file", "sent_total", "sent_summary_ext", "coverage_ext@20", "path_ext", "path_abs",
                                 "coverage_abs@20"])


def main():
    files = sorted(Path(".").glob(GLOB_PATTERN))
    if not files:
        files = [Path(f"/mnt/data/article{i}.txt") for i in range(1, 6) if Path(f"/mnt/data/article{i}.txt").exists()]
    if not files:
        raise SystemExit("No input files found next to the script.")
    df = summarize_files([str(p) for p in files])
    try:
        from caas_jupyter_tools import display_dataframe_to_user
        display_dataframe_to_user("Task 2 – Summaries", df)
    except Exception:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
