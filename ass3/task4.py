from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


OUTDIR = Path("outputs_task4")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Small multilingual corpus (20 sentences, 4 languages) — adapted from your file
CORPUS = [
    # English (EN)
    ("en", "OpenAI released a new model in San Francisco, with Sam Altman presenting the demo."),
    ("en", "Apple announced iPhone sales growth at its Cupertino headquarters."),
    ("en", "The United Nations held a climate summit in New York City this September."),
    ("en", "NASA confirmed that the Artemis mission will return to the Moon."),
    ("en", "The Economist reported on inflation trends in the European Union."),
    # Spanish (ES)
    ("es", "El presidente de España visitó Barcelona para inaugurar un nuevo hospital."),
    ("es", "Messi ganó el premio en París y agradeció a su equipo del Inter Miami."),
    ("es", "La Universidad de Buenos Aires publicó un estudio sobre educación pública."),
    ("es", "El Banco de México mantuvo sin cambios la tasa de interés."),
    ("es", "Iberia abrió una nueva ruta entre Madrid y Bogotá."),
    # Russian (RU)
    ("ru", "Президент посетил Санкт-Петербург и встретился с губернатором города."),
    ("ru", "Яндекс объявил об обновлении сервиса в Москве на прошлой неделе."),
    ("ru", "Сбербанк инвестирует в стартапы в области искусственного интеллекта."),
    ("ru", "Сборная России сыграла товарищеский матч в Казани."),
    ("ru", "Компания Ростелеком расширяет сеть в Сибири."),
    # Korean (KO)
    ("ko", "서울에서 열리는 컨퍼런스에서 현대자동차가 새로운 전기차를 공개했다."),
    ("ko", "부산시는 해운대구에 관광 인프라 투자를 발표했다."),
    ("ko", "카카오와 네이버는 인공지능 연구 협력을 강화한다고 밝혔다."),
    ("ko", "삼성전자는 반도체 공장 증설 계획을 발표했다."),
    ("ko", "한국은행은 기준금리를 동결했다.")
]

def clean_text(s: str) -> str:
    # Lowercase + keep unicode letters/spaces/basic punctuation
    s = s.lower()
    # Remove digits and some symbols (language-ID on chars benefits from keeping letters)
    s = re.sub(r"[0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_dataframe(corpus):
    rows = [{"lang": lang, "text": clean_text(txt)} for lang, txt in corpus]
    df = pd.DataFrame(rows)
    return df

def plot_confusion_matrix(cm, labels, outpath):
    fig = plt.figure(figsize=(6, 5), dpi=140)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    # Write counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=160)
    plt.close(fig)

def main():
    df = build_dataframe(CORPUS)
    df.to_csv(OUTDIR / "task4_language_corpus.csv", index=False, encoding="utf-8")
    
    X = df["text"].values
    y = df["lang"].values
    
    # Stratified split since dataset is small
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Character n-gram TF-IDF + Logistic Regression
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y)))

    # Save metrics
    (OUTDIR / "metrics").mkdir(exist_ok=True, parents=True)
    with open(OUTDIR / "metrics" / "accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.3f}\n")
    with open(OUTDIR / "metrics" / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Save confusion matrix plot and raw CSV
    plot_confusion_matrix(cm, labels=sorted(np.unique(y)), outpath=OUTDIR / "metrics" / "confusion_matrix.png")
    pd.DataFrame(cm, index=sorted(np.unique(y)), columns=sorted(np.unique(y))).to_csv(
        OUTDIR / "metrics" / "confusion_matrix.csv", encoding="utf-8"
    )

    # Save predictions
    pd.DataFrame({"text": X_test, "y_true": y_test, "y_pred": y_pred}).to_csv(
        OUTDIR / "predictions.csv", index=False, encoding="utf-8"
    )

    # Save the trained pipeline (for reuse)
    import joblib
    joblib.dump(pipe, OUTDIR / "language_id_model.joblib")

    print(f"Done. Accuracy={acc:.3f}. Artifacts saved under {OUTDIR}/")
    return acc

if __name__ == "__main__":
    main()
