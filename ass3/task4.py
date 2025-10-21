import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # already save-only; keep this
import matplotlib.pyplot as plt
from matplotlib import font_manager
import platform

OUTDIR = "outputs_task4"
SPACY_MODEL = "xx_ent_wiki_sm"
TOPK_PER_TYPE = 20

CORPUS = [
    # --- English (EN) ---
    ("en", "OpenAI released a new model in San Francisco, with Sam Altman presenting the demo."),
    ("en", "Apple announced iPhone sales growth at its Cupertino headquarters."),
    ("en", "The United Nations held a climate summit in New York City this September."),
    ("en", "NASA confirmed that the Artemis mission will return to the Moon."),
    ("en", "The Economist reported on inflation trends in the European Union."),
    # --- Spanish (ES) ---
    ("es", "El presidente de España visitó Barcelona para inaugurar un nuevo hospital."),
    ("es", "Messi ganó el premio en París y agradeció a su equipo del Inter Miami."),
    ("es", "La Universidad de Buenos Aires publicó un estudio sobre educación pública."),
    ("es", "El Banco de México mantuvo sin cambios la tasa de interés."),
    ("es", "Iberia abrió una nueva ruta entre Madrid y Bogotá."),
    # --- Russian (RU) ---
    ("ru", "Президент посетил Санкт-Петербург и встретился с губернатором города."),
    ("ru", "Яндекс объявил об обновлении сервиса в Москве на прошлой неделе."),
    ("ru", "Сбербанк инвестирует в стартапы в области искусственного интеллекта."),
    ("ru", "Сборная России сыграла товарищеский матч в Казани."),
    ("ru", "Компания Ростелеком расширяет сеть в Сибири."),
    # --- Korean (KO) ---
    ("ko", "서울에서 열리는 컨퍼런스에서 현대자동차가 새로운 전기차를 공개했다."),
    ("ko", "부산시는 해운대구에 관광 인프라 투자를 발표했다."),
    ("ko", "카카오와 네이버는 인공지능 연구 협력을 강화한다고 밝혔다."),
    ("ko", "삼성전자는 반도체 공장 증설 계획을 발표했다."),
    ("ko", "한국은행은 기준금리를 동결했다.")
]


def set_font_fallback():
    # Build a fallback list that actually contains CJK + Latin-extended glyphs
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
    fallback = [name for name in preferred if name in installed] or ["DejaVu Sans"]

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = fallback
    matplotlib.rcParams["axes.unicode_minus"] = False
    # Optional: better embedding in PDFs
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def ensure_spacy(model_name: str):
    import importlib, subprocess, sys
    try:
        importlib.import_module(model_name)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        except Exception:
            pass
    import spacy
    return spacy.load(model_name)


def normalize_entity_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t.strip())
    return t.strip(" '\"()[]{}.,;:")


def plot_counts(df_counts: pd.DataFrame, outpath: Path, title: str):
    if df_counts.empty:
        return
    plt.figure(figsize=(10, 5), dpi=140)
    x = np.arange(len(df_counts))
    plt.bar(x, df_counts["count"].values)
    plt.xticks(x, df_counts["item"].values, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", dpi=160)
    plt.close()


def main():
    set_font_fallback()
    nlp = ensure_spacy(SPACY_MODEL)
    rows = []
    for lang, sent in CORPUS:
        doc = nlp(sent)
        for ent in doc.ents:
            rows.append((lang, sent, normalize_entity_text(ent.text), ent.label_))

    if not rows:
        raise SystemExit("No entities extracted. Try a different model or adjust the corpus.")

    df = pd.DataFrame(rows, columns=["lang", "sentence", "entity", "label"])
    outdir = Path(OUTDIR);
    outdir.mkdir(parents=True, exist_ok=True)

    # Save raw extractions
    df.to_csv(outdir / "task4_entities_raw.csv", index=False, encoding="utf-8")

    # Per-language counts (entity surface forms)
    per_lang = df.groupby(["lang", "label", "entity"]).size().reset_index(name="count")
    per_lang.sort_values(["lang", "label", "count"], ascending=[True, True, False]).to_csv(
        outdir / "task4_entities_per_language.csv", index=False, encoding="utf-8"
    )

    # Global counts
    global_counts = df.groupby(["label", "entity"]).size().reset_index(name="count")
    global_counts.sort_values(["label", "count"], ascending=[True, False]).to_csv(
        outdir / "task4_entities_global.csv", index=False, encoding="utf-8"
    )

    # Plots: per-language, per label (top-K)
    for lang in sorted(df["lang"].unique()):
        sub_lang = per_lang[per_lang["lang"] == lang]
        for lab in sorted(sub_lang["label"].unique()):
            sub = sub_lang[sub_lang["label"] == lab].sort_values("count", ascending=False).head(TOPK_PER_TYPE)
            if sub.empty: continue
            plot_counts(
                sub.rename(columns={"entity": "item"})[["item", "count"]],
                outdir / f"top_{lang}_{lab}.png",
                f"Top {len(sub)} {lab} — {lang}"
            )

    # Plots: global per label
    for lab in sorted(global_counts["label"].unique()):
        sub = global_counts[global_counts["label"] == lab].sort_values("count", ascending=False).head(TOPK_PER_TYPE)
        if sub.empty: continue
        plot_counts(
            sub.rename(columns={"entity": "item"})[["item", "count"]],
            outdir / f"global_top_{lab}.png",
            f"Global Top {len(sub)} {lab}"
        )

    print(f"Processed {len(CORPUS)} sentences across {len(set(l for l, _ in CORPUS))} languages. Outputs in {OUTDIR}/")


if __name__ == "__main__":
    main()
