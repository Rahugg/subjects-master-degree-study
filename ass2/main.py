#!/usr/bin/env python3
# Astana ML reports: linear (cold days) and logistic (flight delays).
# - Shows future predictions (2025/2030/2035) on the linear plot
# - Tolerant to real-world CSV quirks for the logistic part

import os
from time import perf_counter
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
np.seterr(all="ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

SAFE_MIN, SAFE_MAX = -500.0, 500.0
OUT_DIR = "./reports_out"
COLD_CSV = "astana_cold_days.csv"
FLIGHTS_CSV = "astana_flight_delays.csv"
FUTURE_YEARS = (2025, 2030, 2035)
CASE_A = (-28, 1, 9, 2)   # T, Snow, Wind, Vis
CASE_B = (-12, 0, 3, 10)

def normalize(x: np.ndarray):
    # z = (x - mu) / sigma
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sigma, mu, sigma

def denorm_coef(w: np.ndarray, b: float, mu: np.ndarray, sigma: np.ndarray):
    # Convert weights learned on normalized X back to original units
    w_orig = w / sigma.ravel()
    b_orig = b - (mu @ (w / sigma).T).item()
    return w_orig, b_orig

def sigmoid_stable(z: np.ndarray):
    # Numerically safe sigmoid
    z = np.clip(z, SAFE_MIN, SAFE_MAX)
    return 1.0 / (1.0 + np.exp(-z))

def safe_matmul(a: np.ndarray, b: np.ndarray):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return a @ b

def gd_linear(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 4000):
    # Linear regression via batch GD on MSE (bias handled by prepend-ones trick)
    Xn, mu, sigma = normalize(X)
    Xb = np.c_[np.ones(len(Xn)), Xn].astype(np.float64, copy=False)
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    t0 = perf_counter()
    for _ in range(epochs):
        yhat = safe_matmul(Xb, w)
        grad = (2 / len(Xb)) * safe_matmul(Xb.T, (yhat - y))
        w -= lr * grad
    t = perf_counter() - t0
    w_denorm, b_denorm = denorm_coef(w[1:], w[0], mu, sigma)
    return w_denorm, b_denorm, t

def gd_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 3000, l2: float = 1e-4):
    # Logistic regression via batch GD with L2 on weights (bias unpenalized)
    Xn, mu, sigma = normalize(X)
    Xb = np.c_[np.ones(len(Xn)), Xn].astype(np.float64, copy=False)
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    reg = np.r_[0.0, np.full(Xn.shape[1], l2)]
    t0 = perf_counter()
    for _ in range(epochs):
        lin = np.clip(safe_matmul(Xb, w), SAFE_MIN, SAFE_MAX)
        yhat = 1.0 / (1.0 + np.exp(-lin))
        grad = (1 / len(Xb)) * safe_matmul(Xb.T, (yhat - y)) + reg * w
        w -= lr * grad
    t = perf_counter() - t0
    w_denorm, b_denorm = denorm_coef(w[1:], w[0], mu, sigma)
    return w_denorm, b_denorm, t

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------
    # Linear regression: cold days
    # -----------------------------
    lin_df = pd.read_csv(COLD_CSV)
    if not {"year", "cold_days_below_minus30"}.issubset(lin_df.columns):
        raise ValueError(f"{COLD_CSV} must have columns: 'year', 'cold_days_below_minus30'")
    lin_df = (lin_df[["year", "cold_days_below_minus30"]]
              .apply(pd.to_numeric, errors="coerce")
              .dropna()
              .sort_values("year")
              .reset_index(drop=True))
    X_lin = lin_df[["year"]].to_numpy(dtype=np.float64, copy=False)
    y_lin = lin_df["cold_days_below_minus30"].to_numpy(dtype=np.float64, copy=False)

    w_lin, b_lin, t_lin = gd_linear(X_lin, y_lin)
    yhat_lin = X_lin.ravel() * w_lin[0] + b_lin
    rmse_lin = float(np.sqrt(mean_squared_error(y_lin, yhat_lin)))

    t0 = perf_counter()
    sk_lin = LinearRegression().fit(X_lin, y_lin)
    t_lin_sk = perf_counter() - t0
    rmse_lin_sk = float(np.sqrt(mean_squared_error(y_lin, sk_lin.predict(X_lin))))

    fy = np.array(FUTURE_YEARS, dtype=np.int64)
    future_years_arr = fy.reshape(-1, 1).astype(np.float64)
    pred_ours_lin = np.maximum(0, future_years_arr.ravel() * w_lin[0] + b_lin)
    pred_sk_lin = np.maximum(0, sk_lin.predict(future_years_arr))

    lin_summary = pd.DataFrame({
        "model": ["From-scratch GD", "scikit-learn"],
        "coef": [float(w_lin[0]), float(sk_lin.coef_[0])],
        "intercept": [float(b_lin), float(sk_lin.intercept_)],
        "rmse": [rmse_lin, rmse_lin_sk],
        "train_time_s": [float(t_lin), float(t_lin_sk)],
    })
    pred_years_df = pd.DataFrame({"year": fy,
                                  "pred_days_ours": pred_ours_lin,
                                  "pred_days_sklearn": pred_sk_lin})

    # Plot: historical points + extended fits + future markers
    plt.figure()
    plt.scatter(X_lin.ravel(), y_lin, label="Data")
    x_all = np.concatenate([X_lin.ravel(), future_years_arr.ravel()])
    x_grid = np.linspace(x_all.min(), x_all.max(), 300).reshape(-1, 1)
    plt.plot(x_grid.ravel(), x_grid.ravel() * w_lin[0] + b_lin, label="From-scratch fit")
    plt.plot(x_grid.ravel(), sk_lin.predict(x_grid), label="sklearn fit")
    plt.scatter(FUTURE_YEARS, pred_ours_lin, s=60, label="Predictions")
    for yr, yv in zip(FUTURE_YEARS, pred_ours_lin):
        yv = max(0, yv)
        plt.annotate(f"{int(yr)}: {yv:.2f}", xy=(yr, yv), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Year")
    plt.ylabel("Days below –30°C")
    plt.title("Astana: Cold Days Trend (Linear Regression)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "plot_linear_cold_days.png"), bbox_inches="tight", dpi=160)
    plt.close()

    # -----------------------------------
    # Logistic regression: flight delays
    # -----------------------------------
    log_df = pd.read_csv(FLIGHTS_CSV)

    # Minimal resilience to real datasets
    for col in ["temperature", "snowfall", "wind", "visibility", "delayed"]:
        if col not in log_df.columns:
            log_df[col] = np.nan

    # Heuristic unit normalizations (remove if you are certain about units)
    if log_df["snowfall"].max(skipna=True) and log_df["snowfall"].max(skipna=True) > 50:
        log_df["snowfall"] = log_df["snowfall"] / 10.0      # mm -> cm
    if log_df["wind"].max(skipna=True) and log_df["wind"].max(skipna=True) < 35:
        log_df["wind"] = log_df["wind"] * 3.6               # m/s -> km/h

    for c in ["temperature", "snowfall", "wind", "visibility", "delayed"]:
        log_df[c] = pd.to_numeric(log_df[c], errors="coerce")

    # Auto-proxy label only if labels missing (transparent fallback)
    if log_df["delayed"].isna().all():
        proxy = (
            (log_df["temperature"] <= -25) |
            (log_df["snowfall"].fillna(0) > 0.5) |
            (log_df["wind"].fillna(0) >= 28) |
            (log_df["visibility"].fillna(10) <= 2.0)
        ).astype(int)
        log_df["delayed"] = proxy

    log_df = log_df.dropna(subset=["temperature", "snowfall", "wind", "visibility", "delayed"]).reset_index(drop=True)

    X_log = log_df[["temperature", "snowfall", "wind", "visibility"]].to_numpy(dtype=np.float64, copy=False)
    y_log = log_df["delayed"].astype(int, copy=False).to_numpy()

    w_log, b_log, t_log = gd_logistic(X_log, y_log)
    proba_ours = sigmoid_stable(safe_matmul(X_log, w_log) + b_log)
    pred_ours = (proba_ours >= 0.5).astype(int, copy=False)
    acc_ours = float(accuracy_score(y_log, pred_ours))

    t0 = perf_counter()
    sk_log = LogisticRegression(max_iter=2000).fit(X_log, y_log)
    t_log_sk = perf_counter() - t0
    proba_sk = sk_log.predict_proba(X_log)[:, 1]
    pred_sk = (proba_sk >= 0.5).astype(int, copy=False)
    acc_sk = float(accuracy_score(y_log, pred_sk))

    log_summary = pd.DataFrame({
        "model": ["From-scratch GD (L2=1e-4)", "scikit-learn"],
        "coef_or_weights": [list(np.round(w_log, 6)), list(np.round(sk_log.coef_.ravel(), 6))],
        "intercept": [float(b_log), float(sk_log.intercept_[0])],
        "accuracy": [acc_ours, acc_sk],
        "train_time_s": [float(t_log), float(t_log_sk)]
    })

    # Two demo cases (A/B) as requested
    cases = pd.DataFrame(
        {"temperature": [CASE_A[0], CASE_B[0]],
         "snowfall": [CASE_A[1], CASE_B[1]],
         "wind": [CASE_A[2], CASE_B[2]],
         "visibility": [CASE_A[3], CASE_B[3]]},
        index=["Case A", "Case B"]
    )
    X_cases = cases.to_numpy(dtype=np.float64, copy=False)
    proba_cases_ours = sigmoid_stable(safe_matmul(X_cases, w_log) + b_log)
    proba_cases_sk = sk_log.predict_proba(X_cases)[:, 1]
    pred_cases_ours = (proba_cases_ours >= 0.5).astype(int, copy=False)
    pred_cases_sk = (proba_cases_sk >= 0.5).astype(int, copy=False)
    cases_out = pd.DataFrame({
        "case": cases.index,
        "ours_prob": proba_cases_ours,
        "ours_class": pred_cases_ours,
        "sklearn_prob": proba_cases_sk,
        "sklearn_class": pred_cases_sk
    })

    # Probability vs wind while holding others at dataset means
    means = log_df[["temperature", "snowfall", "visibility"]].mean()
    wind_grid = np.linspace(log_df["wind"].min(), log_df["wind"].max(), 100)
    grid_features = np.column_stack([
        np.full_like(wind_grid, means["temperature"], dtype=float),
        np.full_like(wind_grid, means["snowfall"], dtype=float),
        wind_grid.astype(float),
        np.full_like(wind_grid, means["visibility"], dtype=float),
    ])
    proba_grid_ours = sigmoid_stable(safe_matmul(grid_features, w_log) + b_log)
    proba_grid_sk = sk_log.predict_proba(grid_features)[:, 1]

    plt.figure()
    plt.plot(wind_grid, proba_grid_ours, label="From-scratch")
    plt.plot(wind_grid, proba_grid_sk, label="scikit-learn")
    plt.xlabel("Wind")
    plt.ylabel("Delay probability")
    plt.title("Flight delay probability vs wind (others fixed at mean)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "plot_logistic_delay_vs_wind.png"), bbox_inches="tight", dpi=160)
    plt.close()

    # Outputs
    lin_summary.to_csv(os.path.join(OUT_DIR, "linear_summary.csv"), index=False)
    pred_years_df.to_csv(os.path.join(OUT_DIR, "linear_future_predictions.csv"), index=False)
    log_summary.to_csv(os.path.join(OUT_DIR, "logistic_summary.csv"), index=False)
    cases_out.to_csv(os.path.join(OUT_DIR, "logistic_cases.csv"), index=False)

    with pd.ExcelWriter(os.path.join(OUT_DIR, "astana_reports.xlsx"), engine="xlsxwriter") as xlw:
        lin_summary.to_excel(xlw, sheet_name="linear_summary", index=False)
        pred_years_df.to_excel(xlw, sheet_name="linear_future_predictions", index=False)
        log_summary.to_excel(xlw, sheet_name="logistic_summary", index=False)
        cases_out.to_excel(xlw, sheet_name="logistic_cases", index=False)

    # Minimal markdown summary for quick review
    md = []
    md.append("# Astana Regression Reports\n")
    md.append("## Linear Regression\n")
    md.append(lin_summary.round(6).to_markdown(index=False))
    md.append("\n\n### Future Predictions\n")
    md.append(pred_years_df.round(2).to_markdown(index=False))
    md.append("\n\n![](plot_linear_cold_days.png)\n")
    md.append("\n## Logistic Regression\n")
    md.append(log_summary.to_markdown(index=False))
    md.append("\n\n### Cases A/B\n")
    md.append(cases_out.round({"ours_prob":4, "sklearn_prob":4}).to_markdown(index=False))
    md.append("\n\n![](plot_logistic_delay_vs_wind.png)\n")
    with open(os.path.join(OUT_DIR, "astana_reports.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Console recap
    print("=== Linear Regression – From Scratch ===")
    print(f"coef={w_lin[0]:.6f}  intercept={b_lin:.6f}  rmse={rmse_lin:.6f}  time_s={t_lin:.6f}")
    print("=== Linear Regression – scikit-learn ===")
    print(f"coef={sk_lin.coef_[0]:.6f}  intercept={sk_lin.intercept_:.6f}  rmse={rmse_lin_sk:.6f}  time_s={t_lin_sk:.6f}")
    for yr, a, b in zip(fy, pred_ours_lin, pred_sk_lin):
        print(f"{int(yr)}: ours={a:.2f}, sklearn={b:.2f}")

    print("\n=== Logistic Regression – From Scratch ===")
    print(f"weights={np.round(w_log,6).tolist()}  intercept={b_log:.6f}  acc={acc_ours:.6f}  time_s={t_log:.6f}")
    print("=== Logistic Regression – scikit-learn ===")
    print(f"weights={np.round(sk_log.coef_.ravel(),6).tolist()}  intercept={sk_log.intercept_[0]:.6f}  acc={acc_sk:.6f}  time_s={t_log_sk:.6f}")

    print("\n=== Flight Delay Predictions (Cases) ===")
    for name, p1, c1, p2, c2 in zip(["Case A", "Case B"], proba_cases_ours, pred_cases_ours, proba_cases_sk, pred_cases_sk):
        print(f"{name}: ours_p={p1:.4f} ours_y={int(c1)} | sklearn_p={p2:.4f} sklearn_y={int(c2)}")

    print(f"\nSaved outputs to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
