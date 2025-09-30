#!/usr/bin/env python3
# Runs all assignments by LOADING pre-extracted CSVs (no PDF parsing).
# Place these CSVs next to this script:
#   - flight_delays_from_pdf.csv
#   - traffic_congestion_astana.csv
#   - cold_days_from_pdf.csv   (may be header-only if you don't have the numbers)

import os
import warnings
from time import perf_counter

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

OUT_DIR = "./reports_out"
FLIGHT_CSV  = "flight_delays_from_pdf.csv"
TRAFFIC_CSV = "traffic_congestion_astana.csv"
COLD_CSV    = "cold_days_from_pdf.csv"   # columns: year,cold_days_below_minus30

# -------- Linear regression (Normal Equation) --------
def linear_normal_eq(X: np.ndarray, y: np.ndarray):
    t0 = perf_counter()
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    t = perf_counter() - t0
    b = float(theta[0]); w = theta[1:].astype(float)
    return w, b, t

# -------- Logistic regression (from scratch, GD + L2 on weights) --------
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def logistic_gd(Xb: np.ndarray, y: np.ndarray, lr=0.1, epochs=8000, l2=1e-4):
    w = np.zeros(Xb.shape[1])
    reg = np.r_[0.0, np.full(Xb.shape[1]-1, l2)]  # no penalty on bias
    t0 = perf_counter()
    for _ in range(epochs):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / len(y) + reg * w
        w -= lr * grad
    t = perf_counter() - t0
    b = float(w[0]); weights = w[1:].copy()
    return weights, b, t

# -------- SVM (from scratch, linear, hinge loss, SGD) --------
def svm_linear_sgd(X: np.ndarray, y: np.ndarray, C=1.0, epochs=60, lr0=0.2):
    yb = np.where(y==1, 1.0, -1.0)
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    t0 = perf_counter()
    for epoch in range(epochs):
        lr = lr0 / (1.0 + 0.1*epoch)
        for i in range(len(X)):
            margin = yb[i]*(np.dot(w, X[i]) + b)
            if margin >= 1:
                w -= lr * (w / (C*len(X)))
            else:
                w -= lr * (w / (C*len(X)) - yb[i]*X[i])
                b += lr * yb[i]
    t = perf_counter() - t0
    return w, b, t

def svm_predict_proba_linear(X: np.ndarray, w: np.ndarray, b: float):
    return sigmoid(X @ w + b)  # quick calibration

# -------- Decision Tree (from scratch, binary splits, max_depth=3) --------
class DTNode:
    __slots__ = ("feat","thr","left","right","pred","depth")
    def __init__(self, pred=None, feat=None, thr=None, left=None, right=None, depth=0):
        self.pred = pred; self.feat = feat; self.thr = thr
        self.left = left; self.right = right; self.depth = depth

def gini_impurity(y):
    if len(y)==0: return 0.0
    p = np.mean(y==1)
    return 2*p*(1-p)

def best_split(X, y):
    n, d = X.shape
    best = (None, None, -1e9)
    for j in range(d):
        vals = np.unique(X[:, j])
        for thr in vals:
            left = y[X[:, j] <= thr]
            right = y[X[:, j] > thr]
            if len(left)==0 or len(right)==0: continue
            gain = - (len(left)/n)*gini_impurity(left) - (len(right)/n)*gini_impurity(right)
            if gain > best[2]:
                best = (j, thr, gain)
    return best[0], best[1]

def build_tree(X, y, depth=0, max_depth=3, min_leaf=5):
    if depth>=max_depth or len(np.unique(y))==1 or len(y)<2*min_leaf:
        return DTNode(pred=int(np.round(np.mean(y))), depth=depth)
    j, thr = best_split(X, y)
    if j is None:
        return DTNode(pred=int(np.round(np.mean(y))), depth=depth)
    left_idx = X[:, j] <= thr
    right_idx = ~left_idx
    if left_idx.sum()<min_leaf or right_idx.sum()<min_leaf:
        return DTNode(pred=int(np.round(np.mean(y))), depth=depth)
    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth, min_leaf)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth, min_leaf)
    return DTNode(feat=j, thr=thr, left=left, right=right, depth=depth)

def tree_predict_one(x, node: DTNode):
    while node.left is not None and node.right is not None:
        node = node.left if x[node.feat] <= node.thr else node.right
    return node.pred

def tree_predict(X, root: DTNode):
    return np.array([tree_predict_one(x, root) for x in X], dtype=int)

# -------- Main --------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # =========================
    # Assignment 1: Linear Task
    # =========================
    if os.path.exists(COLD_CSV):
        cold_df = pd.read_csv(COLD_CSV)
    else:
        cold_df = pd.DataFrame(columns=["year","cold_days_below_minus30"])

    if not cold_df.empty and {"year","cold_days_below_minus30"}.issubset(cold_df.columns):
        lin_df = (cold_df[["year","cold_days_below_minus30"]]
                  .apply(pd.to_numeric, errors="coerce")
                  .dropna()
                  .sort_values("year"))
        if not lin_df.empty:
            lin_df = lin_df[lin_df["year"] >= 2010]
    else:
        lin_df = pd.DataFrame()

    if not lin_df.empty:
        X_lin = lin_df[["year"]].to_numpy(float)
        y_lin = lin_df["cold_days_below_minus30"].to_numpy(float)

        Xb = np.c_[np.ones(len(X_lin)), X_lin]
        w_lin, b_lin, t_lin_ours = linear_normal_eq(Xb, y_lin)
        yhat_ours = (Xb @ np.r_[b_lin, w_lin]).ravel()
        rmse_ours = float(np.sqrt(mean_squared_error(y_lin, yhat_ours)))

        t0 = perf_counter()
        sk_lin = LinearRegression().fit(X_lin, y_lin)
        t_lin_sk = perf_counter() - t0
        rmse_sk = float(np.sqrt(mean_squared_error(y_lin, sk_lin.predict(X_lin))))

        fy = np.array([2025, 2030, 2035], dtype=float).reshape(-1,1)
        pred_ours = np.maximum(0, (np.c_[np.ones(len(fy)), fy] @ np.r_[b_lin, w_lin]).ravel())
        pred_sk = np.maximum(0, sk_lin.predict(fy))

        pd.DataFrame({
            "model": ["NormalEq (ours)", "scikit-learn"],
            "coef": [float(w_lin[0]), float(sk_lin.coef_[0])],
            "intercept": [float(b_lin), float(sk_lin.intercept_)],
            "rmse": [rmse_ours, rmse_sk],
            "train_time_s": [t_lin_ours, t_lin_sk]
        }).to_csv(os.path.join(OUT_DIR, "linear_summary.csv"), index=False)

        pd.DataFrame({
            "year": [2025, 2030, 2035],
            "pred_days_ours": pred_ours,
            "pred_days_sklearn": pred_sk
        }).to_csv(os.path.join(OUT_DIR, "linear_future_predictions.csv"), index=False)

        # Plot
        plt.figure()
        plt.scatter(X_lin.ravel(), y_lin, label="Data (>=2010)")
        x_all = np.concatenate([X_lin.ravel(), fy.ravel()])
        xg = np.linspace(x_all.min(), x_all.max(), 200).reshape(-1,1)
        ours_line = (np.c_[np.ones(len(xg)), xg] @ np.r_[b_lin, w_lin]).ravel()
        sk_line = sk_lin.predict(xg)
        plt.plot(xg.ravel(), ours_line, label="NormalEq (ours)")
        plt.plot(xg.ravel(), sk_line, label="sklearn")
        plt.scatter([2025,2030,2035], pred_ours, s=60, label="Predictions")
        for yr, yv in zip([2025,2030,2035], pred_ours):
            plt.annotate(f"{int(yr)}: {yv:.2f}", xy=(yr, yv), xytext=(5,5), textcoords="offset points")
        plt.xlabel("Year"); plt.ylabel("Days < −30°C")
        plt.title("Cold days (trained on ≥ 2010)")
        plt.legend()
        plt.savefig(os.path.join(OUT_DIR, "linear_trend.png"), bbox_inches="tight", dpi=160)
        plt.close()
    else:
        with open(os.path.join(OUT_DIR, "linear_INFO.txt"), "w") as f:
            f.write("No usable rows in cold_days_from_pdf.csv; linear task skipped.\n")

    # ==================================
    # Assignment 2: Logistic (7-row demo)
    # ==================================
    log_df = pd.read_csv(FLIGHT_CSV)
    X_log = log_df[["temperature","snowfall","wind","visibility"]].to_numpy(float)
    y_log = log_df["delayed"].to_numpy(int)

    Xb = np.c_[np.ones(len(X_log)), X_log]
    w_log, b_log, t_log_ours = logistic_gd(Xb, y_log, lr=0.1, epochs=8000, l2=1e-4)
    p_ours = sigmoid(Xb @ np.r_[b_log, w_log])
    acc_ours = accuracy_score(y_log, (p_ours>=0.5).astype(int))

    t0 = perf_counter()
    sk_lr = LogisticRegression(max_iter=5000).fit(X_log, y_log)
    t_log_sk = perf_counter() - t0
    p_sk = sk_lr.predict_proba(X_log)[:,1]
    acc_sk = accuracy_score(y_log, (p_sk>=0.5).astype(int))

    caseA = np.array([[-28, 1, 9, 2]], dtype=float)
    caseB = np.array([[-12, 0, 3, 10]], dtype=float)

    # From-scratch model
    pA_ours = sigmoid(np.c_[np.ones((1, 1)), caseA] @ np.r_[b_log, w_log]).item()
    pB_ours = sigmoid(np.c_[np.ones((1, 1)), caseB] @ np.r_[b_log, w_log]).item()

    # scikit-learn
    pA_sk = sk_lr.predict_proba(caseA)[0, 1].item()
    pB_sk = sk_lr.predict_proba(caseB)[0, 1].item()

    pd.DataFrame({
        "model": ["From-scratch GD (L2=1e-4)", "scikit-learn"],
        "accuracy": [acc_ours, acc_sk],
        "train_time_s": [t_log_ours, t_log_sk],
        "weights_or_coef": [np.round(w_log,6).tolist(), np.round(sk_lr.coef_.ravel(),6).tolist()],
        "intercept": [b_log, float(sk_lr.intercept_[0])]
    }).to_csv(os.path.join(OUT_DIR, "logistic_summary.csv"), index=False)

    pd.DataFrame({
        "case": ["A(-28,1,9,2)","B(-12,0,3,10)"],
        "ours_prob": [pA_ours, pB_ours],
        "ours_class": [int(pA_ours>=0.5), int(pB_ours>=0.5)],
        "sk_prob": [pA_sk, pB_sk],
        "sk_class": [int(pA_sk>=0.5), int(pB_sk>=0.5)]
    }).to_csv(os.path.join(OUT_DIR, "logistic_cases.csv"), index=False)

    # Optional plot: prob vs wind (others fixed at mean of demo table)
    means = log_df[["temperature","snowfall","visibility"]].mean()
    wind_grid = np.linspace(log_df["wind"].min(), log_df["wind"].max(), 120)
    grid = np.column_stack([
        np.full_like(wind_grid, means["temperature"], dtype=float),
        np.full_like(wind_grid, round(log_df["snowfall"].mean()), dtype=float),
        wind_grid.astype(float),
        np.full_like(wind_grid, means["visibility"], dtype=float),
    ])
    ours_curve = sigmoid(np.c_[np.ones(len(grid)), grid] @ np.r_[b_log, w_log])
    sk_curve = sk_lr.predict_proba(grid)[:,1]
    plt.figure()
    plt.plot(wind_grid, ours_curve, label="From-scratch")
    plt.plot(wind_grid, sk_curve, label="sklearn")
    plt.xlabel("Wind (m/s)"); plt.ylabel("Delay probability")
    plt.title("Delay probability vs Wind (demo table)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "logistic_prob_vs_wind.png"), bbox_inches="tight", dpi=160)
    plt.close()

    # ============================================
    # Assignment 3: Traffic Congestion (SVM & DT)
    # ============================================
    traffic_df = pd.read_csv(TRAFFIC_CSV)
    feats = ["hour","day_of_week","temperature","precipitation","event"]
    X_tr = traffic_df[feats].to_numpy(float)
    y_tr = traffic_df["congestion"].to_numpy(int)

    # SVM (scratch)
    w_svm, b_svm, t_svm_ours = svm_linear_sgd(X_tr, y_tr, C=1.0, epochs=60, lr0=0.2)
    yhat_svm = (X_tr @ w_svm + b_svm >= 0).astype(int)
    acc_svm = accuracy_score(y_tr, yhat_svm)
    prec_svm = precision_score(y_tr, yhat_svm)
    rec_svm  = recall_score(y_tr, yhat_svm)

    # Decision Tree (scratch)
    t0 = perf_counter()
    tree_root = build_tree(X_tr, y_tr, max_depth=3, min_leaf=5)
    t_dt_ours = perf_counter() - t0
    yhat_dt = tree_predict(X_tr, tree_root)
    acc_dt = accuracy_score(y_tr, yhat_dt)
    prec_dt = precision_score(y_tr, yhat_dt)
    rec_dt  = recall_score(y_tr, yhat_dt)

    # sklearn baselines
    t0 = perf_counter(); svc = SVC(probability=True).fit(X_tr, y_tr); t_svm_sk = perf_counter()-t0
    t0 = perf_counter(); dtc = DecisionTreeClassifier(max_depth=3).fit(X_tr, y_tr); t_dt_sk = perf_counter()-t0
    yhat_svc = svc.predict(X_tr); yhat_dtc = dtc.predict(X_tr)
    svc_metrics = dict(acc=accuracy_score(y_tr,yhat_svc), prec=precision_score(y_tr,yhat_svc), rec=recall_score(y_tr,yhat_svc))
    dtc_metrics = dict(acc=accuracy_score(y_tr,yhat_dtc), prec=precision_score(y_tr,yhat_dtc), rec=recall_score(y_tr,yhat_dtc))

    pd.DataFrame([
        {"model":"SVM-from-scratch", "acc":acc_svm, "prec":prec_svm, "rec":rec_svm, "train_time_s":t_svm_ours},
        {"model":"DT-from-scratch (max_depth=3)", "acc":acc_dt, "prec":prec_dt, "rec":rec_dt, "train_time_s":t_dt_ours},
        {"model":"SVC(sklearn)", "acc":svc_metrics["acc"], "prec":svc_metrics["prec"], "rec":svc_metrics["rec"], "train_time_s":t_svm_sk},
        {"model":"DecisionTree(max_depth=3)", "acc":dtc_metrics["acc"], "prec":dtc_metrics["prec"], "rec":dtc_metrics["rec"], "train_time_s":t_dt_sk},
    ]).to_csv(os.path.join(OUT_DIR, "traffic_summary.csv"), index=False)

    # Case predictions
    caseA = np.array([[8,2,-15,12,1]], dtype=float)
    caseB = np.array([[14,5,25,0,0]], dtype=float)
    out_cases = pd.DataFrame({
        "case": ["A","B"],
        "svm_scratch_prob": svm_predict_proba_linear(np.vstack([caseA,caseB]), w_svm, b_svm),
        "svm_scratch_pred": [(caseA @ w_svm + b_svm >= 0).astype(int)[0], (caseB @ w_svm + b_svm >= 0).astype(int)[0]],
        "svc_prob": svc.predict_proba(np.vstack([caseA,caseB]))[:,1],
        "svc_pred": svc.predict(np.vstack([caseA,caseB])),
        "dt_scratch_pred": [tree_predict(caseA, tree_root)[0], tree_predict(caseB, tree_root)[0]],
        "dt_sklearn_pred": dtc.predict(np.vstack([caseA,caseB]))
    })
    out_cases.to_csv(os.path.join(OUT_DIR, "traffic_cases.csv"), index=False)

    # Optional decision boundary (SVC) over (hour, precipitation)
    try:
        hmin,hmax = X_tr[:,0].min()-1, X_tr[:,0].max()+1
        pmin,pmax = X_tr[:,3].min()-1, X_tr[:,3].max()+1
        xx, yy = np.meshgrid(np.linspace(hmin,hmax,200), np.linspace(pmin,pmax,200))
        means = traffic_df[feats].mean()
        gridX = np.column_stack([xx.ravel(), np.full(xx.size, means["day_of_week"]),
                                 np.full(xx.size, means["temperature"]),
                                 yy.ravel(),
                                 np.full(xx.size, round(means["event"]))])
        zz = svc.predict(gridX).reshape(xx.shape)
        plt.figure()
        plt.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5,0.5,1.5])
        plt.scatter(X_tr[:,0], X_tr[:,3], c=y_tr, s=20, edgecolors="k")
        plt.xlabel("hour"); plt.ylabel("precipitation")
        plt.title("Traffic SVC decision boundary (hour vs precipitation)")
        plt.savefig(os.path.join(OUT_DIR, "traffic_decision_boundary.png"), bbox_inches="tight", dpi=160)
        plt.close()
    except Exception:
        pass

    print("Done. Outputs in:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()