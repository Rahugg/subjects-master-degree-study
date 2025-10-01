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

OUT_DIR     = "./reports_out"
FLIGHT_CSV  = "flight_delays_from_pdf.csv"
TRAFFIC_CSV = "traffic_congestion_astana.csv"
COLD_CSV    = "cold_days_from_pdf.csv"   # columns: year,cold_days_below_minus30

def fit_linear_from_scratch_normal_eq(X_with_bias: np.ndarray, y: np.ndarray):
    t0 = perf_counter()
    theta, *_ = np.linalg.lstsq(X_with_bias, y, rcond=None)
    train_time_s = perf_counter() - t0
    bias = float(theta[0])
    weights = theta[1:].astype(float)
    return weights, bias, train_time_s

def _sigmoid_stable(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def fit_logistic_from_scratch_gd_l2(
    X_with_bias: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 8000,
    l2: float = 1e-4,
):
    w_all = np.zeros(X_with_bias.shape[1])
    reg = np.r_[0.0, np.full(X_with_bias.shape[1] - 1, l2)]  # no penalty on bias
    t0 = perf_counter()
    for _ in range(epochs):
        p = _sigmoid_stable(X_with_bias @ w_all)
        grad = (X_with_bias.T @ (p - y)) / len(y) + reg * w_all
        w_all -= lr * grad
    train_time_s = perf_counter() - t0
    bias = float(w_all[0])
    weights = w_all[1:].copy()
    return weights, bias, train_time_s

def fit_svm_linear_from_scratch_sgd(
    X: np.ndarray, y_binary01: np.ndarray, C: float = 1.0, epochs: int = 60, lr0: float = 0.2
):
    y_pm1 = np.where(y_binary01 == 1, 1.0, -1.0)
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    t0 = perf_counter()
    for epoch in range(epochs):
        lr = lr0 / (1.0 + 0.1 * epoch)
        for i in range(len(X)):
            margin = y_pm1[i] * (np.dot(w, X[i]) + b)
            if margin >= 1:
                w -= lr * (w / (C * len(X)))  # only L2 shrink
            else:
                w -= lr * (w / (C * len(X)) - y_pm1[i] * X[i])
                b += lr * y_pm1[i]
    train_time_s = perf_counter() - t0
    return w, b, train_time_s

def predict_proba_svm_linear_via_sigmoid(X: np.ndarray, w: np.ndarray, b: float):
    """Quick non-calibrated probability proxy via sigmoid(w^T x + b)."""
    return _sigmoid_stable(X @ w + b)


class TreeNode:
    __slots__ = ("feat", "thr", "left", "right", "pred", "depth")
    def __init__(self, pred=None, feat=None, thr=None, left=None, right=None, depth=0):
        self.pred = pred
        self.feat = feat
        self.thr = thr
        self.left = left
        self.right = right
        self.depth = depth

def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    p = np.mean(y == 1)
    return 2 * p * (1 - p)

def tree_best_split(X, y):
    n, d = X.shape
    best_feat, best_thr, best_gain = None, None, -1e9
    for j in range(d):
        for thr in np.unique(X[:, j]):
            left = y[X[:, j] <= thr]
            right = y[X[:, j] > thr]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = - (len(left)/n) * gini_impurity(left) - (len(right)/n) * gini_impurity(right)
            if gain > best_gain:
                best_feat, best_thr, best_gain = j, thr, gain
    return best_feat, best_thr

def tree_build(X, y, depth=0, max_depth=3, min_leaf=5):
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2 * min_leaf:
        return TreeNode(pred=int(np.round(np.mean(y))), depth=depth)
    j, thr = tree_best_split(X, y)
    if j is None:
        return TreeNode(pred=int(np.round(np.mean(y))), depth=depth)
    left_idx = X[:, j] <= thr
    right_idx = ~left_idx
    if left_idx.sum() < min_leaf or right_idx.sum() < min_leaf:
        return TreeNode(pred=int(np.round(np.mean(y))), depth=depth)
    left = tree_build(X[left_idx], y[left_idx], depth + 1, max_depth, min_leaf)
    right = tree_build(X[right_idx], y[right_idx], depth + 1, max_depth, min_leaf)
    return TreeNode(feat=j, thr=thr, left=left, right=right, depth=depth)

def tree_predict_row(x, node: TreeNode):
    while node.left is not None and node.right is not None:
        node = node.left if x[node.feat] <= node.thr else node.right
    return node.pred

def tree_predict(X, root: TreeNode):
    return np.array([tree_predict_row(x, root) for x in X], dtype=int)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(COLD_CSV):
        cold_df = pd.read_csv(COLD_CSV)
    else:
        cold_df = pd.DataFrame(columns=["year", "cold_days_below_minus30"])

    if not cold_df.empty and {"year", "cold_days_below_minus30"}.issubset(cold_df.columns):
        lin_df = (
            cold_df[["year", "cold_days_below_minus30"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .sort_values("year")
        )
        if not lin_df.empty:
            lin_df = lin_df[lin_df["year"] >= 2010]
    else:
        lin_df = pd.DataFrame()

    if not lin_df.empty:
        X_lin = lin_df[["year"]].to_numpy(float)
        y_lin = lin_df["cold_days_below_minus30"].to_numpy(float)

        # ==== FROM SCRATCH ====
        X_lin_with_bias = np.c_[np.ones(len(X_lin)), X_lin]
        lin_w_scratch, lin_b_scratch, lin_time_scratch = fit_linear_from_scratch_normal_eq(
            X_lin_with_bias, y_lin
        )
        lin_preds_train_scratch = (X_lin_with_bias @ np.r_[lin_b_scratch, lin_w_scratch]).ravel()
        rmse_lin_scratch = float(np.sqrt(mean_squared_error(y_lin, lin_preds_train_scratch)))

        # ==== SCIKIT-LEARN ====
        t0 = perf_counter()
        lin_model_sklearn = LinearRegression().fit(X_lin, y_lin)
        lin_time_sklearn = perf_counter() - t0
        rmse_lin_sklearn = float(np.sqrt(mean_squared_error(y_lin, lin_model_sklearn.predict(X_lin))))

        # Forecasts
        future_years = np.array([2025, 2030, 2035], dtype=float).reshape(-1, 1)
        preds_lin_scratch = np.maximum(
            0,
            (np.c_[np.ones(len(future_years)), future_years] @ np.r_[lin_b_scratch, lin_w_scratch]).ravel(),
        )
        preds_lin_sklearn = np.maximum(0, lin_model_sklearn.predict(future_years))

        # Outputs
        pd.DataFrame({
            "model": ["Linear-FromScratch(NormalEq)", "Linear-ScikitLearn"],
            "coef": [float(lin_w_scratch[0]), float(lin_model_sklearn.coef_[0])],
            "intercept": [float(lin_b_scratch), float(lin_model_sklearn.intercept_)],
            "rmse": [rmse_lin_scratch, rmse_lin_sklearn],
            "train_time_s": [lin_time_scratch, lin_time_sklearn],
        }).to_csv(os.path.join(OUT_DIR, "linear_summary.csv"), index=False)

        pd.DataFrame({
            "year": [2025, 2030, 2035],
            "pred_days_from_scratch": preds_lin_scratch,
            "pred_days_sklearn": preds_lin_sklearn,
        }).to_csv(os.path.join(OUT_DIR, "linear_future_predictions.csv"), index=False)

        plt.figure()
        plt.scatter(X_lin.ravel(), y_lin, label="Data (>=2010)")
        x_all = np.concatenate([X_lin.ravel(), future_years.ravel()])
        xg = np.linspace(x_all.min(), x_all.max(), 200).reshape(-1, 1)
        line_scratch = (np.c_[np.ones(len(xg)), xg] @ np.r_[lin_b_scratch, lin_w_scratch]).ravel()
        line_sklearn = lin_model_sklearn.predict(xg)
        plt.plot(xg.ravel(), line_scratch, label="Linear-FromScratch")
        plt.plot(xg.ravel(), line_sklearn, label="Linear-ScikitLearn")
        plt.scatter([2025, 2030, 2035], preds_lin_scratch, s=60, label="Forecasts (scratch)")
        for yr, yv in zip([2025, 2030, 2035], preds_lin_scratch):
            plt.annotate(f"{int(yr)}: {yv:.2f}", xy=(yr, yv), xytext=(5, 5), textcoords="offset points")
        plt.xlabel("Year"); plt.ylabel("Days < −30°C")
        plt.title("Cold days (trained on ≥ 2010)")
        plt.legend()
        plt.savefig(os.path.join(OUT_DIR, "linear_trend.png"), bbox_inches="tight", dpi=160)
        plt.close()
    else:
        with open(os.path.join(OUT_DIR, "linear_INFO.txt"), "w") as f:
            f.write("No usable rows in cold_days_from_pdf.csv; linear task skipped.\n")

    log_df = pd.read_csv(FLIGHT_CSV)
    X_log = log_df[["temperature", "snowfall", "wind", "visibility"]].to_numpy(float)
    y_log = log_df["delayed"].to_numpy(int)

    # ==== FROM SCRATCH ====
    X_log_with_bias = np.c_[np.ones(len(X_log)), X_log]
    log_w_scratch, log_b_scratch, log_time_scratch = fit_logistic_from_scratch_gd_l2(
        X_log_with_bias, y_log, lr=0.1, epochs=8000, l2=1e-4
    )
    probs_log_scratch = _sigmoid_stable(X_log_with_bias @ np.r_[log_b_scratch, log_w_scratch])
    acc_log_scratch = accuracy_score(y_log, (probs_log_scratch >= 0.5).astype(int))

    # ==== SCIKIT-LEARN ====
    t0 = perf_counter()
    log_model_sklearn = LogisticRegression(max_iter=5000).fit(X_log, y_log)
    log_time_sklearn = perf_counter() - t0
    probs_log_sklearn = log_model_sklearn.predict_proba(X_log)[:, 1]
    acc_log_sklearn = accuracy_score(y_log, (probs_log_sklearn >= 0.5).astype(int))

    # Cases A/B
    caseA = np.array([[-28, 1, 9, 2]], dtype=float)
    caseB = np.array([[-12, 0, 3, 10]], dtype=float)

    probA_scratch = _sigmoid_stable(np.c_[np.ones((1, 1)), caseA] @ np.r_[log_b_scratch, log_w_scratch]).item()
    probB_scratch = _sigmoid_stable(np.c_[np.ones((1, 1)), caseB] @ np.r_[log_b_scratch, log_w_scratch]).item()

    probA_sklearn = log_model_sklearn.predict_proba(caseA)[0, 1].item()
    probB_sklearn = log_model_sklearn.predict_proba(caseB)[0, 1].item()

    # Outputs
    pd.DataFrame({
        "model": ["Logistic-FromScratch(GD,L2=1e-4)", "Logistic-ScikitLearn"],
        "accuracy": [acc_log_scratch, acc_log_sklearn],
        "train_time_s": [log_time_scratch, log_time_sklearn],
        "weights_or_coef": [
            np.round(log_w_scratch, 6).tolist(),
            np.round(log_model_sklearn.coef_.ravel(), 6).tolist(),
        ],
        "intercept": [log_b_scratch, float(log_model_sklearn.intercept_[0])],
    }).to_csv(os.path.join(OUT_DIR, "logistic_summary.csv"), index=False)

    pd.DataFrame({
        "case": ["A(-28,1,9,2)", "B(-12,0,3,10)"],
        "prob_from_scratch": [probA_scratch, probB_scratch],
        "class_from_scratch": [int(probA_scratch >= 0.5), int(probB_scratch >= 0.5)],
        "prob_sklearn": [probA_sklearn, probB_sklearn],
        "class_sklearn": [int(probA_sklearn >= 0.5), int(probB_sklearn >= 0.5)],
    }).to_csv(os.path.join(OUT_DIR, "logistic_cases.csv"), index=False)

    # Optional plot: probability vs wind (others fixed at means)
    means = log_df[["temperature", "snowfall", "visibility"]].mean()
    wind_grid = np.linspace(log_df["wind"].min(), log_df["wind"].max(), 120)
    grid = np.column_stack([
        np.full_like(wind_grid, means["temperature"], dtype=float),
        np.full_like(wind_grid, round(log_df["snowfall"].mean()), dtype=float),
        wind_grid.astype(float),
        np.full_like(wind_grid, means["visibility"], dtype=float),
    ])
    curve_scratch = _sigmoid_stable(np.c_[np.ones(len(grid)), grid] @ np.r_[log_b_scratch, log_w_scratch])
    curve_sklearn = log_model_sklearn.predict_proba(grid)[:, 1]
    plt.figure()
    plt.plot(wind_grid, curve_scratch, label="Logistic-FromScratch")
    plt.plot(wind_grid, curve_sklearn, label="Logistic-ScikitLearn")
    plt.xlabel("Wind (m/s)"); plt.ylabel("Delay probability")
    plt.title("Delay probability vs Wind (demo table)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "logistic_prob_vs_wind.png"), bbox_inches="tight", dpi=160)
    plt.close()

    traffic_df = pd.read_csv(TRAFFIC_CSV)
    traffic_features = ["hour", "day_of_week", "temperature", "precipitation", "event"]
    X_tr = traffic_df[traffic_features].to_numpy(float)
    y_tr = traffic_df["congestion"].to_numpy(int)

    # ==== SVM — FROM SCRATCH (LINEAR) ====
    svm_w_scratch, svm_b_scratch, svm_time_scratch = fit_svm_linear_from_scratch_sgd(
        X_tr, y_tr, C=1.0, epochs=60, lr0=0.2
    )
    svm_preds_scratch = (X_tr @ svm_w_scratch + svm_b_scratch >= 0).astype(int)
    svm_acc_scratch = accuracy_score(y_tr, svm_preds_scratch)
    svm_prec_scratch = precision_score(y_tr, svm_preds_scratch)
    svm_rec_scratch = recall_score(y_tr, svm_preds_scratch)

    # ==== DECISION TREE — FROM SCRATCH ====
    t0 = perf_counter()
    tree_root_scratch = tree_build(X_tr, y_tr, max_depth=3, min_leaf=5)
    dt_time_scratch = perf_counter() - t0
    dt_preds_scratch = tree_predict(X_tr, tree_root_scratch)
    dt_acc_scratch = accuracy_score(y_tr, dt_preds_scratch)
    dt_prec_scratch = precision_score(y_tr, dt_preds_scratch)
    dt_rec_scratch = recall_score(y_tr, dt_preds_scratch)

    # ==== SCIKIT-LEARN BASELINES ====
    t0 = perf_counter()
    svc_model_sklearn = SVC(probability=True).fit(X_tr, y_tr)
    svm_time_sklearn = perf_counter() - t0

    t0 = perf_counter()
    dtc_model_sklearn = DecisionTreeClassifier(max_depth=3).fit(X_tr, y_tr)
    dt_time_sklearn = perf_counter() - t0

    yhat_svc = svc_model_sklearn.predict(X_tr)
    yhat_dtc = dtc_model_sklearn.predict(X_tr)
    svc_metrics = dict(
        acc=accuracy_score(y_tr, yhat_svc),
        prec=precision_score(y_tr, yhat_svc),
        rec=recall_score(y_tr, yhat_svc),
    )
    dtc_metrics = dict(
        acc=accuracy_score(y_tr, yhat_dtc),
        prec=precision_score(y_tr, yhat_dtc),
        rec=recall_score(y_tr, yhat_dtc),
    )

    # Outputs
    pd.DataFrame([
        {"model": "SVM-FromScratch(Linear)", "acc": svm_acc_scratch, "prec": svm_prec_scratch, "rec": svm_rec_scratch, "train_time_s": svm_time_scratch},
        {"model": "DecisionTree-FromScratch(max_depth=3)", "acc": dt_acc_scratch, "prec": dt_prec_scratch, "rec": dt_rec_scratch, "train_time_s": dt_time_scratch},
        {"model": "SVC-ScikitLearn", "acc": svc_metrics["acc"], "prec": svc_metrics["prec"], "rec": svc_metrics["rec"], "train_time_s": svm_time_sklearn},
        {"model": "DecisionTree-ScikitLearn(max_depth=3)", "acc": dtc_metrics["acc"], "prec": dtc_metrics["prec"], "rec": dtc_metrics["rec"], "train_time_s": dt_time_sklearn},
    ]).to_csv(os.path.join(OUT_DIR, "traffic_summary.csv"), index=False)

    # Cases A/B for all models
    caseA = np.array([[8, 2, -15, 12, 1]], dtype=float)
    caseB = np.array([[14, 5, 25, 0, 0]], dtype=float)
    cases_stack = np.vstack([caseA, caseB])

    out_cases = pd.DataFrame({
        "case": ["A", "B"],
        "svm_from_scratch_prob": predict_proba_svm_linear_via_sigmoid(cases_stack, svm_w_scratch, svm_b_scratch),
        "svm_from_scratch_pred": [(caseA @ svm_w_scratch + svm_b_scratch >= 0).astype(int)[0],
                                  (caseB @ svm_w_scratch + svm_b_scratch >= 0).astype(int)[0]],
        "svc_sklearn_prob": svc_model_sklearn.predict_proba(cases_stack)[:, 1],
        "svc_sklearn_pred": svc_model_sklearn.predict(cases_stack),
        "dt_from_scratch_pred": [tree_predict(caseA, tree_root_scratch)[0], tree_predict(caseB, tree_root_scratch)[0]],
        "dt_sklearn_pred": dtc_model_sklearn.predict(cases_stack),
    })
    out_cases.to_csv(os.path.join(OUT_DIR, "traffic_cases.csv"), index=False)

    # Optional: SVC decision boundary over (hour, precipitation)
    try:
        hmin, hmax = X_tr[:, 0].min() - 1, X_tr[:, 0].max() + 1
        pmin, pmax = X_tr[:, 3].min() - 1, X_tr[:, 3].max() + 1
        xx, yy = np.meshgrid(np.linspace(hmin, hmax, 200), np.linspace(pmin, pmax, 200))
        means = traffic_df[traffic_features].mean()
        gridX = np.column_stack([
            xx.ravel(),
            np.full(xx.size, means["day_of_week"]),
            np.full(xx.size, means["temperature"]),
            yy.ravel(),
            np.full(xx.size, round(means["event"])),
        ])
        zz = svc_model_sklearn.predict(gridX).reshape(xx.shape)
        plt.figure()
        plt.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5])
        plt.scatter(X_tr[:, 0], X_tr[:, 3], c=y_tr, s=20, edgecolors="k")
        plt.xlabel("hour"); plt.ylabel("precipitation")
        plt.title("Traffic SVC decision boundary (hour vs precipitation)")
        plt.savefig(os.path.join(OUT_DIR, "traffic_decision_boundary.png"), bbox_inches="tight", dpi=160)
        plt.close()
    except Exception:
        pass

    print("Done. Outputs in:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
