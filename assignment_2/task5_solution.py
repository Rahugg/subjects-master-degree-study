import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_users=100, n_items=20, density=0.4, seed=42):
    rng = np.random.default_rng(seed)
    R = np.zeros((n_users, n_items), dtype=int)
    mask = rng.random((n_users, n_items)) < density
    R[mask] = rng.integers(1, 6, size=mask.sum())
    return R

def split_observations(R, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    nz = np.argwhere(R > 0)
    rng.shuffle(nz)
    n_test = int(len(nz) * test_ratio)
    test_idx = nz[:n_test]
    train = R.copy().astype(np.float64)
    test = np.zeros_like(train)
    train[tuple(test_idx.T)] = 0
    test[tuple(test_idx.T)] = R[tuple(test_idx.T)]
    return train, test, test_idx

def matrix_factorization(R, k=8, lr=0.01, reg=0.05, epochs=40, clip=0.5, seed=42):
    rng = np.random.default_rng(seed)
    m, n = R.shape
    P = 0.05 * rng.standard_normal((m, k)).astype(np.float64)
    Q = 0.05 * rng.standard_normal((n, k)).astype(np.float64)
    obs = np.argwhere(R > 0)
    history = []
    for _ in range(epochs):
        rng.shuffle(obs)
        for i, j in obs:
            r = R[i, j]
            pred = float(P[i] @ Q[j])
            e = r - pred
            gP = e * Q[j] - reg * P[i]
            gQ = e * P[i] - reg * Q[j]
            updP = np.clip(lr * gP, -clip, clip)
            updQ = np.clip(lr * gQ, -clip, clip)
            P[i] += updP
            Q[j] += updQ
            if not np.isfinite(P[i]).all():
                P[i] = np.nan_to_num(P[i], nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(Q[j]).all():
                Q[j] = np.nan_to_num(Q[j], nan=0.0, posinf=0.0, neginf=0.0)
        pr = (P @ Q.T)[tuple(obs.T)]
        rmse = float(np.sqrt(np.mean((R[tuple(obs.T)] - pr) ** 2)))
        history.append(rmse)
        lr *= 0.98
    return P, Q, history

def cosine_sim_matrix(X):
    X = X.astype(np.float64, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    denom = np.maximum(norms, 1e-12)
    Xn = X / denom
    S = Xn @ Xn.T
    S = np.nan_to_num(S, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.clip(S, -1.0, 1.0)
    return S

def user_cf_rankings(train, k_neighbors=5):
    S = cosine_sim_matrix(train)
    n_users, _ = train.shape
    rankings = []
    for u in range(n_users):
        s = S[u].copy()
        s[u] = -1
        nn = np.argsort(-s)[:k_neighbors]
        taken = train[u] > 0
        scores = train[nn].mean(axis=0).astype(np.float64)
        scores[taken] = -np.inf
        order = np.argsort(-scores)
        rankings.append(order.tolist())
    return rankings

def item_cf_rankings(train):
    X = train.T
    S = cosine_sim_matrix(X)
    n_users, _ = train.shape
    rankings = []
    for u in range(n_users):
        user_vec = train[u].astype(np.float64)
        scores = user_vec @ S
        scores[user_vec > 0] = -np.inf
        order = np.argsort(-scores)
        rankings.append(order.tolist())
    return rankings

def mf_rankings(P, Q, train):
    R_hat = P @ Q.T
    n_users, _ = train.shape
    rankings = []
    for u in range(n_users):
        scores = R_hat[u].copy()
        scores[train[u] > 0] = -np.inf
        order = np.argsort(-scores)
        rankings.append(order.tolist())
    return rankings, R_hat

def precision_recall_curves(R_true, rankings, ks=(1, 3, 5, 10), threshold=4):
    n_users = R_true.shape[0]
    relevant = [set(np.where(R_true[u] >= threshold)[0]) for u in range(n_users)]
    per_user_P = {k: np.full(n_users, np.nan, dtype=np.float64) for k in ks}
    per_user_R = {k: np.full(n_users, np.nan, dtype=np.float64) for k in ks}
    for u in range(n_users):
        rel = relevant[u]
        if len(rel) == 0:
            continue
        rank_u = rankings[u]
        for k in ks:
            topk = rank_u[:k]
            hits = sum(1 for it in topk if it in rel)
            per_user_P[k][u] = hits / k
            per_user_R[k][u] = hits / len(rel)
    macro_P = {k: float(np.nanmean(per_user_P[k])) if np.any(~np.isnan(per_user_P[k])) else 0.0 for k in ks}
    macro_R = {k: float(np.nanmean(per_user_R[k])) if np.any(~np.isnan(per_user_R[k])) else 0.0 for k in ks}
    return per_user_P, per_user_R, macro_P, macro_R

def print_curves(label, macro_P, macro_R):
    ks = sorted(macro_P.keys())
    print(f"{label} Precision/Recall:")
    for k in ks:
        print(f"  @[{k:>2}]  P={macro_P[k]:.3f}  R={macro_R[k]:.3f}")

def plot_history(history):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training RMSE")
    plt.tight_layout()
    plt.show()

def heatmap_Q(Q):
    plt.figure()
    plt.imshow(Q, aspect='auto')
    plt.xlabel("Factors")
    plt.ylabel("Items")
    plt.title("Q Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def top_items_per_factor(Q, topn=3):
    res = {}
    for f in range(Q.shape[1]):
        idx = np.argsort(-np.abs(Q[:, f]))[:topn]
        res[f] = idx.tolist()
    return res

def demo():
    R = generate_data()
    train, test, _ = split_observations(R, 0.2)
    P, Q, history = matrix_factorization(train, k=8, lr=0.02, reg=0.05, epochs=40, clip=0.5)
    u_rank = user_cf_rankings(train, k_neighbors=5)
    i_rank = item_cf_rankings(train)
    m_rank, R_hat = mf_rankings(P, Q, train)
    ks = (1, 3, 5, 10)
    _, _, macroP_user, macroR_user = precision_recall_curves(test, u_rank, ks)
    _, _, macroP_item, macroR_item = precision_recall_curves(test, i_rank, ks)
    _, _, macroP_mf,   macroR_mf   = precision_recall_curves(test, m_rank, ks)
    print_curves("User-CF", macroP_user, macroR_user)
    print_curves("Item-CF", macroP_item, macroR_item)
    print_curves("MF",      macroP_mf,   macroR_mf)
    rng = np.random.default_rng(0)
    users = rng.choice(R.shape[0], size=3, replace=False)
    for u in users:
        print("User", int(u),
              "MF:", m_rank[u][:5],
              "UserCF:", u_rank[u][:5],
              "ItemCF:", i_rank[u][:5])
    obs = np.argwhere(test > 0)
    if len(obs) > 0:
        preds = R_hat[tuple(obs.T)]
        rmse = float(np.sqrt(np.mean((test[tuple(obs.T)] - preds) ** 2)))
    else:
        rmse = 0.0
    print("Test RMSE (MF):", round(rmse, 4))
    print("Top items per factor:", top_items_per_factor(Q, topn=3))
    plot_history(history)
    heatmap_Q(Q)

if __name__ == "__main__":
    demo()
