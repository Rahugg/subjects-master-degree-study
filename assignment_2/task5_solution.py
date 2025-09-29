
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

def user_based_recommend(R, user, k=5, topn=5):
    S = cosine_sim_matrix(R)
    s = S[user].copy()
    s[user] = -1
    nn = np.argsort(-s)[:k]
    taken = R[user] > 0
    scores = R[nn].mean(axis=0)
    scores[taken] = -1
    idx = np.argsort(-scores)[:topn]
    return idx

def item_based_recommend(R, user, topn=5):
    X = R.T
    S = cosine_sim_matrix(X)
    user_vec = R[user]
    scores = user_vec @ S
    scores[user_vec > 0] = -1
    idx = np.argsort(-scores)[:topn]
    return idx

def precision_recall_at_k(R_true, rec_indices, k=5, threshold=4):
    users = R_true.shape[0]
    p = []
    r = []
    for u in range(users):
        rel = np.where(R_true[u] >= threshold)[0]
        if len(rel) == 0:
            continue
        recs = rec_indices[u]
        hits = np.isin(recs, rel).sum()
        p.append(hits / k)
        r.append(hits / len(rel))
    if len(p) == 0:
        return 0.0, 0.0
    return float(np.mean(p)), float(np.mean(r))

def evaluate_all(R, train, test, P, Q, k=5):
    Pn = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    Qn = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    R_hat = Pn @ Qn.T
    users = R.shape[0]
    rec_mf = []
    rec_user = []
    rec_item = []
    for u in range(users):
        taken = train[u] > 0
        scores = R_hat[u].copy()
        scores[taken] = -1
        rec_mf.append(np.argsort(-scores)[:k])
        rec_user.append(user_based_recommend(train, u, 5, k))
        rec_item.append(item_based_recommend(train, u, k))
    obs = np.argwhere(test > 0)
    if len(obs) == 0:
        rmse = 0.0
    else:
        preds = R_hat[tuple(obs.T)]
        rmse = float(np.sqrt(np.mean((test[tuple(obs.T)] - preds) ** 2)))
    p_mf, r_mf = precision_recall_at_k(test, rec_mf, k)
    p_u, r_u = precision_recall_at_k(test, rec_user, k)
    p_i, r_i = precision_recall_at_k(test, rec_item, k)
    return {"RMSE_MF": rmse, "P5_MF": p_mf, "R5_MF": r_mf, "P5_User": p_u, "R5_User": r_u, "P5_Item": p_i, "R5_Item": r_i}, rec_mf, rec_user, rec_item, R_hat

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
    metrics, rec_mf, rec_user, rec_item, R_hat = evaluate_all(R, train, test, P, Q, 5)
    users = np.random.default_rng(0).choice(R.shape[0], size=3, replace=False)
    for u in users:
        print("User", int(u), "MF:", rec_mf[u].tolist(), "UserCF:", rec_user[u].tolist(), "ItemCF:", rec_item[u].tolist())
    print(metrics)
    plot_history(history)
    heatmap_Q(Q)

if __name__ == "__main__":
    demo()
