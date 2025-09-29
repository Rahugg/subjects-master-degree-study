
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_users=100, n_items=20, density=0.4, seed=42):
    rng = np.random.default_rng(seed)
    R = np.zeros((n_users, n_items), dtype=int)
    mask = rng.random((n_users, n_items)) < density
    R[mask] = rng.integers(1, 6, size=mask.sum())
    return R

def train_test_split(R, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    nz = np.argwhere(R > 0)
    n_test = int(len(nz) * test_ratio)
    rng.shuffle(nz)
    test_idx = nz[:n_test]
    train = R.copy()
    train[tuple(test_idx.T)] = 0
    test = np.zeros_like(R)
    test[tuple(test_idx.T)] = R[tuple(test_idx.T)]
    return train, test

def cosine_sim_matrix(X):
    X = X.astype(np.float64)
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
    return idx, scores[idx]

def item_based_recommend(R, user, topn=5):
    X = R.T
    S = cosine_sim_matrix(X)
    user_vec = R[user]
    scores = user_vec @ S
    scores[user_vec > 0] = -1
    idx = np.argsort(-scores)[:topn]
    return idx, scores[idx]

def precision_recall_at_k(train, test, recommend_fn, k=5):
    users = np.arange(train.shape[0])
    precisions = []
    recalls = []
    for u in users:
        recs, _ = recommend_fn(train, u, topn=k)
        true_pos = np.where(test[u] >= 4)[0]
        if len(true_pos) == 0:
            continue
        hit = np.isin(recs, true_pos).sum()
        precisions.append(hit / k)
        recalls.append(hit / len(true_pos))
    if len(precisions) == 0:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))

def averages(R):
    item_avg = np.divide(R.sum(axis=0), (R > 0).sum(axis=0), out=np.zeros(R.shape[1], float), where=(R > 0).sum(axis=0) > 0)
    user_avg = np.divide(R.sum(axis=1), (R > 0).sum(axis=1), out=np.zeros(R.shape[0], float), where=(R > 0).sum(axis=1) > 0)
    return user_avg, item_avg

def plot_top5(labels, scores, title="Top-5 Recommendations"):
    plt.figure()
    plt.bar(labels, scores)
    plt.title(title)
    plt.xlabel("Item")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

def demo():
    R = generate_data()
    user_avg, item_avg = averages(R)
    train, test = train_test_split(R)
    u = 0
    idx_u, sc_u = user_based_recommend(train, u, k=5, topn=5)
    idx_i, sc_i = item_based_recommend(train, u, topn=5)
    p_u, r_u = precision_recall_at_k(train, test, lambda X, user, topn=5: user_based_recommend(X, user, 5, topn), 5)
    p_i, r_i = precision_recall_at_k(train, test, lambda X, user, topn=5: item_based_recommend(X, user, topn), 5)
    plot_top5([f"C{j}" for j in idx_u], sc_u, "User-based Top-5")
    plot_top5([f"C{j}" for j in idx_i], sc_i, "Item-based Top-5")
    print("User averages shape:", user_avg.shape)
    print("Item averages shape:", item_avg.shape)
    print("User-based precision@5:", round(p_u, 3), "recall@5:", round(r_u, 3))
    print("Item-based precision@5:", round(p_i, 3), "recall@5:", round(r_i, 3))

if __name__ == "__main__":
    demo()
