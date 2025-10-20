import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_users=100, n_items=20, density=0.7, seed=42):
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
    return np.clip(S, -1.0, 1.0)

def topk_similar_items(R, item, k=5):
    X = R.T
    S = cosine_sim_matrix(X)
    sims = S[item].copy()
    sims[item] = -1.0
    nn = np.argsort(-sims)[:k]
    return nn, sims[nn]

def print_items_neighbors(item, nn, sims):
    print(f"Top-{len(nn)} similar items to item {item}:")
    for r, (j, c) in enumerate(zip(nn, sims), 1):
        print(f"{r}. item {j} — cosine={c:.3f}")

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

def topk_similar_users(R, user, k=5):
    S = cosine_sim_matrix(R)
    sims = S[user].copy()
    sims[user] = -1.0
    nn = np.argsort(-sims)[:k]
    return nn, sims[nn]

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

def plot_top5(labels, scores, title="Top-5", xlabel="Item", ylabel="Score"):
    plt.figure()
    plt.bar(labels, scores)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def print_users_with_cosine(idx, cos, user):
    print(f"Top-{len(idx)} similar users to user {user}:")
    for r, (v, c) in enumerate(zip(idx, cos), 1):
        print(f"{r}. user {v} — cosine={c:.3f}")

def demo():
    R = generate_data()
    user_avg, item_avg = averages(R)
    train, test = train_test_split(R)
    print(np.argwhere(train > 0).shape)
    print(np.argwhere(test > 0).shape)
    print(np.count_nonzero(train))
    print(np.count_nonzero(test))
    u = 0

    idx_u, sc_u = user_based_recommend(train, u, k=5, topn=5)
    nn_u, cos_u = topk_similar_users(train, u, k=5)

    p_u, r_u = precision_recall_at_k(train, test, lambda X, user, topn=5: user_based_recommend(X, user, 5, topn), 5)

    print_users_with_cosine(nn_u, cos_u, u)
    plot_top5([f"U{j}" for j in nn_u], cos_u, f"Top-5 Users similar to U{u}", "User", "Cosine similarity")
    plot_top5([f"C{j}" for j in idx_u], sc_u, "User-based Top-5", "Item", "Score")
    nn_it, cos_it = topk_similar_items(train, item=0, k=5)
    print_items_neighbors(0, nn_it, cos_it)

    print("User averages shape:", user_avg.shape)
    print("Item averages shape:", item_avg.shape)
    print("User-based precision@5:", round(p_u, 3), "recall@5:", round(r_u, 3))

if __name__ == "__main__":
    demo()