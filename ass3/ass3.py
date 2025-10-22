import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

np.random.seed(7)

def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
def sigmoid(z): return 1/(1+np.exp(-z))
def sigmoid_grad(z): s = sigmoid(z); return s*(1-s)
def tanh(z): return np.tanh(z)
def tanh_grad(z): return 1 - np.tanh(z)**2

ACT = {"relu": (relu, relu_grad), "tanh": (tanh, tanh_grad), "sigmoid": (sigmoid, sigmoid_grad)}

def init_params():
    W1 = np.random.randn(2,2) * 0.5
    b1 = np.zeros((1,2))
    W2 = np.random.randn(2,1) * 0.5
    b2 = np.zeros((1,1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward(params, X, act="relu"):
    f, _ = ACT[act]
    z1 = X @ params["W1"] + params["b1"]
    h1 = f(z1)
    z2 = h1 @ params["W2"] + params["b2"]
    yhat = sigmoid(z2)
    return yhat, {"X": X, "z1": z1, "h1": h1, "z2": z2, "yhat": yhat}

def loss_mse(y, yhat): return 0.5 * np.mean((y - yhat) ** 2)
def accuracy(y, yhat): return np.mean((yhat >= 0.5) == (y == 1))

def backward(params, cache, y, act="relu"):
    _, fgrad = ACT[act]
    n = y.shape[0]
    yhat, z2, h1, X = cache["yhat"], cache["z2"], cache["h1"], cache["X"]
    dl_dyhat = (yhat - y) / n
    dl_dz2 = dl_dyhat * sigmoid_grad(z2)
    dW2 = h1.T @ dl_dz2
    db2 = np.sum(dl_dz2, axis=0, keepdims=True)
    dh1 = dl_dz2 @ params["W2"].T
    dz1 = dh1 * fgrad(cache["z1"])
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def step(params, grads, opt, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, rho=0.9):
    name, st = opt["name"], opt["state"]
    if name == "sgd":
        for k in params: params[k] -= lr * grads[k]
    elif name == "rmsprop":
        for k in params:
            st["v"][k] = rho * st["v"][k] + (1 - rho) * (grads[k] ** 2)
            params[k] -= lr * grads[k] / (np.sqrt(st["v"][k]) + eps)
    elif name == "adam":
        st["t"] += 1
        for k in params:
            st["m"][k] = beta1 * st["m"][k] + (1 - beta1) * grads[k]
            st["v"][k] = beta2 * st["v"][k] + (1 - beta2) * (grads[k] ** 2)
            mhat = st["m"][k] / (1 - beta1 ** st["t"])
            vhat = st["v"][k] / (1 - beta2 ** st["t"])
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
    else:
        raise ValueError("unknown optimizer")

def make_opt(params, name="sgd"):
    st = {"m": {}, "v": {}, "t": 0}
    for k, v in params.items():
        st["m"][k] = np.zeros_like(v)
        st["v"][k] = np.zeros_like(v)
    return {"name": name, "state": st}

def train(X, y, act="relu", opt_name="sgd", lr=0.05, epochs=200, batch=32):
    params = init_params()
    opt = make_opt(params, opt_name)
    n = X.shape[0]
    losses = []
    for _ in range(epochs):
        idx = np.random.permutation(n)
        Xs, ys = X[idx], y[idx]
        for i in range(0, n, batch):
            xb, yb = Xs[i:i+batch], ys[i:i+batch]
            yhat, cache = forward(params, xb, act)
            grads = backward(params, cache, yb, act)
            step(params, grads, opt, lr=lr)
        yhat_all, _ = forward(params, X, act)
        losses.append(loss_mse(y, yhat_all))
    return params, np.array(losses)

def ensure_dataset(path):
    if path.exists():
        df = pd.read_csv(path)
        need = {"x1","x2","y"}.issubset({c.lower() for c in df.columns})
        if not need:
            n = 200
            x1 = np.random.poisson(2.0, size=n)
            x2 = np.random.poisson(2.0, size=n)
            margin = (x1 - x2) + np.random.normal(0, 0.5, size=n)
            y = (margin > 0).astype(int)
            df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
            df.to_csv(path, index=False)
    else:
        n = 200
        x1 = np.random.poisson(2.0, size=n)
        x2 = np.random.poisson(2.0, size=n)
        margin = (x1 - x2) + np.random.normal(0, 0.5, size=n)
        y = (margin > 0).astype(int)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        df.to_csv(path, index=False)
    return df

def save_plot(losses, title, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()

def main():
    OUT = Path.cwd() / "outputs"
    OUT.mkdir(exist_ok=True)
    data_path = Path("dataset_Ai_essentials_ass3.csv")
    df = ensure_dataset(data_path)
    X = df[["x1","x2"]].values.astype(float)
    y = df["y"].values.reshape(-1,1).astype(float)

    p_base, losses_base = train(X, y, act="relu", opt_name="sgd", lr=0.05, epochs=200)
    save_plot(losses_base, "Training Loss — ReLU + SGD", OUT / "loss_relu_sgd.png")
    yhat_all, _ = forward(p_base, X, "relu")
    acc_base = accuracy(y, yhat_all)

    rng = np.random.default_rng(42)
    sample_x = np.array([[2.0, 1.0]])
    sample_y = np.array([[1.0]])
    params_s = {
        "W1": rng.uniform(-1,1,size=(2,2)),
        "b1": np.zeros((1,2)),
        "W2": rng.uniform(-1,1,size=(2,1)),
        "b2": np.zeros((1,1)),
    }
    yhat_s, cache_s = forward(params_s, sample_x, "relu")
    L_s = loss_mse(sample_y, yhat_s)
    dl_dyhat = (yhat_s - sample_y)
    dl_dz2 = dl_dyhat * sigmoid_grad(cache_s["z2"])
    dW2 = cache_s["h1"].T @ dl_dz2
    eta = 0.1
    vi_old = params_s["W2"][0,0]
    vi_new = vi_old - eta * dW2[0,0]

    acts = ["relu","tanh","sigmoid"]
    opts = ["sgd","adam","rmsprop"]
    rows = []
    for a in acts:
        for o in opts:
            p, losses = train(X, y, act=a, opt_name=o, lr=0.05, epochs=150)
            yhat, _ = forward(p, X, a)
            acc = accuracy(y, yhat)
            rows.append({"activation": a, "optimizer": o, "final_loss": float(losses[-1]), "accuracy": float(acc)})
            save_plot(losses, f"Training Loss — {a.upper()} + {o.upper()}", OUT / f"loss_{a}_{o}.png")

    res = pd.DataFrame(rows).sort_values(["activation","optimizer"])
    res.to_csv(OUT / "results.csv", index=False)

    with open(OUT / "summary.txt", "w") as f:
        f.write("Baseline ReLU+SGD accuracy: %.3f\n" % acc_base)
        f.write("Backprop single-sample:\n")
        f.write("yhat=%.6f, loss=%.6f, dL_dvi=%.6f, vi_old=%.6f, vi_new=%.6f, eta=%.3f\n" % (
            yhat_s[0,0], L_s, dW2[0,0], vi_old, vi_new, eta
        ))
        f.write("\nGrid results:\n")
        f.write(res.to_csv(index=False))

if __name__ == "__main__":
    main()
