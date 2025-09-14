#!/usr/bin/env python3
# per_device_vs_shared.py
# ESN(+LR) classification on UCI-HAR: per-device (Dirichlet split) vs shared baseline.

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def resolve_data_dir(user_path: str | None) -> str:
    if user_path:
        root = Path(user_path)
        assert (root / "train" / "Inertial Signals").exists(), \
            f"'Inertial Signals' not found under {user_path}"
        return str(root)
    for c in [
        "./UCI_HAR_Dataset",
        "./UCI HAR Dataset",
        "/content/gdrive2/MyDrive/UCI_HAR_Dataset",   # added candidate
        "/content/drive/MyDrive/UCI_HAR_Dataset",
        "/content/drive/MyDrive/UCI HAR Dataset",
        "/content/UCI_HAR_Dataset",
        "/content/UCI HAR Dataset",
    ]:
        p = Path(c)
        if (p / "train" / "Inertial Signals").exists():
            return str(p)
    raise FileNotFoundError("Pass --data_dir pointing to the UCI_HAR_Dataset root.")


def _read_matrix_txt(p: Path) -> np.ndarray:
    return np.loadtxt(p)


def load_split(root: str, split: str):
    sig = Path(root) / split / "Inertial Signals"

    def trio(names):
        paths = [sig / n for n in names]
        return paths if all(p.exists() for p in paths) else None

    acc = trio([f"body_acc_x_{split}.txt", f"body_acc_y_{split}.txt", f"body_acc_z_{split}.txt"])
    if acc is None:
        acc = trio([f"total_acc_x_{split}.txt", f"total_acc_y_{split}.txt", f"total_acc_z_{split}.txt"])
    if acc is None:
        raise FileNotFoundError("Accelerometer files not found.")

    gyro = trio([f"body_gyro_x_{split}.txt", f"body_gyro_y_{split}.txt", f"body_gyro_z_{split}.txt"])
    if gyro is None:
        raise FileNotFoundError("Gyroscope files not found.")

    mats = [_read_matrix_txt(p) for p in (acc + gyro)]
    X = np.stack(mats, axis=-1).astype(np.float32)
    y = np.loadtxt(Path(root) / split / f"y_{split}.txt").astype(int) - 1
    assert X.shape[1:] == (128, 6) and len(X) == len(y)
    return X, y


def fit_channel_norm(X):
    mu = X.mean(axis=(0, 1), keepdims=True)
    sd = X.std(axis=(0, 1), keepdims=True) + 1e-8
    return mu, sd


def apply_channel_norm(X, mu, sd):
    return (X - mu) / sd


def dirichlet_split_indices(y, n_devices, alpha, rng):
    idx = np.arange(len(y))
    rng.shuffle(idx)
    per_class = {}
    for c in np.unique(y):
        cls_idx = idx[y[idx] == c]
        share = rng.dirichlet(alpha * np.ones(n_devices))
        counts = np.floor(share * len(cls_idx)).astype(int)
        while counts.sum() < len(cls_idx):
            counts[rng.integers(0, n_devices)] += 1
        parts = np.split(cls_idx, np.cumsum(counts)[:-1])
        per_class[c] = parts
    device_indices = []
    for d in range(n_devices):
        d_idx = np.concatenate([per_class[c][d] for c in per_class.keys()])
        rng.shuffle(d_idx)
        device_indices.append(d_idx)
    return device_indices


class ESN:
    def __init__(self, n_res=1200, in_dim=6, leak=0.45, spectral_radius=0.90, sparsity=0.95, rng=None):
        self.n_res = n_res
        self.in_dim = in_dim
        self.leak = leak
        self.rng = rng or np.random.default_rng()
        self.Win = self.rng.uniform(-0.3, 0.3, size=(n_res, in_dim)).astype(np.float32)
        W = self.rng.normal(0, 1, size=(n_res, n_res)).astype(np.float32)
        mask = (self.rng.random((n_res, n_res)) < sparsity)
        W[mask] = 0.0
        eig = max(1e-6, np.max(np.abs(np.linalg.eigvals(W))))
        self.Wres = (W / eig * spectral_radius).astype(np.float32)

    def encode_features(self, X, R, washout=20):
        N, T, _ = X.shape
        Win = self.Win[:R, :]
        Wres = self.Wres[:R, :R]
        a = self.leak
        feats = np.zeros((N, 4 * R), dtype=np.float32)
        for n in range(N):
            h = np.zeros(R, dtype=np.float32)
            H = []
            for t in range(T):
                pre = Win @ X[n, t] + Wres @ h
                h = (1.0 - a) * h + a * np.tanh(pre)
                if t >= washout:
                    H.append(h.copy())
            if H:
                H = np.asarray(H)
                last = H[-1]
                mean = H.mean(0)
                mmax = H.max(0)
                sstd = H.std(0)
            else:
                last = h
                mean = np.zeros(R, np.float32)
                mmax = np.zeros(R, np.float32)
                sstd = np.zeros(R, np.float32)
            feats[n] = np.concatenate([last, mean, mmax, sstd])
        return feats


def train_readout(Htr, ytr, pca_dim, C, max_iter=3000):
    scaler = StandardScaler().fit(Htr)
    Htr_s = scaler.transform(Htr)
    pca_dim_eff = int(min(pca_dim, Htr_s.shape[1], max(10, Htr_s.shape[0] - 1)))
    pca = PCA(n_components=pca_dim_eff, random_state=0).fit(Htr_s)
    Ztr = pca.transform(Htr_s)
    clf = LogisticRegression(
        solver="lbfgs",
        multi_class="multinomial",
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        n_jobs=-1
    ).fit(Ztr, ytr)
    return clf, scaler, pca


def eval_readout(clf, scaler, pca, Hte, yte):
    Zte = pca.transform(scaler.transform(Hte))
    yhat = clf.predict(Zte)
    return accuracy_score(yte, yhat)


def main(args):
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = load_split(data_dir, "train")
    Xte, yte = load_split(data_dir, "test")
    mu, sd = fit_channel_norm(Xtr)
    Xtr_n = apply_channel_norm(Xtr, mu, sd)
    Xte_n = apply_channel_norm(Xte, mu, sd)

    dev_indices = dirichlet_split_indices(ytr, n_devices=args.n_devices, alpha=args.alpha, rng=rng)
    print("[split] per-device train sizes:", [len(ix) for ix in dev_indices])

    R_list = [400, 600, 800, 1200]
    results = []

    for d, idx in enumerate(dev_indices):
        Xd, yd = Xtr_n[idx], ytr[idx]
        esn = ESN(n_res=max(R_list), in_dim=6,
                  leak=args.leak, spectral_radius=args.spectral_radius, sparsity=0.95,
                  rng=np.random.default_rng(args.seed + 1000 + d))
        Htr_full = esn.encode_features(Xd, R=max(R_list), washout=args.washout)
        Hte_full = esn.encode_features(Xte_n, R=max(R_list), washout=args.washout)
        for R in R_list:
            Htr = Htr_full[:, : 4 * R]
            Hte = Hte_full[:, : 4 * R]
            pca_hint = args.pca_small if R <= 800 else args.pca_large
            clf, scaler, pca = train_readout(Htr, yd, pca_dim=pca_hint, C=args.C)
            acc = eval_readout(clf, scaler, pca, Hte, yte)
            results.append({"model": "per-device", "device": f"D{d}", "R": R, "acc": float(acc)})
            print(f"[per-device D{d}] R={R} acc={acc:.3f}")

    esn_shared = ESN(n_res=max(R_list), in_dim=6,
                     leak=args.leak, spectral_radius=args.spectral_radius, sparsity=0.95,
                     rng=np.random.default_rng(args.seed + 4242))
    Htr_full = esn_shared.encode_features(Xtr_n, R=max(R_list), washout=args.washout)
    Hte_full = esn_shared.encode_features(Xte_n, R=max(R_list), washout=args.washout)
    for R in R_list:
        Htr = Htr_full[:, : 4 * R]
        Hte = Hte_full[:, : 4 * R]
        pca_hint = args.pca_small if R <= 800 else args.pca_large
        clf, scaler, pca = train_readout(Htr, ytr, pca_dim=pca_hint, C=args.C)
        acc = eval_readout(clf, scaler, pca, Hte, yte)
        results.append({"model": "shared", "device": "ALL", "R": R, "acc": float(acc)})
        print(f"[shared] R={R} acc={acc:.3f}")

    df = pd.DataFrame(results)
    csv_path = out_dir / "offline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    perdev = df[df["model"] == "per-device"]
    for dev, g in perdev.groupby("device"):
        g = g.sort_values("R")
        ax.plot(g["R"], g["acc"], marker="o", linewidth=1.6, label=dev)
    shared = df[df["model"] == "shared"].sort_values("R")
    ax.plot(shared["R"], shared["acc"], linestyle="--", linewidth=2.5, marker="o", label="Shared")
    ax.set_xlabel("Reservoir size R")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(title="Series", ncol=3, fontsize=8)
    plt.tight_layout()
    png_path = out_dir / "offline_results.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--n_devices", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--washout", type=int, default=20)
    parser.add_argument("--leak", type=float, default=0.45)
    parser.add_argument("--spectral_radius", type=float, default=0.90)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--pca_small", type=int, default=256)
    parser.add_argument("--pca_large", type=int, default=384)

    import sys
    args = parser.parse_args([]) if "ipykernel" in sys.modules else parser.parse_args()

    # Auto-pick dataset dir if not passed
    if args.data_dir is None:
        for cand in [
            "/content/gdrive2/MyDrive/UCI_HAR_Dataset",
            "/content/drive/MyDrive/UCI_HAR_Dataset",
            "/content/UCI_HAR_Dataset",
            "./UCI_HAR_Dataset",
        ]:
            p = Path(cand)
            if (p / "train" / "Inertial Signals").exists():
                args.data_dir = str(p)
                break

    main(args)

