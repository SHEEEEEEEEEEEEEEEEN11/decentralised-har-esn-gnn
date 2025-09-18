#!/usr/bin/env python3
"""
Per-device ESN classification on UCI-HAR with Dirichlet device splits.

- Trains one ESN->LR per device and slice R in {400,600,800,1200}
- Selects R* by device-specific validation accuracy (after calibration)
- Evaluates on the common test set; saves counts & row-normalised confusion matrices
- Robust to missing classes per device (pads probabilities to 6 classes)
- Optional: prior correction and train balancing
- Saves a small JSON summary of best R and metrics per device.

Usage:
  python offline_classification/per_device_classification.py \
      --data_dir ./data/UCI_HAR_Dataset \
      --out_dir artifacts/perdev \
      --summary_path artifacts/perdev_summary.json \
      --alpha 5.0 --seed 42 --calibration temperature \
      --balance_train none --use_prior_correction 0
"""

import os
import argparse
import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

T_EXPECTED, C_EXPECTED = 128, 6
K_CLASSES = 6
N_DEVICES = 6
R_CHOICES = [400, 600, 800, 1200]

LEAK = 0.45
WASHOUT = 32
SPECTRAL = 0.90
SPARSITY = 0.05
INPUT_SCALE = 0.6
PCA_VARIANCE = 0.95
ECE_BINS = 15

LABEL_NAMES = ["WALKING", "WALKING_UP", "WALKING_DOWN", "SITTING", "STANDING", "LAYING"]


def _load_signals(split_root: str, split_name: str) -> np.ndarray:
    names = [
        f'body_acc_x_{split_name}.txt', f'body_acc_y_{split_name}.txt', f'body_acc_z_{split_name}.txt',
        f'body_gyro_x_{split_name}.txt', f'body_gyro_y_{split_name}.txt', f'body_gyro_z_{split_name}.txt'
    ]
    sigs = []
    for n in names:
        arr = np.loadtxt(os.path.join(split_root, "Inertial Signals", n))
        sigs.append(arr[..., None])
    return np.concatenate(sigs, axis=2).astype(np.float32)  # (N,128,6)

def load_uci_har(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tr = _load_signals(os.path.join(root, "train"), "train")
    X_te = _load_signals(os.path.join(root, "test"),  "test")
    y_tr = np.loadtxt(os.path.join(root, "train", "y_train.txt")).astype(int) - 1
    y_te = np.loadtxt(os.path.join(root, "test",  "y_test.txt")).astype(int) - 1
    print(f"[loader] X_train={X_tr.shape}, y_train={y_tr.shape} | X_test={X_te.shape}, y_test={y_te.shape}")
    return X_tr, y_tr, X_te, y_te


def dirichlet_partition(y: np.ndarray, n_devices: int, alpha: float, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_by_dev = [[] for _ in range(n_devices)]
    for c in np.unique(y):
        idx_c = np.nonzero(y == c)[0]
        rng.shuffle(idx_c)
        p = rng.dirichlet(alpha * np.ones(n_devices))
        counts = (p * len(idx_c)).astype(int)
        while counts.sum() < len(idx_c):
            counts[rng.integers(0, n_devices)] += 1
        cur = 0
        for d in range(n_devices):
            take = counts[d]
            if take > 0:
                idx_by_dev[d].extend(idx_c[cur:cur+take])
            cur += take
    for d in range(n_devices):
        idx = np.array(idx_by_dev[d], dtype=int)
        rng.shuffle(idx)
        idx_by_dev[d] = idx
    return idx_by_dev


def init_esn_full(C: int, R_full: int, input_scale: float, spectral: float, sparsity: float, seed: int):
    rng = np.random.default_rng(seed)
    Win_full  = (rng.normal(0,1,size=(R_full,C)).astype(np.float32) * input_scale)
    Wres_full = rng.normal(0,1,size=(R_full,R_full)).astype(np.float32)
    mask = (rng.random(size=Wres_full.shape) < sparsity)
    Wres_full *= mask
    # spectral scaling (power iteration)
    v = rng.normal(size=(R_full,)).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(50):
        v = Wres_full @ v
        v /= (np.linalg.norm(v) + 1e-12)
    lam_est = float(np.linalg.norm(Wres_full @ v))
    if lam_est > 0:
        Wres_full *= (spectral / lam_est)
    return Win_full, Wres_full

def esn_encode_batch(X: np.ndarray, Win_full, Wres_full, R_sub: int,
                     leak: float = LEAK, washout: int = WASHOUT) -> np.ndarray:
    N, T, C = X.shape
    Win  = Win_full[:R_sub, :]
    Wres = Wres_full[:R_sub, :R_sub]
    feats = np.empty((N, 3*R_sub), dtype=np.float32)
    for i in range(N):
        h = np.zeros((R_sub,), np.float32)
        H_last = None
        H_sum  = np.zeros((R_sub,), np.float32)
        H_sq   = np.zeros((R_sub,), np.float32)
        cnt = 0
        for t in range(T):
            pre = Win @ X[i,t] + Wres @ h
            h = (1.0 - leak) * h + leak * np.tanh(pre)
            if t >= washout:
                H_last = h
                H_sum += h
                H_sq  += h*h
                cnt += 1
        K = max(cnt, 1)
        mu  = H_sum / K
        var = np.maximum(H_sq / K - mu**2, 0.0)
        std = np.sqrt(var)
        last = H_last if H_last is not None else h
        feats[i] = np.concatenate([last, mu, std])
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, M: int = ECE_BINS) -> float:
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, M+1)
    ece = 0.0
    N = len(y_true)
    for b in range(M):
        lo, hi = bins[b], bins[b+1]
        idx = ((confidences > lo) & (confidences <= hi)) if b > 0 else ((confidences >= lo) & (confidences <= hi))
        if idx.sum() == 0:
            continue
        acc  = (preds[idx] == y_true[idx]).mean()
        conf = confidences[idx].mean()
        ece += (idx.sum() / N) * abs(acc - conf)
    return float(ece)

def softmax_logits(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= (p.sum(axis=1, keepdims=True) + 1e-12)
    return p

def temperature_scale_fit(logits_val: np.ndarray, y_val: np.ndarray,
                          grid: np.ndarray = np.linspace(0.5, 5.0, 19)) -> float:
    z = logits_val - logits_val.max(axis=1, keepdims=True)
    bestT, bestNLL = 1.0, 1e9
    for T in grid:
        p = np.exp(z / T); p /= p.sum(axis=1, keepdims=True)
        nll = log_loss(y_val, p, labels=np.arange(K_CLASSES))
        if nll < bestNLL:
            bestNLL, bestT = nll, float(T)
    return bestT

class VectorScaling:
    """Multiclass vector scaling via LR on logits."""
    def __init__(self, C=1.0, seed=42):
        self.lr = LogisticRegression(solver="lbfgs", C=C, max_iter=2000, random_state=seed)
    def fit(self, logits_val, y_val):
        self.lr.fit(logits_val, y_val); return self
    def predict_proba(self, logits):
        z = self.lr.decision_function(logits)
        return softmax_logits(z)

def align_proba(P: np.ndarray, classes_present: np.ndarray, K: int) -> np.ndarray:
    """Pad/reindex P (N x C_present) to N x K over classes 0..K-1."""
    N = P.shape[0]
    full = np.zeros((N, K), dtype=np.float64)
    full[:, classes_present] = P
    full = np.clip(full, 1e-12, 1.0)
    full /= full.sum(axis=1, keepdims=True)
    return full.astype(np.float32)


def save_confmats(cm_counts: np.ndarray, cm_norm: np.ndarray, out_base: str):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    # counts
    fig, ax = plt.subplots(figsize=(6.2,5.0))
    im = ax.imshow(cm_counts, interpolation="nearest")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(np.arange(K_CLASSES)); ax.set_yticks(np.arange(K_CLASSES))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right"); ax.set_yticklabels(LABEL_NAMES)
    for i in range(K_CLASSES):
        for j in range(K_CLASSES):
            ax.text(j,i,str(int(cm_counts[i,j])),ha="center",va="center",fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_base + "_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # normalised
    fig, ax = plt.subplots(figsize=(6.2,5.0))
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(np.arange(K_CLASSES)); ax.set_yticks(np.arange(K_CLASSES))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right"); ax.set_yticklabels(LABEL_NAMES)
    for i in range(K_CLASSES):
        for j in range(K_CLASSES):
            ax.text(j,i,f"{cm_norm[i,j]*100:.1f}%",ha="center",va="center",fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_base + "_norm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def safe_train_val_split(Xd: np.ndarray, yd: np.ndarray, seed: int, test_size: float = 0.20):
    present = np.bincount(yd, minlength=K_CLASSES)
    present = present[present > 0]
    can_stratify = (present.size >= 2) and (present.min() >= 2)
    if can_stratify:
        return train_test_split(Xd, yd, test_size=test_size, random_state=seed, stratify=yd)
    warnings.warn("[per-device] Too few samples in a class; falling back to non-stratified split.")
    return train_test_split(Xd, yd, test_size=test_size, random_state=seed, stratify=None)

def balance_by_class(X: np.ndarray, y: np.ndarray, mode: str, seed: int):
    """Equalise class counts among classes present in TRAIN:
       - 'undersample': downsample to min count
       - 'oversample' : upsample (with replacement) to max count
       - 'none'       : no balancing
       Classes absent in TRAIN remain absent (no synthesis)."""
    mode = mode.lower()
    if mode == "none":
        return X, y
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=K_CLASSES)
    present = np.where(counts > 0)[0]
    if present.size < 2:
        warnings.warn("[per-device] Training has <2 classes; skipping balancing.")
        return X, y
    idxs = {c: np.where(y == c)[0] for c in present}
    if mode == "undersample":
        n = int(np.min([len(idxs[c]) for c in present]))
        sel = np.concatenate([rng.choice(idxs[c], size=n, replace=False) for c in present])
    elif mode == "oversample":
        n = int(np.max([len(idxs[c]) for c in present]))
        sel = np.concatenate([rng.choice(idxs[c], size=n, replace=True) for c in present])
    else:
        return X, y
    rng.shuffle(sel)
    return X[sel], y[sel]


def run_per_device(data_dir: str, alpha: float, seed: int, out_dir: str,
                   calibration: str = "temperature",
                   balance_train: str = "none",
                   use_prior_correction: bool = False,
                   pca_variance: float = PCA_VARIANCE,
                   summary_path: str = "artifacts/perdev_summary.json"):
    X_tr_full, y_tr_full, X_te, y_te = load_uci_har(data_dir)
    assert X_tr_full.shape[1:] == (T_EXPECTED, C_EXPECTED)
    assert X_te.shape[1:] == (T_EXPECTED, C_EXPECTED)

    
    pi_global = np.bincount(y_tr_full, minlength=K_CLASSES).astype(np.float64)
    pi_global = np.clip(pi_global / pi_global.sum(), 1e-8, 1.0)

    parts = dirichlet_partition(y_tr_full, n_devices=N_DEVICES, alpha=alpha, seed=seed)
    summary = []

    for d in range(N_DEVICES):
        idx = parts[d]
        Xd, yd = X_tr_full[idx], y_tr_full[idx]
        print(f"[D{d}] class counts:", np.bincount(yd, minlength=K_CLASSES).tolist())

        X_tr, X_va, y_tr, y_va = safe_train_val_split(Xd, yd, seed=seed+10*d, test_size=0.20)

        
        X_tr_bal, y_tr_bal = balance_by_class(X_tr, y_tr, balance_train, seed=seed+20*d)

        
        mu  = X_tr_bal.mean(axis=(0,1), keepdims=True).astype(np.float32)
        sd  = (X_tr_bal.std(axis=(0,1), keepdims=True) + 1e-8).astype(np.float32)
        norm = lambda X: (X - mu) / sd
        X_tr_z, X_va_z, X_te_z = norm(X_tr_bal), norm(X_va), norm(X_te)

        
        R_full = max(R_CHOICES)
        Win_full, Wres_full = init_esn_full(C=C_EXPECTED, R_full=R_full,
                                            input_scale=INPUT_SCALE, spectral=SPECTRAL,
                                            sparsity=SPARSITY, seed=seed+100*d)

        
        pi_train = np.bincount(y_tr, minlength=K_CLASSES).astype(np.float64)
        pi_train = np.clip(pi_train / max(pi_train.sum(), 1.0), 1e-8, 1.0)
        log_prior_adj = np.log(pi_global / pi_train) if use_prior_correction else np.zeros(K_CLASSES, dtype=np.float64)

        perR = {}
        for R in R_CHOICES:
            # Encode
            H_tr = esn_encode_batch(X_tr_z, Win_full, Wres_full, R)
            H_va = esn_encode_batch(X_va_z, Win_full, Wres_full, R)
            H_te = esn_encode_batch(X_te_z, Win_full, Wres_full, R)

            # Scale + PCA
            scaler = StandardScaler().fit(H_tr)
            Z_tr, Z_va, Z_te = scaler.transform(H_tr), scaler.transform(H_va), scaler.transform(H_te)
            pca = PCA(n_components=pca_variance, svd_solver="full").fit(Z_tr)
            Z_tr, Z_va, Z_te = pca.transform(Z_tr), pca.transform(Z_va), pca.transform(Z_te)

            
            ntr = len(y_tr_bal)
            C_adj = float(np.clip(10.0 * (ntr / 1000.0), 0.1, 5.0))

            clf = LogisticRegression(solver="lbfgs", C=C_adj, max_iter=20000, random_state=seed+d)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(Z_tr, y_tr_bal)

            
            adj_vec = log_prior_adj[clf.classes_]
            logits_va = clf.decision_function(Z_va) + adj_vec
            logits_te = clf.decision_function(Z_te) + adj_vec

            
            calib = calibration.lower()
            if calib == "vector" and np.unique(y_va).size >= 2:
                calibrator = VectorScaling(C=1.0, seed=seed+d).fit(logits_va, y_va)
                val_probs_raw = calibrator.predict_proba(logits_va)
                P_te_raw = calibrator.predict_proba(logits_te)
            else:
                T_star = temperature_scale_fit(logits_va, y_va)
                val_probs_raw = softmax_logits(logits_va / max(T_star, 1e-8))
                P_te_raw = softmax_logits(logits_te / max(T_star, 1e-8))

            # Align VAL probs to all 6 classes before computing val acc
            val_probs = align_proba(val_probs_raw, clf.classes_, K_CLASSES)
            val_acc = accuracy_score(y_va, val_probs.argmax(1))

            # Align TEST probs to all 6 classes
            P_te = align_proba(P_te_raw, clf.classes_, K_CLASSES)

            perR[R] = dict(val_acc=val_acc, P_te=P_te)

        
        best_val = max(v["val_acc"] for v in perR.values())
        best_R_candidates = [R for R,v in perR.items() if abs(v["val_acc"] - best_val) < 1e-12]
        best_R = max(best_R_candidates)

        
        P_te = perR[best_R]["P_te"]
        y_pred = P_te.argmax(1)
        test_acc = accuracy_score(y_te, y_pred)
        nll = log_loss(y_te, P_te, labels=np.arange(K_CLASSES))
        ece = expected_calibration_error(P_te, y_te, M=ECE_BINS)

        cm_counts = confusion_matrix(y_te, y_pred, labels=np.arange(K_CLASSES))
        cm_norm   = confusion_matrix(y_te, y_pred, labels=np.arange(K_CLASSES), normalize="true")
        base = os.path.join(out_dir, f"D{d}_R{best_R}")
        save_confmats(cm_counts, cm_norm, base)

        print(f"[dev D{d}] best_R={best_R} | VAL acc={best_val:.3f} | "
              f"TEST acc={test_acc:.3f} | NLL={nll:.3f} | ECE={ece*100:.2f}% "
              f"| saved: {base}_counts.png, {base}_norm.png")

        summary.append({
            "device": int(d),
            "best_R": int(best_R),
            "val_acc": float(best_val),
            "test_acc": float(test_acc),
            "nll": float(nll),
            "ece_percent": float(ece*100.0)
        })

    print("\nPer-device best-R summary (by validation accuracy):")
    print("Dev  R*   ValAcc   TestAcc   NLL     ECE(%)")
    for row in summary:
        d,bR,vA,tA,nll,ece = row["device"], row["best_R"], row["val_acc"], row["test_acc"], row["nll"], row["ece_percent"]
        print(f"D{d}  {str(bR).ljust(4)} {vA:6.3f}   {tA:7.3f}  {nll:6.3f}   {ece:6.2f}")

    
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"alpha": float(alpha), "seed": int(seed), "summary": summary}, f, indent=2)
    print(f"[saved] Summary -> {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Per-device ESN on UCI-HAR with Dirichlet splits.")
    p.add_argument("--data_dir", type=str, default=os.getenv("UCI_HAR_DIR", "./data/UCI_HAR_Dataset"),
                   help="Path to UCI_HAR_Dataset root (with train/test).")
    p.add_argument("--out_dir", type=str, default="artifacts/perdev",
                   help="Output directory for figures (counts & normalised).")
    p.add_argument("--summary_path", type=str, default="artifacts/perdev_summary.json",
                   help="Path to save JSON summary with per-device best R and metrics.")
    p.add_argument("--alpha", type=float, default=5.0, help="Dirichlet concentration.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--calibration", type=str, default="temperature", choices=["temperature","vector"],
                   help="Calibration method.")
    p.add_argument("--balance_train", type=str, default="none", choices=["none","undersample","oversample"],
                   help="Balance train split per device.")
    p.add_argument("--use_prior_correction", type=int, default=0,
                   help="1 to enable log(pi_global/pi_device) prior correction before calibration.")
    p.add_argument("--pca_variance", type=float, default=PCA_VARIANCE, help="PCA retained variance.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_per_device(
        data_dir=args.data_dir,
        alpha=args.alpha,
        seed=args.seed,
        out_dir=args.out_dir,
        calibration=args.calibration,
        balance_train=args.balance_train,
        use_prior_correction=bool(args.use_prior_correction),
        pca_variance=args.pca_variance,
        summary_path=args.summary_path
    )
