
import os, warnings, math
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

T_EXPECTED, C_EXPECTED = 128, 6
R_FULL = 1200
R_CHOICES = [400, 600, 800, 1200]
LEAK, WASHOUT = 0.25, 32
SPECTRAL, SPARSITY, INPUT_SCALE = 0.85, 0.05, 0.6
PCA_VARIANCE = 0.95
RNG_SEED = 42
K_CLASSES = 6

def _load_signals(split_root, split_name):
    names = [
        f'body_acc_x_{split_name}.txt', f'body_acc_y_{split_name}.txt', f'body_acc_z_{split_name}.txt',
        f'body_gyro_x_{split_name}.txt', f'body_gyro_y_{split_name}.txt', f'body_gyro_z_{split_name}.txt'
    ]
    sigs = []
    for n in names:
        arr = np.loadtxt(os.path.join(split_root, "Inertial Signals", n))
        sigs.append(arr[..., None])  # (N, 128) -> (N, 128, 1)
    X = np.concatenate(sigs, axis=2).astype(np.float32)  # (N, 128, 6)
    return X

def load_uci_har(root):
    X_tr = _load_signals(os.path.join(root, "train"), "train")
    X_te = _load_signals(os.path.join(root, "test"),  "test")
    y_tr = np.loadtxt(os.path.join(root, "train", "y_train.txt")).astype(int) - 1
    y_te = np.loadtxt(os.path.join(root, "test",  "y_test.txt")).astype(int) - 1
    print(f"[loader] X_train={X_tr.shape}, y_train={y_tr.shape} | X_test={X_te.shape}, y_test={y_te.shape}")
    return X_tr, y_tr, X_te, y_te

DATA_DIR = '/content/drive/MyDrive/UCI_HAR_Dataset'
X_train_full, y_train_full, X_test_full, y_test_full = load_uci_har(DATA_DIR)


X_tr, X_va, y_tr, y_va = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=RNG_SEED, stratify=y_train_full
)
X_te, y_te = X_test_full, y_test_full

mu  = X_tr.mean(axis=(0,1), keepdims=True).astype(np.float32)
sd  = (X_tr.std(axis=(0,1), keepdims=True) + 1e-8).astype(np.float32)
def norm(X): return (X - mu) / sd

def init_esn_full(C, R_full, input_scale, spectral, sparsity, seed):
    rng = np.random.default_rng(seed)
    Win_full  = (rng.normal(0,1,size=(R_full,C)).astype(np.float32) * input_scale)
    Wres_full = rng.normal(0,1,size=(R_full,R_full)).astype(np.float32)
    mask      = (rng.random(size=Wres_full.shape) < sparsity)
    Wres_full *= mask
    # spectral scaling (power iteration estimate)
    v = rng.normal(size=(R_full,)).astype(np.float32); v /= (np.linalg.norm(v)+1e-12)
    for _ in range(50):
        v = Wres_full @ v; v /= (np.linalg.norm(v)+1e-12)
    lam_est = float(np.linalg.norm(Wres_full @ v))
    if lam_est > 1e-8: Wres_full *= (spectral/lam_est)
    return Win_full, Wres_full

def esn_encode_batch(X, Win_full, Wres_full, R_sub, leak=0.25, washout=32):
    N, T, C = X.shape
    Win  = Win_full[:R_sub, :]
    Wres = Wres_full[:R_sub, :R_sub]
    feats = np.empty((N, 3*R_sub), dtype=np.float32)
    for i in range(N):
        h = np.zeros((R_sub,), np.float32)
        H_last=None; H_sum=np.zeros((R_sub,),np.float32); H_sq=np.zeros((R_sub,),np.float32); cnt=0
        for t in range(T):
            pre = Win @ X[i,t] + Wres @ h
            h = (1.0-leak)*h + leak*np.tanh(pre)
            if t >= washout:
                H_last=h; H_sum+=h; H_sq+=h*h; cnt+=1
        K=max(cnt,1)
        mu_ = H_sum/K
        std = np.sqrt(np.maximum(H_sq/K - mu_**2, 0.0))
        last = H_last if H_last is not None else h
        feats[i] = np.concatenate([last, mu_, std])
    # row-wise L2 normalise
    return feats / (np.linalg.norm(feats,axis=1,keepdims=True)+1e-8)

def grid_temperature(logits_val, y_val, grid=np.linspace(0.5, 5.0, 19)):
    z = logits_val - logits_val.max(axis=1, keepdims=True)
    bestT, bestNLL = 1.0, 1e9
    for T in grid:
        p = np.exp(z/T); p /= p.sum(axis=1, keepdims=True)
        nll = log_loss(y_val, p, labels=np.arange(p.shape[1]))
        if nll < bestNLL: bestNLL, bestT = nll, T
    return float(bestT)

def expected_calibration_error(probs, y_true, M=15):
    """ECE with equal-width confidence bins on max-prob class."""
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, M+1)
    ece = 0.0
    N = len(y_true)
    for b in range(M):
        lo, hi = bins[b], bins[b+1]
        idx = (confidences > lo) & (confidences <= hi) if b>0 else (confidences >= lo) & (confidences <= hi)
        if idx.sum() == 0: continue
        acc  = (preds[idx] == y_true[idx]).mean()
        conf = confidences[idx].mean()
        ece += (idx.sum()/N) * abs(acc - conf)
    return float(ece)

Win_full, Wres_full = init_esn_full(C_EXPECTED, R_FULL, INPUT_SCALE, SPECTRAL, SPARSITY, seed=RNG_SEED)

X_tr_z, X_va_z, X_te_z = norm(X_tr), norm(X_va), norm(X_te)

results = {}
best_R, best_val_acc = None, -1.0
for R in R_CHOICES:
    H_tr = esn_encode_batch(X_tr_z, Win_full, Wres_full, R, LEAK, WASHOUT)
    H_va = esn_encode_batch(X_va_z, Win_full, Wres_full, R, LEAK, WASHOUT)
    H_te = esn_encode_batch(X_te_z, Win_full, Wres_full, R, LEAK, WASHOUT)

    sc   = StandardScaler().fit(H_tr)
    Z_tr = sc.transform(H_tr); Z_va = sc.transform(H_va); Z_te = sc.transform(H_te)

    pca  = PCA(n_components=PCA_VARIANCE, svd_solver="full").fit(Z_tr)
    Z_tr = pca.transform(Z_tr); Z_va = pca.transform(Z_va); Z_te = pca.transform(Z_te)

    clf = LogisticRegression(max_iter=6000, solver="lbfgs", C=0.7, random_state=RNG_SEED)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(Z_tr, y_tr)
        if any(isinstance(w.message, ConvergenceWarning) for w in wlist):
            clf = LogisticRegression(max_iter=10000, solver="lbfgs", C=0.7, random_state=RNG_SEED)
            clf.fit(Z_tr, y_tr)

    
    T_star = grid_temperature(clf.decision_function(Z_va), y_va)

    
    z_va = clf.decision_function(Z_va) / max(T_star,1e-8)
    z_va -= z_va.max(axis=1, keepdims=True)
    P_va = np.exp(z_va); P_va /= P_va.sum(axis=1, keepdims=True)
    val_acc = accuracy_score(y_va, P_va.argmax(1))
    print(f"[R={R}] Val Acc = {val_acc:.4f} (PCA dims={Z_va.shape[1]})")

    results[R] = dict(scaler=sc, pca=pca, clf=clf, T=T_star)

    if val_acc > best_val_acc:
        best_val_acc, best_R = val_acc, R

print(f"\n>> Best slice by validation accuracy: R={best_R} (val acc={best_val_acc:.4f})")


best = results[best_R]
Z_te_best = best["pca"].transform(best["scaler"].transform(
    esn_encode_batch(X_te_z, Win_full, Wres_full, best_R, LEAK, WASHOUT)
))
z_te = best["clf"].decision_function(Z_te_best) / max(best["T"], 1e-8)
z_te -= z_te.max(axis=1, keepdims=True)
P_te = np.exp(z_te); P_te /= P_te.sum(axis=1, keepdims=True)
y_pred = P_te.argmax(1)

acc = accuracy_score(y_te, y_pred)
mf1 = f1_score(y_te, y_pred, average="macro")
nll = log_loss(y_te, P_te, labels=np.arange(K_CLASSES))
ece = expected_calibration_error(P_te, y_te, M=15)

print(f"\n[Test @ R={best_R}] Acc={acc:.4f} | Macro-F1={mf1:.4f} | NLL={nll:.4f} | ECE={ece*100:.2f}%")


cm = confusion_matrix(y_te, y_pred, labels=np.arange(K_CLASSES))
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, interpolation='nearest')  # no manual colors as requested
ax.set_title(f'Confusion Matrix (R={best_R})')
ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
ax.set_xticks(np.arange(K_CLASSES)); ax.set_yticks(np.arange(K_CLASSES))
# annotate cells
for i in range(K_CLASSES):
    for j in range(K_CLASSES):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
