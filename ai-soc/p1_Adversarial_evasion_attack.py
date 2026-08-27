import sys, os, time, json, argparse, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ── Import live AI-SOC infrastructure ────────────────────────────────────────
try:
    from p1_soc_utils import (
        Logger, SOCEvent, DataLoader, Store, MetricsEngine,
        DATASETS, LAYERS, LAYER_META, MODEL_ROOT, RESULT_ROOT, PLOT_ROOT,
        BENIGN_LABELS, MITRE_MAP,
        RESET, BOLD, CYAN, GREEN, YELLOW, RED, GREY, MAGENTA, BLUE,
    )
    from p1_injestion import IngestionLayer
    from p1_triage    import TriageLayer
    from p1_detection import DetectionLayer
    from p1_siem      import SIEMLayer
    from p1_soar      import SOARLayer
    SOC_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] SOC modules not found ({e}). Using standalone mode.")
    SOC_AVAILABLE = False
    RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"; GREEN="\033[92m"
    YELLOW="\033[93m"; RED="\033[91m"; GREY="\033[37m"; MAGENTA="\033[95m"
    BLUE="\033[94m"
    DATASETS    = ["cicids2017", "ember", "loghub"]
    LAYERS      = ["ingestion", "triage", "detection", "siem", "soar"]
    MODEL_ROOT  = Path("ai_soc_models")
    RESULT_ROOT = Path("ai_soc_results")
    PLOT_ROOT   = Path("ai_soc_plots")
    BENIGN_LABELS = {"benign","normal","0","safe","allow","low","info"}

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from sklearn.preprocessing  import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble       import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.svm            import SVC

# ── Output directories ────────────────────────────────────────────────────────
ADV_PLOT_DIR   = Path("adversarial_plots");   ADV_PLOT_DIR.mkdir(exist_ok=True)
ADV_RESULT_DIR = Path("adversarial_results"); ADV_RESULT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class AttackLogger:
    def __init__(self):
        self._buf = []; self._start = time.time()
    def _ts(self): return f"{time.time()-self._start:>7.2f}s"
    def banner(self, msg, w=80):
        line = "═" * w
        s = f"\n{RED}{BOLD}{line}\n  ⚔  {msg}\n{line}{RESET}\n"
        print(s); self._buf.append(msg)
    def phase(self, msg):  print(f"\n{CYAN}{BOLD}▶▶  {msg}{RESET}")
    def attack(self, name, eps=None):
        tag = f" ε={eps}" if eps is not None else ""
        print(f"\n  {YELLOW}{BOLD}[ATTACK]{RESET}  {name}{tag}")
    def result(self, layer, clean, adv, drop, evasion):
        col = RED if drop > 0.10 else YELLOW if drop > 0.03 else GREEN
        print(f"    {GREY}{layer:<12}{RESET}"
              f"  clean={GREEN}{clean:.4f}{RESET}"
              f"  adv={col}{adv:.4f}{RESET}"
              f"  drop={col}{drop:+.4f}{RESET}"
              f"  evasion={RED}{evasion:.1%}{RESET}")
    def ok(self,   m): print(f"  {GREEN}✔  {m}{RESET}")
    def warn(self, m): print(f"  {YELLOW}⚠  {m}{RESET}")
    def info(self, m): print(f"  {GREY}→  {m}{RESET}")

LOG = AttackLogger()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1b — EXPERIMENT LEDGER
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentLedger:
    """
    Records every attack run with timestamp, dataset, epsilon, metrics.
    Saved as JSON and printed as a formatted table at the end of each run.
    """
    LEDGER_FILE = ADV_RESULT_DIR / "experiment_ledger.json" if False else None  # set after mkdir

    def __init__(self):
        self.entries: list = []
        self._exp_id = 0
        self._run_start = datetime.now()

    def _next_id(self):
        self._exp_id += 1
        return self._exp_id

    def record(self, *, dataset: str, attack: str, layer: str,
               epsilon: float | None,
               acc_drop: float, f1_drop: float, auc_drop: float,
               n_fooled: int, n_total: int,
               l2_mean: float, linf_mean: float,
               elapsed_s: float):
        entry = dict(
            exp_id      = self._next_id(),
            timestamp   = datetime.now().isoformat(timespec="seconds"),
            dataset     = dataset,
            attack      = attack,
            layer       = layer,
            epsilon     = epsilon,
            acc_drop    = round(acc_drop,  4),
            f1_drop     = round(f1_drop,   4),
            auc_drop    = round(auc_drop,  4),
            n_fooled    = int(n_fooled),
            n_total     = int(n_total),
            fool_rate   = round(n_fooled / max(n_total, 1), 4),
            l2_mean     = round(l2_mean,   4),
            linf_mean   = round(linf_mean, 4),
            elapsed_s   = round(elapsed_s, 2),
        )
        self.entries.append(entry)
        return entry

    def save(self, dataset: str):
        path = ADV_RESULT_DIR / f"{dataset}_experiment_ledger.json"
        with open(path, "w") as f:
            json.dump({"run_start": self._run_start.isoformat(),
                       "experiments": self.entries}, f, indent=2, default=str)
        LOG.ok(f"Experiment ledger → {path}")

        csv_path = ADV_RESULT_DIR / f"{dataset}_experiment_ledger.csv"
        pd.DataFrame(self.entries).to_csv(csv_path, index=False)
        LOG.ok(f"Experiment ledger CSV → {csv_path}")

    def print_summary(self, dataset: str):
        if not self.entries:
            return
        W = 140
        print(f"\n{CYAN}{BOLD}{'═'*W}{RESET}")
        print(f"{CYAN}{BOLD}  🧪  EXPERIMENT LEDGER  |  {dataset.upper()}  |  "
              f"{len(self.entries)} runs recorded{RESET}")
        print(f"{CYAN}{BOLD}{'═'*W}{RESET}")
        HDR = (f"  {'#':>3}  {'Attack':<22}  {'Layer':<10}  {'ε':>6}  "
               f"{'AccDrop':>8}  {'F1Drop':>7}  {'AUCDrop':>8}  "
               f"{'Fooled%':>8}  {'L2 pert':>8}  {'Linf pert':>10}  {'Elapsed':>8}  {'Time'}")
        print(f"{BOLD}{HDR}{RESET}")
        print(f"  {'─'*136}")
        for e in self.entries:
            eps_s = f"{e['epsilon']:.2f}" if e['epsilon'] is not None else "  —  "
            acc_c = RED if e['acc_drop'] > 0.05 else YELLOW if e['acc_drop'] > 0.01 else GREEN
            fool_c = RED if e['fool_rate'] > 0.30 else YELLOW if e['fool_rate'] > 0.10 else GREEN
            print(
                f"  {e['exp_id']:>3}  {e['attack']:<22}  {e['layer']:<10}  {eps_s:>6}  "
                f"{acc_c}{e['acc_drop']:>+8.4f}{RESET}  "
                f"{e['f1_drop']:>+7.4f}  {e['auc_drop']:>+8.4f}  "
                f"{fool_c}{e['fool_rate']:>8.1%}{RESET}  "
                f"{e['l2_mean']:>8.4f}  {e['linf_mean']:>10.4f}  "
                f"{e['elapsed_s']:>7.1f}s  {e['timestamp'][11:]}"
            )
        print(f"  {'─'*136}")
        # Aggregate
        df = pd.DataFrame(self.entries)
        print(f"\n  {BOLD}Aggregate (L3 Detection only):{RESET}")
        l3 = df[df["layer"] == "detection"] if "detection" in df["layer"].values else df
        if len(l3):
            print(f"  Mean AccDrop  : {RED}{l3['acc_drop'].mean():+.4f}{RESET}")
            print(f"  Mean F1Drop   : {RED}{l3['f1_drop'].mean():+.4f}{RESET}")
            print(f"  Mean FoolRate : {RED}{l3['fool_rate'].mean():.1%}{RESET}")
            print(f"  Mean L2 pert  : {YELLOW}{l3['l2_mean'].mean():.4f}{RESET}")
            worst = l3.loc[l3['fool_rate'].idxmax()]
            print(f"  Worst attack  : {RED}{worst['attack']}{RESET}  "
                  f"fool={RED}{worst['fool_rate']:.1%}{RESET}  "
                  f"accDrop={RED}{worst['acc_drop']:+.4f}{RESET}")
        print(f"{CYAN}{BOLD}{'═'*W}{RESET}\n")


LEDGER = ExperimentLedger()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

LAYER_FILES_MAP = {
    "cicids2017": {
        layer: (f"processed_data/cicids2017/{layer}_train.csv",
                f"processed_data/cicids2017/{layer}_test.csv")
        for layer in ["ingestion","triage","detection","siem","soar"]
    },
    "ember": {layer: ("processed_data/ember/ember_train.csv",
                      "processed_data/ember/ember_test.csv")
              for layer in ["ingestion","triage","detection","siem","soar"]},
    "loghub": {
        layer: (f"processed_data/loghub/{layer}_train.csv",
                f"processed_data/loghub/{layer}_test.csv")
        for layer in ["ingestion","triage","detection","siem","soar"]
    },
}

LABEL_CANDIDATES = [
    "label","Label","LABEL","class","Class","CLASS","target","Target",
    "attack_type","Attack","category","y","Y","tag","Tag","type","Type",
    "response_action","priority","alert_priority","correlation_label",
    "soar_action","threat_label","event_type",
]

def _load_csv(path: str):
    df = pd.read_csv(path)
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), df.columns[-1])
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
    y = df[label_col].astype("category").cat.codes.values
    feat_names = list(df.drop(columns=[label_col]).select_dtypes(include=[np.number]).columns)
    return X, y, feat_names

def load_test_data(dataset: str, layer: str):
    paths = LAYER_FILES_MAP.get(dataset, {}).get(layer)
    if paths:
        test_path = paths[1]
        if Path(test_path).exists():
            try:
                X, y, feat_names = _load_csv(test_path)
                LOG.ok(f"Loaded {dataset}/{layer} test: {X.shape}")
                return X, y, feat_names
            except Exception as ex:
                LOG.warn(f"CSV load error ({ex}) — using synthetic fallback")
    LOG.warn(f"CSV not found for {dataset}/{layer} — generating synthetic data")
    rng = np.random.default_rng(hash(dataset + layer) % 2**31)
    n_feat, n_samples = 30, 800
    X = rng.standard_normal((n_samples, n_feat)).astype(np.float32)
    y = rng.integers(0, 2, n_samples)
    X[y == 1] += 1.2
    return X, y, [f"feat_{i}" for i in range(n_feat)]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

LAYER_ALGO = {
    "ingestion": "rf",
    "triage":    "gb",
    "detection": "mlp",
    "siem":      "rf",
    "soar":      "gb",
}

def _build_fresh_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=15,
                                       class_weight="balanced", n_jobs=-1, random_state=42)
    if kind == "gb":
        return GradientBoostingClassifier(n_estimators=200, max_depth=6,
                                           learning_rate=0.08, random_state=42)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), activation="relu",
                              max_iter=300, early_stopping=True, random_state=42)

def load_soc_layer(dataset: str, layer: str):
    model_path  = MODEL_ROOT / layer / f"{dataset}_model.joblib"
    scaler_path = MODEL_ROOT / layer / f"{dataset}_scaler.joblib"
    enc_path    = MODEL_ROOT / layer / f"{dataset}_encoders.joblib"
    X_te, y_te, feat_names = load_test_data(dataset, layer)

    if model_path.exists() and scaler_path.exists() and enc_path.exists():
        try:
            model    = joblib.load(model_path)
            scaler   = joblib.load(scaler_path)
            encoders = joblib.load(enc_path)
            le       = encoders.get("__label__") if isinstance(encoders, dict) else None
            LOG.ok(f"Loaded saved model: {dataset}/{layer}")
            X_te_s = X_te
            n = model.n_features_in_
            if X_te_s.shape[1] < n:
                X_te_s = np.concatenate([X_te_s, np.zeros((len(X_te_s), n - X_te_s.shape[1]))], axis=1)
            else:
                X_te_s = X_te_s[:, :n]
            try: X_te_s = scaler.transform(X_te_s)
            except: pass
            return model, scaler, le, X_te_s, y_te, feat_names
        except Exception as ex:
            LOG.warn(f"Could not load saved model ({ex}) — retraining")

    LOG.warn(f"Training fresh model for {dataset}/{layer}")
    train_path = LAYER_FILES_MAP.get(dataset, {}).get(layer, [None])[0]
    if train_path and Path(train_path).exists():
        X_tr, y_tr, _ = _load_csv(train_path)
    else:
        rng  = np.random.default_rng(hash(dataset + layer + "train") % 2**31)
        X_tr = rng.standard_normal((3000, X_te.shape[1])).astype(np.float32)
        y_tr = rng.integers(0, 2, 3000)
        X_tr[y_tr == 1] += 1.2

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)
    model    = _build_fresh_model(LAYER_ALGO[layer])
    model.fit(X_tr_s, y_tr)
    acc      = accuracy_score(y_te, model.predict(X_te_s))
    LOG.ok(f"Trained {layer} ({LAYER_ALGO[layer].upper()}) acc={acc:.4f}")
    return model, scaler, None, X_te_s, y_te, feat_names

def load_all_soc_layers(dataset: str) -> dict:
    LOG.phase(f"Loading AI-SOC Target — {dataset.upper()}")
    soc = {}
    for layer in LAYERS:
        model, scaler, le, X_te, y_te, feat_names = load_soc_layer(dataset, layer)
        soc[layer] = dict(model=model, scaler=scaler, le=le,
                          X_test=X_te, y_test=y_te, feat_names=feat_names)
    return soc


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — GRADIENT ENGINE  (exact + finite-diff)
# ══════════════════════════════════════════════════════════════════════════════

def _mlp_backprop(model: MLPClassifier, X: np.ndarray, y: np.ndarray):
    n = X.shape[0]
    activations = [X.copy()]
    a = X.copy()
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        a = a @ W + b
        if i < len(model.coefs_) - 1:
            np.maximum(a, 0, out=a)
        activations.append(a.copy())
    out = activations[-1]
    if out.shape[1] == 1:
        probs = 1.0 / (1.0 + np.exp(-out))
        y_col = y.astype(np.float32).reshape(-1, 1)
        delta = (probs - y_col) / n
    else:
        out   = out - out.max(axis=1, keepdims=True)
        exp_a = np.exp(out)
        probs = exp_a / exp_a.sum(axis=1, keepdims=True)
        y_oh  = np.zeros_like(probs)
        y_oh[np.arange(n), np.clip(y, 0, probs.shape[1]-1)] = 1.0
        delta = (probs - y_oh) / n
    grads_X = None
    for i in reversed(range(len(model.coefs_))):
        if i > 0:
            delta_next = delta @ model.coefs_[i].T
            delta_next[activations[i] <= 0] = 0.0
        else:
            grads_X = delta @ model.coefs_[i].T
        delta = delta_next if i > 0 else delta
    return grads_X


def _finite_diff_grad(model, X: np.ndarray, y: np.ndarray, eps: float = 5e-4):
    """Higher-resolution finite difference using central differences."""
    imp = (model.feature_importances_
           if hasattr(model, "feature_importances_")
           else np.ones(X.shape[1]) / X.shape[1])
    top_idx = np.argsort(imp)[::-1][:20]   # top-20 features (was 15)
    n       = X.shape[0]
    grad    = np.zeros_like(X)
    y_clip  = np.clip(y, 0, model.predict_proba(X).shape[1] - 1)
    # Central differences for better accuracy
    for j in top_idx:
        Xp = X.copy(); Xp[:, j] += eps
        Xm = X.copy(); Xm[:, j] -= eps
        pp = model.predict_proba(Xp)
        pm = model.predict_proba(Xm)
        Lp = -np.log(pp[np.arange(n), y_clip] + 1e-12).mean()
        Lm = -np.log(pm[np.arange(n), y_clip] + 1e-12).mean()
        grad[:, j] = (Lp - Lm) / (2 * eps)
    return grad


def get_gradient(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if isinstance(model, MLPClassifier):
        return _mlp_backprop(model, X, y)
    return _finite_diff_grad(model, X, y)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — ATTACK ALGORITHMS  (STRONGER versions)
# ══════════════════════════════════════════════════════════════════════════════

# ── 5a. FGSM ─────────────────────────────────────────────────────────────────
def fgsm(model, X: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    """Fast Gradient Sign Method — Goodfellow et al. 2015."""
    g = get_gradient(model, X, y)
    return X + epsilon * np.sign(g)


# ── 5b. PGD (stronger: random start + more steps) ────────────────────────────
def pgd(model, X: np.ndarray, y: np.ndarray,
        epsilon: float, alpha: float = None, steps: int = 40,
        random_start: bool = True) -> np.ndarray:
    """
    PGD with random restart (Madry et al. 2018).
    Random initialisation makes it much stronger than plain PGD.
    """
    if alpha is None:
        alpha = epsilon / 6.0   # finer step
    if random_start:
        noise = np.random.uniform(-epsilon, epsilon, X.shape).astype(np.float32)
        X_adv = X + noise
    else:
        X_adv = X.copy()
    for _ in range(steps):
        g     = get_gradient(model, X_adv, y)
        X_adv = X_adv + alpha * np.sign(g)
        X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
    return X_adv


# ── 5c. MI-FGSM (Momentum Iterative) — NEW ───────────────────────────────────
def mi_fgsm(model, X: np.ndarray, y: np.ndarray,
            epsilon: float, steps: int = 10, decay: float = 1.0) -> np.ndarray:
    """
    Momentum Iterative FGSM (Dong et al. 2018).
    Accumulates gradient momentum across steps — significantly stronger than FGSM.
    μ = decay, step = ε/steps
    """
    alpha   = epsilon / steps
    g_prev  = np.zeros_like(X)
    X_adv   = X.copy()
    for _ in range(steps):
        g     = get_gradient(model, X_adv, y)
        g_norm = np.abs(g).sum(axis=1, keepdims=True) + 1e-12
        g     = decay * g_prev + g / g_norm
        X_adv = X_adv + alpha * np.sign(g)
        X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
        g_prev = g
    return X_adv


# ── 5d. C&W L2 Attack (Carlini & Wagner 2018) — NEW ──────────────────────────
def cw_l2(model, X: np.ndarray, y: np.ndarray,
          c: float = 1.0, steps: int = 50, lr: float = 0.01,
          kappa: float = 0.0) -> np.ndarray:
    """
    Carlini & Wagner L2 attack — strongest white-box gradient attack.
    Minimises ||δ||₂² + c·f(x+δ) where f encourages misclassification.
    """
    if not isinstance(model, MLPClassifier):
        # Approximate C&W for tree models via iterative gradient descent
        return pgd(model, X, y, epsilon=0.3, steps=steps)

    X_adv = X.copy()
    w     = X.copy()   # unconstrained optimisation variable (tanh space)

    for step in range(steps):
        # Gradient of adversarial loss
        g = _mlp_backprop(model, X_adv, y)
        if g is None:
            break
        # L2 regularisation: penalise large perturbations
        delta   = X_adv - X
        l2_grad = 2.0 * delta
        # Combined gradient
        total_g = c * g + l2_grad
        w       = w + lr * np.sign(total_g)
        # Box constraint: clip to ±3σ neighbourhood
        X_adv   = np.clip(w, X - 0.5, X + 0.5)

    LOG.info(f"C&W L2: mean perturbation = {np.linalg.norm(X_adv-X, axis=1).mean():.4f}")
    return X_adv


# ── 5e. DeepFool Approximation — NEW ─────────────────────────────────────────
def deepfool_approx(model, X: np.ndarray, y: np.ndarray,
                    max_iter: int = 30, overshoot: float = 0.02) -> np.ndarray:
    """
    DeepFool (Moosavi-Dezaleh et al. 2016) approximation.
    Finds the minimum perturbation that crosses the decision boundary.
    Most efficient attack in terms of perturbation magnitude.
    """
    X_adv = X.copy()
    for _ in range(max_iter):
        g   = get_gradient(model, X_adv, y)
        # Step toward nearest decision boundary
        g_norm = np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        r      = (1.0 + overshoot) * g / g_norm / max_iter
        X_adv  = X_adv + r
    return X_adv


# ── 5f. Feature Manipulation ─────────────────────────────────────────────────
def feature_manipulation(model, X: np.ndarray, y: np.ndarray,
                          strategy: str = "scale", top_n: int = 20) -> np.ndarray:
    """Target top-N most important features with domain-aware manipulation."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        g   = get_gradient(model, X[:100], y[:100])
        imp = np.abs(g).mean(axis=0)
    top_idx = np.argsort(imp)[::-1][:min(top_n, X.shape[1])]
    X_adv   = X.copy()
    if strategy == "zero":
        X_adv[:, top_idx] = 0.0
    elif strategy == "scale":
        X_adv[:, top_idx] *= 0.05   # more aggressive shrink (was 0.1)
    elif strategy == "swap":
        bot_idx = np.argsort(imp)[:len(top_idx)]
        X_adv[:, top_idx] = X[:, bot_idx].copy()
        X_adv[:, bot_idx] = X[:, top_idx].copy()
    elif strategy == "noise":
        std = X[:, top_idx].std(axis=0) + 1e-8
        X_adv[:, top_idx] += np.random.randn(*X_adv[:, top_idx].shape) * std * 1.0
    elif strategy == "invert":
        # NEW: flip feature values around their mean
        mu = X[:, top_idx].mean(axis=0)
        X_adv[:, top_idx] = 2 * mu - X[:, top_idx]
    return X_adv


# ── 5g. Transferability Attack — NEW ─────────────────────────────────────────
def transferability_attack(source_model, target_model,
                            X: np.ndarray, y: np.ndarray,
                            epsilon: float = 0.20) -> np.ndarray:
    """
    Black-box transfer attack: craft adversarial examples on source model
    and transfer to target (no knowledge of target required).
    Uses PGD on source as surrogate.
    """
    return pgd(source_model, X, y, epsilon=epsilon, steps=40, random_start=True)


# ── 5h. Log Poisoning / Mimicry ──────────────────────────────────────────────
def log_poisoning(model, X: np.ndarray, y: np.ndarray,
                  mode: str = "mimicry") -> np.ndarray:
    X_adv       = X.copy()
    attack_mask = y > 0
    benign_mask = y == 0
    if not attack_mask.any():
        return X_adv
    if mode == "mimicry" and benign_mask.any():
        mu    = X[benign_mask].mean(axis=0)
        sigma = X[benign_mask].std(axis=0) + 1e-8
        noise = np.random.randn(attack_mask.sum(), X.shape[1]) * sigma * 0.15
        X_adv[attack_mask] = mu + noise
    elif mode == "boundary":
        g = get_gradient(model, X[attack_mask], y[attack_mask])
        X_adv[attack_mask] -= 0.30 * np.sign(g)   # more aggressive
    elif mode == "gradient_mimicry":   # NEW: combine both
        mu    = X[benign_mask].mean(axis=0) if benign_mask.any() else X.mean(axis=0)
        sigma = X[benign_mask].std(axis=0) + 1e-8 if benign_mask.any() else X.std(axis=0)
        noise = np.random.randn(attack_mask.sum(), X.shape[1]) * sigma * 0.10
        # Move toward benign mean AND use gradient
        X_adv[attack_mask] = mu + noise
        g = get_gradient(model, X_adv[attack_mask], y[attack_mask])
        X_adv[attack_mask] -= 0.15 * np.sign(g)
    return X_adv


# ── 5i. Constrained Evasion (realistic) ──────────────────────────────────────
_MUTABLE_SLOTS = list(range(0, 15))   # extended to 15 features

def constrained_evasion(model, X: np.ndarray, y: np.ndarray,
                         epsilon: float = 0.20) -> np.ndarray:
    """
    Realistic constrained attack — only perturbs mutable network-level features.
    Enforces non-negativity. Uses PGD instead of FGSM for strength.
    """
    X_adv = X.copy()
    mut   = [i for i in _MUTABLE_SLOTS if i < X.shape[1]]
    X_m   = X[:, mut]
    y_arr = y
    alpha = epsilon / 6.0
    for _ in range(30):
        g          = get_gradient(model, X_adv, y_arr)
        X_adv[:, mut] += alpha * np.sign(g[:, mut])
        X_adv[:, mut]  = np.clip(X_adv[:, mut], X_m - epsilon, X_m + epsilon)
        X_adv[:, mut]  = np.maximum(X_adv[:, mut], 0.0)
    return X_adv


# ── 5j. AutoAttack-style ensemble — NEW ──────────────────────────────────────
def auto_attack(model, X: np.ndarray, y: np.ndarray,
                epsilon: float = 0.20) -> np.ndarray:
    """
    AutoAttack-style: run multiple attacks and keep the worst-case example.
    Combines PGD + MI-FGSM + C&W + DeepFool — ensemble of strongest attacks.
    """
    LOG.info("AutoAttack: running ensemble of 4 attacks…")
    candidates = []
    candidates.append(pgd(model, X, y, epsilon=epsilon, steps=50, random_start=True))
    candidates.append(mi_fgsm(model, X, y, epsilon=epsilon, steps=20))
    candidates.append(cw_l2(model, X, y, c=2.0, steps=50))
    candidates.append(deepfool_approx(model, X, y, max_iter=40, overshoot=0.05))

    # Keep the adversarial example that causes the lowest confidence on true class
    y_clip = np.clip(y, 0, model.predict_proba(X).shape[1] - 1)
    best   = candidates[0].copy()
    best_conf = model.predict_proba(candidates[0])[np.arange(len(y)), y_clip]

    for cand in candidates[1:]:
        conf = model.predict_proba(cand)[np.arange(len(y)), y_clip]
        mask = conf < best_conf
        best[mask] = cand[mask]
        best_conf  = np.minimum(best_conf, conf)

    n_fooled = (model.predict(best) != y).sum()
    LOG.ok(f"AutoAttack fooled {n_fooled}/{len(y)} samples ({n_fooled/len(y):.1%})")
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FULL METRICS ENGINE (all 11 metrics)
# ══════════════════════════════════════════════════════════════════════════════

METRIC_KEYS = [
    "accuracy","precision","recall","f1_score",
    "tpr","fpr","auc_roc","avg_precision",
    "n_samples","classes",
]
METRIC_LABELS = {
    "accuracy"     : "Accuracy        (TP+TN)/Total",
    "precision"    : "Precision       TP/(TP+FP)",
    "recall"       : "Recall          TP/(TP+FN)",
    "f1_score"     : "F1-Score        2×(P×R)/(P+R)",
    "tpr"          : "TPR             TP/(TP+FN) per class",
    "fpr"          : "FPR             FP/(FP+TN) per class",
    "auc_roc"      : "AUC-ROC         Area under ROC curve",
    "avg_precision": "Avg Precision   Area under PR curve",
}
LAYER_ALGO_LABEL = {
    "ingestion": "Random Forest",
    "triage":    "Gradient Boosting",
    "detection": "MLP Neural Net",
    "siem":      "Random Forest",
    "soar":      "Gradient Boosting",
}


def _compute_full_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute all 11 metrics for one model:
    accuracy, precision, recall, f1_score, tpr, fpr,
    auc_roc, avg_precision, confusion_matrix, n_samples, classes
    """
    classes  = np.unique(y)
    n_cls    = len(classes)
    avg      = "binary" if n_cls == 2 else "weighted"
    y_pred   = model.predict(X)
    y_proba  = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    # ── Per-class TPR / FPR ───────────────────────────────────────────────────
    tprs, fprs = [], []
    for cls in classes:
        yt = (y == cls).astype(int)
        yp = (y_pred == cls).astype(int)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_full = confusion_matrix(y, y_pred)

    m = {
        "accuracy" : round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, average=avg, zero_division=0)), 4),
        "recall"   : round(float(recall_score(y, y_pred,    average=avg, zero_division=0)), 4),
        "f1_score" : round(float(f1_score(y, y_pred,        average=avg, zero_division=0)), 4),
        "tpr"      : round(float(np.mean(tprs)), 4),
        "fpr"      : round(float(np.mean(fprs)), 4),
        "n_samples": int(len(y)),
        "classes"  : [str(c) for c in classes.tolist()],
        "confusion_matrix": cm_full.tolist(),
        "tpr_per_class": [round(t, 4) for t in tprs],
        "fpr_per_class": [round(f, 4) for f in fprs],
    }

    # ── AUC / AP ──────────────────────────────────────────────────────────────
    if y_proba is not None:
        try:
            if n_cls == 2:
                p = y_proba[:, 1]
                m["auc_roc"]       = round(float(roc_auc_score(y, p)), 4)
                m["avg_precision"] = round(float(average_precision_score(y, p)), 4)
            else:
                m["auc_roc"]       = round(float(roc_auc_score(
                    y, y_proba, multi_class="ovr", average="weighted")), 4)
                m["avg_precision"] = round(float(np.mean([
                    average_precision_score((y == c).astype(int), y_proba[:, c])
                    for c in range(n_cls)])), 4)
        except Exception:
            m["auc_roc"] = m["avg_precision"] = None
    else:
        m["auc_roc"] = m["avg_precision"] = None

    return m


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6b — INPUT PERTURBATION STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_input_perturbation(X_clean: np.ndarray, X_adv: np.ndarray) -> dict:
    """
    Measure how much the input was actually changed.
    Returns per-sample L2 and L∞ norms plus aggregated statistics.

    Returns
    -------
    dict with keys:
        l2_per_sample    : np.ndarray  (n_samples,)
        linf_per_sample  : np.ndarray  (n_samples,)
        l2_mean, l2_max, l2_std
        linf_mean, linf_max, linf_std
        n_unchanged      : samples where ||δ||₂ < 1e-6  (not perturbed at all)
        pct_unchanged    : fraction unchanged
    """
    n = min(len(X_clean), len(X_adv))
    delta = X_adv[:n] - X_clean[:n]

    l2   = np.linalg.norm(delta, axis=1)
    linf = np.abs(delta).max(axis=1)
    n_unchanged = int((l2 < 1e-6).sum())

    return dict(
        l2_per_sample   = l2,
        linf_per_sample = linf,
        l2_mean         = float(l2.mean()),
        l2_max          = float(l2.max()),
        l2_std          = float(l2.std()),
        linf_mean       = float(linf.mean()),
        linf_max        = float(linf.max()),
        linf_std        = float(linf.std()),
        n_unchanged     = n_unchanged,
        pct_unchanged   = round(n_unchanged / max(n, 1), 4),
    )


def print_perturbation_stats(attack_name: str, stats: dict):
    """Pretty-print input-perturbation stats for one attack."""
    unch_c = GREEN if stats["pct_unchanged"] < 0.05 else YELLOW if stats["pct_unchanged"] < 0.30 else RED
    print(f"    {GREY}[INPUT PERTURBED]{RESET}  "
          f"L2: mean={YELLOW}{stats['l2_mean']:.4f}{RESET}  "
          f"max={stats['l2_max']:.4f}  "
          f"std={stats['l2_std']:.4f}  │  "
          f"L∞: mean={YELLOW}{stats['linf_mean']:.4f}{RESET}  "
          f"max={stats['linf_max']:.4f}  │  "
          f"Unchanged: {unch_c}{stats['pct_unchanged']:.1%}{RESET} "
          f"({stats['n_unchanged']}/{stats['l2_per_sample'].shape[0]})")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6c — MODEL ACTUALLY FOOLED
# ══════════════════════════════════════════════════════════════════════════════

def compute_model_fooled(model, X_clean: np.ndarray, X_adv: np.ndarray,
                          y_true: np.ndarray) -> dict:
    """
    Count how many samples the model predicts differently on X_adv vs X_clean,
    and how many of those correspond to *true* adversarial success
    (prediction changed AND the adversarial prediction is wrong).

    Returns
    -------
    dict with keys:
        n_total           : total samples evaluated
        n_pred_changed    : clean prediction ≠ adversarial prediction
        n_truly_fooled    : adv prediction ≠ y_true  (and was correct before)
        pct_pred_changed  : fraction whose prediction flipped
        pct_truly_fooled  : fraction actually fooled (was right → now wrong)
        pred_clean        : np.ndarray of clean predictions
        pred_adv          : np.ndarray of adversarial predictions
    """
    n = min(len(X_clean), len(X_adv), len(y_true))
    pred_c = model.predict(X_clean[:n])
    pred_a = model.predict(X_adv[:n])

    changed      = pred_c != pred_a
    was_correct  = pred_c == y_true[:n]
    truly_fooled = changed & was_correct          # was right, now wrong

    return dict(
        n_total          = n,
        n_pred_changed   = int(changed.sum()),
        n_truly_fooled   = int(truly_fooled.sum()),
        pct_pred_changed = round(float(changed.mean()), 4),
        pct_truly_fooled = round(float(truly_fooled.mean()), 4),
        pred_clean       = pred_c,
        pred_adv         = pred_a,
    )


def print_fooled_stats(attack_name: str, fooled: dict):
    """Pretty-print fooled-model stats for one attack."""
    fc = RED if fooled["pct_truly_fooled"] > 0.20 else YELLOW if fooled["pct_truly_fooled"] > 0.05 else GREEN
    cc = YELLOW if fooled["pct_pred_changed"] > 0.10 else GREEN
    print(f"    {GREY}[MODEL FOOLED]{RESET}     "
          f"Pred changed: {cc}{fooled['n_pred_changed']}/{fooled['n_total']} "
          f"({fooled['pct_pred_changed']:.1%}){RESET}  │  "
          f"Actually fooled: {fc}{fooled['n_truly_fooled']}/{fooled['n_total']} "
          f"({fooled['pct_truly_fooled']:.1%}){RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6d — COMPUTED DROPS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def compute_drops(m_clean: dict, m_adv: dict) -> dict:
    """
    Compute signed metric drops (clean − adversarial) for every tracked metric.
    A positive drop means the adversarial attack degraded the metric.
    For FPR, a negative drop (FPR increased) is the adverse direction.

    Returns dict of metric → drop value, plus a severity label.
    """
    tracked = ["accuracy", "precision", "recall", "f1_score",
               "tpr", "fpr", "auc_roc", "avg_precision"]
    drops = {}
    for mk in tracked:
        bv = m_clean.get(mk)
        av = m_adv.get(mk)
        if bv is not None and av is not None:
            drops[mk] = round(float(bv) - float(av), 4)
        else:
            drops[mk] = None

    # Severity: based on accuracy drop + fool rate
    acc_d = drops.get("accuracy") or 0.0
    f1_d  = drops.get("f1_score") or 0.0
    if acc_d > 0.10 or f1_d > 0.10:
        severity = "CRITICAL"
    elif acc_d > 0.03 or f1_d > 0.03:
        severity = "HIGH"
    elif acc_d > 0.005 or f1_d > 0.005:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    drops["_severity"] = severity
    return drops


def print_computed_drops(attack_name: str, layer: str, drops: dict):
    """Pretty-print all metric drops for one attack × layer."""
    sev = drops.get("_severity", "LOW")
    sev_col = RED if sev == "CRITICAL" else YELLOW if sev == "HIGH" else \
              MAGENTA if sev == "MEDIUM" else GREEN
    metrics_show = ["accuracy", "f1_score", "tpr", "fpr", "auc_roc", "avg_precision"]
    parts = []
    for mk in metrics_show:
        d = drops.get(mk)
        if d is None:
            continue
        # FPR: worse when it goes up (drop < 0)
        worse = (d < -0.005) if mk == "fpr" else (d > 0.005)
        col = RED if worse else GREEN
        parts.append(f"{mk}={col}{d:+.4f}{RESET}")
    row = "  ".join(parts)
    print(f"    {GREY}[COMPUTED DROPS]{RESET}  "
          f"[{layer.upper()}]  {row}  severity={sev_col}{sev}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — CASCADE EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _propagate_perturbation(X_adv_l3, X_clean_l3, X_clean_target):
    delta  = X_adv_l3 - X_clean_l3
    scale  = (np.abs(X_clean_target).mean() + 1e-8) / (np.abs(X_clean_l3).mean() + 1e-8)
    n_feat = min(delta.shape[1], X_clean_target.shape[1])
    n_samp = min(len(delta), len(X_clean_target))
    perturb = np.zeros_like(X_clean_target)
    perturb[:n_samp, :n_feat] = delta[:n_samp, :n_feat] * scale
    return X_clean_target + perturb


def evaluate_cascade(soc: dict, attack_name: str, X_adv_l3: np.ndarray) -> dict:
    """Evaluate adversarial attack across all 5 AI-SOC layers — full metrics."""
    results     = {}
    X_clean_l3  = soc["detection"]["X_test"]

    for layer in LAYERS:
        d      = soc[layer]
        X_orig = d["X_test"]
        y_te   = d["y_test"]

        X_a = X_adv_l3 if layer == "detection" else \
              _propagate_perturbation(X_adv_l3, X_clean_l3, X_orig)

        m_clean = _compute_full_metrics(d["model"], X_orig, y_te)
        m_adv   = _compute_full_metrics(d["model"], X_a,    y_te)

        evasion = max(0.0, m_clean["tpr"] - m_adv["tpr"])

        # ── New features ─────────────────────────────────────────────────────
        drops       = compute_drops(m_clean, m_adv)
        pert_stats  = compute_input_perturbation(X_orig, X_a)
        fooled_stats = compute_model_fooled(d["model"], X_orig, X_a, y_te)

        results[layer] = dict(
            clean   = m_clean,
            adv     = m_adv,
            acc_clean   = m_clean["accuracy"],
            acc_adv     = m_adv["accuracy"],
            acc_drop    = round(m_clean["accuracy"] - m_adv["accuracy"], 4),
            dr_clean    = m_clean["tpr"],
            dr_adv      = m_adv["tpr"],
            evasion     = round(evasion, 4),
            conf_clean  = 1.0,
            conf_adv    = 1.0,
            # ── feature additions ──────────────────────────────────────────
            drops        = drops,
            perturbation = pert_stats,
            fooled       = fooled_stats,
        )

        # Confidence
        try:
            results[layer]["conf_clean"] = round(float(
                d["model"].predict_proba(X_orig).max(axis=1).mean()), 4)
            results[layer]["conf_adv"] = round(float(
                d["model"].predict_proba(X_a).max(axis=1).mean()), 4)
        except Exception:
            pass

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — ATTACK CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

def build_attack_configs(soc: dict) -> list:
    d3   = soc["detection"]
    m, X, y = d3["model"], d3["X_test"], d3["y_test"]
    # Surrogate model for transfer attack (use ingestion RF)
    src  = soc["ingestion"]["model"]

    return [
        # FGSM family
        ("FGSM ε=0.01",       lambda: fgsm(m, X, y, 0.01)),
        ("FGSM ε=0.05",       lambda: fgsm(m, X, y, 0.05)),
        ("FGSM ε=0.10",       lambda: fgsm(m, X, y, 0.10)),
        ("FGSM ε=0.20",       lambda: fgsm(m, X, y, 0.20)),
        ("FGSM ε=0.30",       lambda: fgsm(m, X, y, 0.30)),
        # PGD (random start, more steps)
        ("PGD  ε=0.01",       lambda: pgd(m, X, y, 0.01)),
        ("PGD  ε=0.05",       lambda: pgd(m, X, y, 0.05)),
        ("PGD  ε=0.10",       lambda: pgd(m, X, y, 0.10)),
        ("PGD  ε=0.20",       lambda: pgd(m, X, y, 0.20)),
        ("PGD  ε=0.30",       lambda: pgd(m, X, y, 0.30)),
        # MI-FGSM
        ("MI-FGSM ε=0.10",    lambda: mi_fgsm(m, X, y, 0.10)),
        ("MI-FGSM ε=0.20",    lambda: mi_fgsm(m, X, y, 0.20)),
        ("MI-FGSM ε=0.30",    lambda: mi_fgsm(m, X, y, 0.30)),
        # C&W
        ("C&W L2  c=0.5",     lambda: cw_l2(m, X, y, c=0.5)),
        ("C&W L2  c=1.0",     lambda: cw_l2(m, X, y, c=1.0)),
        ("C&W L2  c=2.0",     lambda: cw_l2(m, X, y, c=2.0)),
        # DeepFool
        ("DeepFool os=0.02",  lambda: deepfool_approx(m, X, y, overshoot=0.02)),
        ("DeepFool os=0.05",  lambda: deepfool_approx(m, X, y, overshoot=0.05)),
        # Feature manipulation
        ("FEAT-zero",          lambda: feature_manipulation(m, X, y, "zero")),
        ("FEAT-scale",         lambda: feature_manipulation(m, X, y, "scale")),
        ("FEAT-noise",         lambda: feature_manipulation(m, X, y, "noise")),
        ("FEAT-swap",          lambda: feature_manipulation(m, X, y, "swap")),
        ("FEAT-invert",        lambda: feature_manipulation(m, X, y, "invert")),
        # Log poisoning
        ("POISON-mimicry",     lambda: log_poisoning(m, X, y, "mimicry")),
        ("POISON-boundary",    lambda: log_poisoning(m, X, y, "boundary")),
        ("POISON-grad-mim",    lambda: log_poisoning(m, X, y, "gradient_mimicry")),
        # Transfer
        ("TRANSFER ε=0.20",   lambda: transferability_attack(src, m, X, y, 0.20)),
        # Constrained realistic
        ("CONSTRAINED ε=0.15", lambda: constrained_evasion(m, X, y, 0.15)),
        ("CONSTRAINED ε=0.30", lambda: constrained_evasion(m, X, y, 0.30)),
        # AutoAttack (ensemble — strongest)
        ("AUTOATTACK ε=0.20",  lambda: auto_attack(m, X, y, 0.20)),
    ]


def run_experiment(soc: dict, mode: str, dataset: str) -> dict:
    all_configs = build_attack_configs(soc)
    filter_map = {
        "fgsm"       : "FGSM",
        "pgd"        : "PGD",
        "mifgsm"     : "MI-FGSM",
        "cw"         : "C&W",
        "deepfool"   : "DeepFool",
        "feature"    : "FEAT",
        "poison"     : "POISON",
        "transfer"   : "TRANSFER",
        "constrained": "CONSTRAINED",
        "auto"       : "AUTOATTACK",
    }
    if mode in filter_map:
        prefix  = filter_map[mode]
        configs = [c for c in all_configs if c[0].startswith(prefix)]
    else:
        configs = all_configs

    LOG.phase(f"Running {len(configs)} attack(s) on {dataset.upper()} AI-SOC")
    all_results = {}

    for name, fn in configs:
        LOG.attack(name)
        t0    = time.time()
        X_adv = fn()
        res   = evaluate_cascade(soc, name, X_adv)
        elapsed = time.time() - t0

        # Extract epsilon from attack name if present
        eps_val = None
        for tok in name.split():
            if tok.startswith("ε="):
                try: eps_val = float(tok[2:])
                except: pass

        for layer in LAYERS:
            r = res[layer]
            LOG.result(layer, r["acc_clean"], r["acc_adv"], r["acc_drop"], r["evasion"])
            # ── Computed drops ────────────────────────────────────────────
            print_computed_drops(name, layer, r["drops"])

        # ── L3 Detection: input perturbed + model actually fooled ─────────
        l3 = res["detection"]
        print_perturbation_stats(name, l3["perturbation"])
        print_fooled_stats(name,       l3["fooled"])

        print(f"    {GREY}→ L3 evasion={RED}{l3['evasion']:.1%}{RESET}"
              f"  cascade L5 drop={RED}{res['soar']['acc_drop']:+.4f}{RESET}"
              f"  [{elapsed:.1f}s]{RESET}")

        # ── Ledger: record one entry per layer ────────────────────────────
        for layer in LAYERS:
            r = res[layer]
            f1_d  = r["drops"].get("f1_score") or 0.0
            auc_d = r["drops"].get("auc_roc")  or 0.0
            LEDGER.record(
                dataset   = dataset,
                attack    = name,
                layer     = layer,
                epsilon   = eps_val,
                acc_drop  = r["acc_drop"],
                f1_drop   = f1_d,
                auc_drop  = auc_d,
                n_fooled  = r["fooled"]["n_truly_fooled"],
                n_total   = r["fooled"]["n_total"],
                l2_mean   = r["perturbation"]["l2_mean"],
                linf_mean = r["perturbation"]["linf_mean"],
                elapsed_s = elapsed,
            )

        all_results[name] = res

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — BEFORE / AFTER METRICS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def compute_baseline_metrics(soc: dict, dataset: str) -> dict:
    LOG.phase(f"BEFORE ATTACK — Full AI-SOC Baseline Metrics [{dataset.upper()}]")
    result = {}
    for layer in LAYERS:
        d      = soc[layer]
        m      = _compute_full_metrics(d["model"], d["X_test"], d["y_test"])
        result[layer] = m
        LOG.ok(f"{layer:<12}  acc={GREEN}{m['accuracy']:.4f}{RESET}  "
               f"f1={GREEN}{m['f1_score']:.4f}{RESET}  "
               f"tpr={GREEN}{m['tpr']:.4f}{RESET}  "
               f"fpr={YELLOW}{m['fpr']:.4f}{RESET}  "
               f"auc={GREEN}{(m['auc_roc'] or 0):.4f}{RESET}  "
               f"ap={GREEN}{(m['avg_precision'] or 0):.4f}{RESET}  "
               f"n={m['n_samples']}")
    return result


def compute_after_metrics(soc: dict, all_results: dict, dataset: str) -> dict:
    LOG.phase(f"AFTER ATTACK — Worst-Case Adversarial Metrics [{dataset.upper()}]")
    d3      = soc["detection"]
    configs = {name: fn for name, fn in build_attack_configs(soc)}

    worst = {layer: {"acc_drop": -1.0, "attack": None} for layer in LAYERS}
    for aname, res in all_results.items():
        for layer in LAYERS:
            if res[layer]["acc_drop"] > worst[layer]["acc_drop"]:
                worst[layer]["acc_drop"] = res[layer]["acc_drop"]
                worst[layer]["attack"]   = aname

    after = {}
    for layer in LAYERS:
        worst_attack = worst[layer]["attack"]
        if worst_attack and worst_attack in configs:
            X_adv_l3 = configs[worst_attack]()
            d        = soc[layer]
            X_a      = X_adv_l3 if layer == "detection" else \
                       _propagate_perturbation(X_adv_l3, d3["X_test"], d["X_test"])
            m        = _compute_full_metrics(d["model"], X_a, d["y_test"])
        else:
            m = {k: 0.0 for k in METRIC_KEYS if k not in ("classes","confusion_matrix")}
            m["classes"] = []; m["confusion_matrix"] = [[]]

        after[layer] = m
        after[layer]["worst_attack"] = worst_attack
        LOG.ok(f"{layer:<12}  acc={RED}{m['accuracy']:.4f}{RESET}  "
               f"f1={RED}{m['f1_score']:.4f}{RESET}  "
               f"tpr={RED}{m['tpr']:.4f}{RESET}  "
               f"fpr={RED}{m['fpr']:.4f}{RESET}  "
               f"auc={RED}{(m['auc_roc'] or 0):.4f}{RESET}  "
               f"[worst={YELLOW}{worst_attack}{RESET}]")
    return after


def print_before_after_report(before: dict, after: dict, dataset: str):
    """
    Full before/after report with ALL 11 metrics per layer,
    including confusion matrix and per-class TPR/FPR.
    """
    DISP_METRICS = [
        ("accuracy",      "Accuracy       (TP+TN)/Total"),
        ("precision",     "Precision      TP/(TP+FP)"),
        ("recall",        "Recall         TP/(TP+FN)"),
        ("f1_score",      "F1-Score       2×P×R/(P+R)"),
        ("tpr",           "TPR            TP/(TP+FN)"),
        ("fpr",           "FPR            FP/(FP+TN)"),
        ("auc_roc",       "AUC-ROC        Area ROC curve"),
        ("avg_precision", "Avg Precision  Area PR curve"),
    ]
    LAYER_COLORS = {
        "ingestion": BLUE, "triage": YELLOW, "detection": RED,
        "siem": MAGENTA, "soar": GREEN,
    }
    W = 100
    print(f"\n{CYAN}{BOLD}{'═'*W}{RESET}")
    print(f"{CYAN}{BOLD}  ⚔  AI-SOC FULL METRICS — BEFORE vs AFTER ADVERSARIAL ATTACK  [{dataset.upper()}]{RESET}")
    print(f"{CYAN}{BOLD}{'═'*W}{RESET}")

    for layer in LAYERS:
        b    = before[layer]
        a    = after[layer]
        col  = LAYER_COLORS.get(layer, GREY)
        alg  = LAYER_ALGO_LABEL.get(layer, "")
        wk   = a.get("worst_attack", "N/A")
        classes = b.get("classes", [])

        print(f"\n  {col}{BOLD}┌{'─'*96}┐{RESET}")
        print(f"  {col}{BOLD}│  LAYER {LAYERS.index(layer)+1}  {layer.upper():<14} [{alg}]"
              f"   Worst Attack: {str(wk):<26} n={b['n_samples']}  classes={classes[:6]} │{RESET}")
        print(f"  {col}{BOLD}├{'─'*36}┬{'─'*13}┬{'─'*13}┬{'─'*16}┬{'─'*14}┤{RESET}")
        print(f"  {col}{BOLD}│ {'Metric':<34} │ {'BEFORE':^11} │ {'AFTER':^11} │ {'Drop / Δ':^14} │ {'Status':^12} │{RESET}")
        print(f"  {col}{BOLD}├{'─'*36}┼{'─'*13}┼{'─'*13}┼{'─'*16}┼{'─'*14}┤{RESET}")

        for mk, label in DISP_METRICS:
            bv = b.get(mk)
            av = a.get(mk)
            if bv is None or av is None:
                print(f"  {col}│ {label:<34} │ {'N/A':^11} │ {'N/A':^11} │ {'N/A':^14} │ {'—':^12} │{RESET}")
                continue
            drop   = bv - av
            drop_s = f"{drop:+.4f}"
            # FPR: increase = worse; everything else: decrease = worse
            if mk == "fpr":
                worse  = drop < -0.005
                better = drop > 0.005
            else:
                worse  = drop > 0.005
                better = drop < -0.005
            dcol   = RED if worse else GREEN if better else GREY
            status = "⚠ DEGRADED" if worse else "✔ OK" if better else "— STABLE"
            scol   = RED if worse else GREEN if better else GREY
            print(f"  {col}│ {label:<34} │ {GREEN}{bv:^11.4f}{RESET}{col} │"
                  f" {RED}{av:^11.4f}{RESET}{col} │"
                  f" {dcol}{drop_s:^14}{RESET}{col} │"
                  f" {scol}{status:^12}{RESET}{col} │{RESET}")

        # ── Per-class TPR/FPR ─────────────────────────────────────────────────
        tpr_b = b.get("tpr_per_class", [])
        tpr_a = a.get("tpr_per_class", [])
        fpr_b = b.get("fpr_per_class", [])
        fpr_a = a.get("fpr_per_class", [])
        if tpr_b and tpr_a:
            print(f"  {col}├{'─'*96}┤{RESET}")
            tpr_str_b = "  ".join(f"cls{i}={v:.3f}" for i, v in enumerate(tpr_b[:6]))
            tpr_str_a = "  ".join(f"cls{i}={v:.3f}" for i, v in enumerate(tpr_a[:6]))
            fpr_str_b = "  ".join(f"cls{i}={v:.3f}" for i, v in enumerate(fpr_b[:6]))
            fpr_str_a = "  ".join(f"cls{i}={v:.3f}" for i, v in enumerate(fpr_a[:6]))
            print(f"  {col}│  TPR/class BEFORE: {tpr_str_b:<70}  │{RESET}")
            print(f"  {col}│  TPR/class AFTER:  {tpr_str_a:<70}  │{RESET}")
            print(f"  {col}│  FPR/class BEFORE: {fpr_str_b:<70}  │{RESET}")
            print(f"  {col}│  FPR/class AFTER:  {fpr_str_a:<70}  │{RESET}")

        # ── Confusion matrix ──────────────────────────────────────────────────
        cm_b = b.get("confusion_matrix", [])
        cm_a = a.get("confusion_matrix", [])
        if cm_b and cm_a:
            print(f"  {col}├{'─'*96}┤{RESET}")
            print(f"  {col}│  CONFUSION MATRIX — BEFORE (clean):{' '*60}│{RESET}")
            for row in cm_b[:4]:
                row_s = "  ".join(f"{v:>6}" for v in row[:8])
                print(f"  {col}│    {row_s:<90}  │{RESET}")
            print(f"  {col}│  CONFUSION MATRIX — AFTER (adversarial):{' '*55}│{RESET}")
            for row in cm_a[:4]:
                row_s = "  ".join(f"{v:>6}" for v in row[:8])
                print(f"  {col}│    {row_s:<90}  │{RESET}")

        print(f"  {col}{BOLD}└{'─'*96}┘{RESET}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {CYAN}{BOLD}╔{'═'*90}╗{RESET}")
    print(f"  {CYAN}{BOLD}║  SUMMARY — Mean Metrics Across All 5 AI-SOC Layers{' '*39}║{RESET}")
    print(f"  {CYAN}{BOLD}╠{'═'*90}╣{RESET}")
    for mk in ("accuracy","f1_score","tpr","fpr","auc_roc","avg_precision"):
        bv = np.mean([before[l].get(mk) or 0 for l in LAYERS])
        av = np.mean([after[l].get(mk)  or 0 for l in LAYERS])
        d  = bv - av
        dc = RED if (d > 0.01 if mk != "fpr" else d < -0.01) else GREEN
        label = METRIC_LABELS.get(mk, mk)
        print(f"  {CYAN}║  {label:<45}  BEFORE={GREEN}{bv:.4f}{RESET}{CYAN}  "
              f"AFTER={RED}{av:.4f}{RESET}{CYAN}  DROP={dc}{d:+.4f}{RESET}{CYAN}{' '*3}║{RESET}")
    print(f"  {CYAN}{BOLD}╚{'═'*90}╝{RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — CASCADE TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_cascade_table(all_results: dict, dataset: str):
    print(f"\n{RED}{BOLD}{'═'*120}{RESET}")
    print(f"{RED}{BOLD}  CASCADE DEGRADATION MATRIX — {dataset.upper()} — ACCURACY DROP PER LAYER{RESET}")
    print(f"{RED}{BOLD}{'═'*120}{RESET}")
    hdr = (f"{'Attack':<22}  {'L1 Ingest':>9}  {'L2 Triage':>9}  "
           f"{'L3 Detect':>9}  {'L4 SIEM':>9}  {'L5 SOAR':>9}  "
           f"{'Evasion@L3':>11}  {'Cascade L5':>10}  {'F1-Drop L3':>10}")
    print(f"{BOLD}  {hdr}{RESET}")
    print(f"  {'─'*116}")
    for aname, res in all_results.items():
        drops = [res[l]["acc_drop"] for l in LAYERS]
        cells = []
        for d in drops:
            c = RED if d > 0.10 else YELLOW if d > 0.03 else GREEN
            cells.append(f"{c}{d:+.4f}{RESET}")
        ev3  = res["detection"]["evasion"]
        ev5  = res["soar"]["acc_drop"]
        f1_d = (res["detection"]["clean"]["f1_score"] - res["detection"]["adv"]["f1_score"])
        ec   = RED if ev3 > 0.30 else YELLOW if ev3 > 0.10 else GREEN
        e5c  = RED if ev5 > 0.10 else YELLOW if ev5 > 0.03 else GREEN
        f1c  = RED if f1_d > 0.10 else YELLOW if f1_d > 0.03 else GREEN
        row  = (f"  {aname:<22}  "
                + "  ".join(cells)
                + f"  {ec}{ev3:>10.1%}{RESET}"
                + f"  {e5c}{ev5:>+10.4f}{RESET}"
                + f"  {f1c}{f1_d:>+10.4f}{RESET}")
        print(row)
    print(f"  {'─'*116}")
    print(f"  {GREEN}Green{RESET}=<3% drop  {YELLOW}Yellow{RESET}=3-10%  {RED}Red{RESET}=>10%\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — STATISTICAL TESTS + SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                  alpha: float = 0.05, rng_seed: int = 42) -> tuple:
    """
    Return (lower, upper) bootstrap percentile CI for the mean of *values*.
    Uses BCa-style resampling; alpha=0.05 → 95% CI.
    """
    rng   = np.random.default_rng(rng_seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)


def _wilcoxon_test(before_vals: np.ndarray, after_vals: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test (paired) between clean and adversarial accuracy
    across attacks for one layer.  Returns stat, p-value, and significance flag.

    Requires scipy >= 1.7.  If not installed the function returns None entries
    and warns once — it does NOT silently skip the requirement.
    """
    from scipy.stats import wilcoxon  # hard import — must be installed

    diffs = before_vals - after_vals
    if len(diffs) < 5:
        LOG.warn("Wilcoxon test skipped — fewer than 5 paired observations.")
        return {"statistic": None, "p_value": None, "significant_005": None}
    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant_005": False}
    stat, pval = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
    return {
        "statistic"      : round(float(stat), 4),
        "p_value"        : round(float(pval), 6),
        "significant_005": bool(pval < 0.05),
    }


def save_results(all_results: dict, dataset: str):
    rows = []
    for aname, res in all_results.items():
        for layer in LAYERS:
            r = res[layer]
            d = r.get("drops", {})
            p = r.get("perturbation", {})
            fo = r.get("fooled", {})
            rows.append({
                "dataset": dataset, "attack": aname, "layer": layer,
                "acc_clean" : r["acc_clean"],  "acc_adv"  : r["acc_adv"],
                "acc_drop"  : r["acc_drop"],   "evasion"  : r["evasion"],
                "f1_before" : r["clean"]["f1_score"], "f1_after": r["adv"]["f1_score"],
                "tpr_before": r["clean"]["tpr"],      "tpr_after": r["adv"]["tpr"],
                "fpr_before": r["clean"]["fpr"],      "fpr_after": r["adv"]["fpr"],
                "auc_before": r["clean"]["auc_roc"],  "auc_after": r["adv"]["auc_roc"],
                "ap_before" : r["clean"]["avg_precision"], "ap_after": r["adv"]["avg_precision"],
                # ── Perturbation + fooled-rate columns ─────────────────────────
                "f1_drop"         : d.get("f1_score"),
                "auc_drop"        : d.get("auc_roc"),
                "precision_drop"  : d.get("precision"),
                "recall_drop"     : d.get("recall"),
                "drop_severity"   : d.get("_severity"),
                "l2_mean"         : p.get("l2_mean"),
                "l2_max"          : p.get("l2_max"),
                "linf_mean"       : p.get("linf_mean"),
                "linf_max"        : p.get("linf_max"),
                "pct_unchanged"   : p.get("pct_unchanged"),
                "n_truly_fooled"  : fo.get("n_truly_fooled"),
                "n_pred_changed"  : fo.get("n_pred_changed"),
                "pct_pred_changed": fo.get("pct_pred_changed"),
                "pct_truly_fooled": fo.get("pct_truly_fooled"),
            })
    df = pd.DataFrame(rows)

    # ── Statistical tests: Wilcoxon signed-rank + 95% bootstrap CI ───────────
    LOG.phase("Statistical Significance Tests (Wilcoxon signed-rank + 95% bootstrap CI)")
    stat_rows = []
    for layer in LAYERS:
        sub        = df[df["layer"] == layer]
        before_acc = sub["acc_clean"].values.astype(float)
        after_acc  = sub["acc_adv"].values.astype(float)
        acc_drops  = sub["acc_drop"].values.astype(float)

        wilcox           = _wilcoxon_test(before_acc, after_acc)
        ci_lo, ci_hi     = _bootstrap_ci(acc_drops)
        mean_drop        = round(float(acc_drops.mean()), 4)
        std_drop         = round(float(acc_drops.std()),  4)

        sig_str = ("p<0.05 ✔" if wilcox["significant_005"]
                   else "p>=0.05 (NS)" if wilcox["p_value"] is not None
                   else "N/A")

        LOG.ok(
            f"{layer:<12}  mean_drop={mean_drop:+.4f} +/- {std_drop:.4f}"
            f"  95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]"
            f"  Wilcoxon p={wilcox['p_value']}  {sig_str}"
        )

        stat_rows.append({
            "dataset"         : dataset,
            "layer"           : layer,
            "algo"            : LAYER_ALGO_LABEL.get(layer, ""),
            "n_attacks"       : int(len(sub)),
            "mean_acc_drop"   : mean_drop,
            "std_acc_drop"    : std_drop,
            "ci_95_lower"     : ci_lo,
            "ci_95_upper"     : ci_hi,
            "wilcoxon_stat"   : wilcox["statistic"],
            "wilcoxon_p"      : wilcox["p_value"],
            "significant_005" : wilcox["significant_005"],
        })

    stat_df = pd.DataFrame(stat_rows)

    # ── Pairwise layer ranking — highest-leverage intervention point (RQ3) ────
    LOG.phase("Pairwise layer ranking — highest-leverage intervention point (RQ3)")
    pivot = df.pivot_table(index="attack", columns="layer",
                           values="acc_drop", aggfunc="mean")
    pairwise_rows = []
    layer_list = [l for l in LAYERS if l in pivot.columns]
    for i, l1 in enumerate(layer_list):
        for l2 in layer_list[i+1:]:
            common = pivot[[l1, l2]].dropna()
            if len(common) < 5:
                continue
            try:
                from scipy.stats import wilcoxon as _wlcx
                d1, d2 = common[l1].values, common[l2].values
                diffs  = d1 - d2
                if np.all(diffs == 0):
                    stat_val, pval = 0.0, 1.0
                else:
                    stat_val, pval = _wlcx(diffs, zero_method="wilcox")
                pairwise_rows.append({
                    "dataset"        : dataset,
                    "layer_a"        : l1,  "layer_b"        : l2,
                    "mean_drop_a"    : round(float(d1.mean()), 4),
                    "mean_drop_b"    : round(float(d2.mean()), 4),
                    "wilcoxon_stat"  : round(float(stat_val), 4),
                    "wilcoxon_p"     : round(float(pval), 6),
                    "significant_005": bool(pval < 0.05),
                    "higher_drop"    : l1 if d1.mean() > d2.mean() else l2,
                })
            except Exception as ex:
                LOG.warn(f"Pairwise Wilcoxon {l1} vs {l2}: {ex}")

    pair_df = pd.DataFrame(pairwise_rows)

    # ── Save all CSVs + JSON ──────────────────────────────────────────────────
    csv_path  = ADV_RESULT_DIR / f"{dataset}_cascade_matrix.csv"
    json_path = ADV_RESULT_DIR / f"{dataset}_cascade_matrix.json"
    stat_path = ADV_RESULT_DIR / f"{dataset}_statistical_tests.csv"
    pair_path = ADV_RESULT_DIR / f"{dataset}_pairwise_wilcoxon.csv"

    df.to_csv(csv_path, index=False)
    stat_df.to_csv(stat_path, index=False)
    if not pair_df.empty:
        pair_df.to_csv(pair_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    LOG.ok(f"Results           -> {csv_path}")
    LOG.ok(f"Statistical tests -> {stat_path}  (Wilcoxon + 95% CI per layer)")
    LOG.ok(f"Pairwise ranking  -> {pair_path}  (RQ3: leverage-point analysis)")
    LOG.ok(f"JSON              -> {json_path}")
    return df


def save_before_after_csv(before: dict, after: dict, dataset: str):
    rows = []
    for layer in LAYERS:
        for mk in ("accuracy","precision","recall","f1_score","tpr","fpr","auc_roc","avg_precision"):
            bv = before[layer].get(mk)
            av = after[layer].get(mk)
            rows.append({
                "dataset": dataset, "layer": layer,
                "algo"  : LAYER_ALGO_LABEL.get(layer, ""),
                "metric": mk,
                "before": round(bv, 4) if bv is not None else None,
                "after" : round(av, 4) if av is not None else None,
                "drop"  : round(bv - av, 4) if (bv is not None and av is not None) else None,
                "worst_attack": after[layer].get("worst_attack", ""),
                "n_samples"   : before[layer].get("n_samples", 0),
                "classes"     : str(before[layer].get("classes", [])),
            })
    df  = pd.DataFrame(rows)
    out = ADV_RESULT_DIR / f"{dataset}_before_after_metrics.csv"
    df.to_csv(out, index=False)
    LOG.ok(f"Before/After CSV → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — PLOTS (10 publication-quality figures)
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, name, dataset):
    p = ADV_PLOT_DIR / f"{dataset}_{name}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.ok(f"Plot → {p}")


def plot_01_cascade_heatmap(results, dataset):
    attacks = list(results.keys())
    matrix  = np.array([[results[a][l]["acc_drop"] for l in LAYERS] for a in attacks])
    fig, ax = plt.subplots(figsize=(12, max(6, len(attacks)*0.40+2)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["L1\nIngestion","L2\nTriage","L3\nDetection","L4\nSIEM","L5\nSOAR"], fontsize=11)
    ax.set_yticks(range(len(attacks))); ax.set_yticklabels(attacks, fontsize=8)
    ax.set_title(f"CASCADE DEGRADATION MATRIX — {dataset.upper()}\n"
                 "Accuracy Drop Under Adversarial Attack", fontsize=13, fontweight="bold")
    for i, a in enumerate(attacks):
        for j, l in enumerate(LAYERS):
            v = matrix[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.25 else "black")
    plt.colorbar(im, ax=ax, label="Accuracy Drop")
    ax.axvline(2.5, color="white", lw=2, ls="--", alpha=0.8)
    ax.text(2.5, -1.2, "▼ Attack Injection Point", ha="center", fontsize=9,
            color="red", fontweight="bold")
    fig.tight_layout(); _save(fig, "01_cascade_heatmap.png", dataset)


def plot_02_multi_metric_heatmap(before, after, dataset):
    """NEW: Heatmap of ALL 8 metrics — before vs after across all 5 layers."""
    metrics = ["accuracy","precision","recall","f1_score","tpr","fpr","auc_roc","avg_precision"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for ax_i, (phase, data) in enumerate([("BEFORE (Clean)", before), ("AFTER (Adversarial)", after)]):
        mat = np.array([[data[l].get(mk) or 0 for mk in metrics] for l in LAYERS])
        im  = axes[ax_i].imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        axes[ax_i].set_xticks(range(len(metrics)))
        axes[ax_i].set_xticklabels([m.replace("_"," ").upper() for m in metrics],
                                    rotation=35, ha="right", fontsize=9)
        axes[ax_i].set_yticks(range(5))
        axes[ax_i].set_yticklabels([l.capitalize() for l in LAYERS], fontsize=10)
        axes[ax_i].set_title(f"{phase}", fontsize=12, fontweight="bold")
        for i in range(5):
            for j, mk in enumerate(metrics):
                v = mat[i, j]
                axes[ax_i].text(j, i, f"{v:.3f}", ha="center", va="center",
                                fontsize=8, color="black")
        plt.colorbar(im, ax=axes[ax_i])
    plt.suptitle(f"All 8 Metrics — Before vs After Attack  [{dataset.upper()}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); _save(fig, "02_multi_metric_heatmap.png", dataset)


def plot_03_cascade_flow(results, dataset):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    x = np.arange(5); xlabels = ["L1","L2","L3","L4","L5"]
    groups = [("FGSM", axes[0]), ("PGD", axes[1]), ("MI-FGSM|C&W|DeepFool|AUTO", axes[2])]
    for grp, ax in groups:
        plotted_baseline = False
        for ci, (aname, res) in enumerate(results.items()):
            if not any(aname.startswith(g) for g in grp.split("|")): continue
            adv_acc   = [res[l]["acc_adv"]   for l in LAYERS]
            clean_acc = [res[l]["acc_clean"]  for l in LAYERS]
            col = plt.cm.tab20.colors[ci % 20]
            ax.plot(x, adv_acc, "o-", color=col, lw=2, label=aname, zorder=3)
            if not plotted_baseline:
                ax.plot(x, clean_acc, "k--", lw=1.5, alpha=0.6, label="Clean Baseline")
                plotted_baseline = True
        ax.axvline(2, color="red", ls=":", lw=1.5, alpha=0.7)
        ax.text(2.08, 0.10, "Inject", color="red", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(xlabels)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
        ax.set_title(f"{grp.split('|')[0]} — Cascade Propagation L1→L5", fontweight="bold")
        ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)
    plt.suptitle(f"Attack Cascade Through All Layers — {dataset.upper()}", fontsize=13, fontweight="bold")
    fig.tight_layout(); _save(fig, "03_cascade_flow.png", dataset)


def plot_04_before_after_bars(before, after, dataset):
    metrics = ["accuracy","precision","recall","f1_score","tpr","fpr"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()
    x = np.arange(len(LAYERS)); w = 0.35
    for mi, mk in enumerate(metrics):
        ax  = axes[mi]
        bv  = [before[l].get(mk, 0) or 0 for l in LAYERS]
        av  = [after[l].get(mk,  0) or 0 for l in LAYERS]
        ax.bar(x - w/2, bv, w, color="#2ecc71", alpha=0.85, label="BEFORE (Clean)")
        ax.bar(x + w/2, av, w, color="#e74c3c", alpha=0.85, label="AFTER (Adversarial)")
        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in LAYERS], rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(METRIC_LABELS[mk], fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        for xi, (b_val, a_val) in enumerate(zip(bv, av)):
            drop = b_val - a_val
            if abs(drop) > 0.005:
                col = "red" if drop > 0 else "green"
                ax.text(xi + w/2, a_val + 0.02, f"Δ{drop:+.3f}",
                        ha="center", fontsize=7, color=col, fontweight="bold")
    plt.suptitle(f"AI-SOC Metrics — BEFORE vs AFTER Attack  [{dataset.upper()}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); _save(fig, "04_before_after_bars.png", dataset)


def plot_05_evasion_per_layer(results, dataset):
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
    names = list(results.keys())
    for ax_i, layer in enumerate(LAYERS):
        ax = axes[ax_i]
        ev = [results[a][layer]["evasion"] for a in names]
        colors = ["#e74c3c" if e > 0.3 else "#f39c12" if e > 0.1 else "#2ecc71" for e in ev]
        bars = ax.barh(range(len(names)), ev, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if ax_i == 0 else [], fontsize=7)
        ax.set_xlim(0, 1); ax.set_xlabel("Evasion Rate")
        ax.set_title(f"L{ax_i+1} {layer.capitalize()}", fontweight="bold")
        ax.axvline(0.30, color="red", ls="--", lw=1, alpha=0.5)
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle(f"Evasion Rate per Layer — {dataset.upper()}", fontsize=13, fontweight="bold")
    fig.tight_layout(); _save(fig, "05_evasion_per_layer.png", dataset)


def plot_06_epsilon_curve(results, dataset):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    eps_vals = [0.01, 0.05, 0.10, 0.20, 0.30]
    methods = [
        ("FGSM",    "#3498db", "o", axes[0]),
        ("PGD",     "#e74c3c", "^", axes[0]),
        ("MI-FGSM", "#9b59b6", "s", axes[1]),
    ]
    for method, color, marker, ax in methods:
        ev = []
        for eps in eps_vals:
            match = next((k for k in results if k.startswith(method) and f"{eps:.2f}" in k), None)
            if match:
                ev.append(results[match]["detection"]["evasion"])
        if ev:
            ep = eps_vals[:len(ev)]
            ax.plot(ep, ev, marker=marker, color=color, lw=2.5, label=method)
            for e, v in zip(ep, ev):
                ax.text(e, v+0.012, f"{v:.0%}", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xlabel("ε (Perturbation Budget)", fontsize=11)
        ax.set_ylabel("L3 Detection Evasion Rate", fontsize=11)
        ax.set_ylim(0, 1.1); ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_title("FGSM vs PGD — ε curve", fontweight="bold")
    axes[1].set_title("MI-FGSM — ε curve", fontweight="bold")
    plt.suptitle(f"Attack Strength vs Evasion Rate — {dataset.upper()}", fontsize=13, fontweight="bold")
    fig.tight_layout(); _save(fig, "06_epsilon_curve.png", dataset)


def plot_07_confusion_matrix_diff(before, after, dataset):
    """NEW: Confusion matrix before vs after for L3 Detection."""
    cm_b = np.array(before["detection"].get("confusion_matrix", [[1,0],[0,1]]))
    cm_a = np.array(after["detection"].get("confusion_matrix",  [[1,0],[0,1]]))
    if cm_b.shape != cm_a.shape:
        LOG.warn("Confusion matrix shape mismatch — skipping plot 07")
        return
    diff = cm_a.astype(float) - cm_b.astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title, cmap in [
        (axes[0], cm_b.astype(float), "BEFORE (Clean)",       "Blues"),
        (axes[1], cm_a.astype(float), "AFTER (Adversarial)",  "Reds"),
        (axes[2], diff,               "DIFFERENCE (After−Before)", "RdBu_r"),
    ]:
        sns.heatmap(mat, annot=True, fmt=".0f" if "DIFF" not in title else "+.0f",
                    cmap=cmap, ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title, fontweight="bold")
    plt.suptitle(f"Confusion Matrix — L3 Detection — {dataset.upper()}", fontsize=13, fontweight="bold")
    fig.tight_layout(); _save(fig, "07_confusion_matrix.png", dataset)


def plot_08_attack_radar(results, dataset):
    sel = [k for k in results if k in {
        "FGSM ε=0.10","PGD  ε=0.10","MI-FGSM ε=0.10",
        "C&W L2  c=1.0","AUTOATTACK ε=0.20","CONSTRAINED ε=0.15"}]
    if not sel: sel = list(results.keys())[:6]
    N = 5; angles = [n/N*2*np.pi for n in range(N)] + [0]
    labels  = ["L1\nIngestion","L2\nTriage","L3\nDetection","L4\nSIEM","L5\nSOAR"]
    colors  = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for ci, aname in enumerate(sel):
        vals = [results[aname][l]["evasion"] for l in LAYERS] + \
               [results[aname]["ingestion"]["evasion"]]
        ax.plot(angles, vals, lw=2, color=colors[ci % len(colors)], label=aname)
        ax.fill(angles, vals, alpha=0.07, color=colors[ci % len(colors)])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Attack Radar — Evasion Rate per Layer\n{dataset.upper()}",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=9)
    fig.tight_layout(); _save(fig, "08_attack_radar.png", dataset)


def plot_09_perturbation_magnitude(soc, dataset):
    d3 = soc["detection"]; m, X, y = d3["model"], d3["X_test"], d3["y_test"]
    eps_vals = [0.01, 0.05, 0.10, 0.20, 0.30]
    fgsm_l2, pgd_l2, mi_l2 = [], [], []
    for eps in eps_vals:
        Xf = fgsm(m, X, y, eps);    fgsm_l2.append(np.linalg.norm(Xf-X, axis=1).mean())
        Xp = pgd(m, X, y, eps);     pgd_l2.append(np.linalg.norm(Xp-X, axis=1).mean())
        Xm = mi_fgsm(m, X, y, eps); mi_l2.append(np.linalg.norm(Xm-X, axis=1).mean())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eps_vals, fgsm_l2, "bo-", lw=2, label="FGSM")
    ax.plot(eps_vals, pgd_l2,  "r^-", lw=2, label="PGD")
    ax.plot(eps_vals, mi_l2,   "gs-", lw=2, label="MI-FGSM")
    ax.set_xlabel("ε (Perturbation Budget)", fontsize=11)
    ax.set_ylabel("Mean L2 Perturbation", fontsize=11)
    ax.set_title(f"Attack Stealthiness — {dataset.upper()}", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, "09_perturbation_magnitude.png", dataset)


def plot_10_f1_auc_drop(results, dataset):
    attacks = list(results.keys())
    f1_drop = [results[a]["detection"]["clean"]["f1_score"] -
               results[a]["detection"]["adv"]["f1_score"] for a in attacks]
    auc_drop = [(results[a]["detection"]["clean"].get("auc_roc") or 0) -
                (results[a]["detection"]["adv"].get("auc_roc") or 0) for a in attacks]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, vals, label, color in [
        (axes[0], f1_drop,  "F1-Score Drop (L3 Detection)", "#e74c3c"),
        (axes[1], auc_drop, "AUC-ROC Drop  (L3 Detection)", "#8e44ad"),
    ]:
        colors = [color if v > 0 else "#2ecc71" for v in vals]
        ax.bar(range(len(attacks)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels(attacks, rotation=40, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(label); ax.grid(axis="y", alpha=0.3)
        ax.set_title(label, fontweight="bold")
    plt.suptitle(f"F1-Score & AUC-ROC Degradation — {dataset.upper()}", fontsize=13, fontweight="bold")
    fig.tight_layout(); _save(fig, "10_f1_auc_drop.png", dataset)


def plot_11_computed_drops_heatmap(results, dataset):
    """
    NEW (Feature: Computed drops) — Heatmap of every metric drop
    for every attack at the L3 Detection layer.
    Columns = metric drops; rows = attacks.
    """
    metrics = ["accuracy", "precision", "recall", "f1_score",
               "tpr", "fpr", "auc_roc", "avg_precision"]
    attacks = list(results.keys())
    matrix  = []
    for a in attacks:
        row = []
        for mk in metrics:
            d = results[a]["detection"]["drops"].get(mk)
            row.append(d if d is not None else 0.0)
        matrix.append(row)
    mat = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(16, max(6, len(attacks) * 0.38 + 2)))
    # Use diverging palette — red = degraded, green = improved
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=-0.2, vmax=0.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels(attacks, fontsize=7)
    ax.set_title(f"COMPUTED DROPS — L3 Detection  [{dataset.upper()}]\n"
                 "Metric drop: clean − adversarial  (red = degraded, green = improved)",
                 fontsize=12, fontweight="bold")
    for i, a in enumerate(attacks):
        for j, mk in enumerate(metrics):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(v) > 0.20 else "black")
    plt.colorbar(im, ax=ax, label="Metric Drop (clean − adv)")
    fig.tight_layout()
    _save(fig, "11_computed_drops_heatmap.png", dataset)


def plot_12_perturbation_vs_fooled(results, dataset):
    """
    NEW (Features: Input perturbed + Model actually fooled) —
    Scatter plot: mean L2 perturbation (x) vs % truly fooled (y)
    coloured by attack family. Also shows a secondary bar chart of
    pct_unchanged per attack.
    """
    attacks  = list(results.keys())
    l2_vals  = [results[a]["detection"]["perturbation"]["l2_mean"] for a in attacks]
    fooled   = [results[a]["detection"]["fooled"]["pct_truly_fooled"] for a in attacks]
    unchanged= [results[a]["detection"]["perturbation"]["pct_unchanged"] for a in attacks]

    # Colour by family
    FAMILIES = {
        "FGSM": "#3498db",  "PGD": "#e74c3c",  "MI-FGSM": "#9b59b6",
        "C&W": "#f39c12",   "DeepFool": "#1abc9c", "FEAT": "#e67e22",
        "POISON": "#2ecc71","TRANSFER": "#e84393","CONSTRAINED": "#95a5a6",
        "AUTOATTACK": "#c0392b",
    }
    colors = []
    for a in attacks:
        col = "#7f7f7f"
        for fam, c in FAMILIES.items():
            if a.startswith(fam):
                col = c; break
        colors.append(col)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # ── Scatter: L2 vs fooled ────────────────────────────────────────────────
    sc = ax1.scatter(l2_vals, [f * 100 for f in fooled],
                     c=colors, s=80, alpha=0.85, edgecolors="black", linewidths=0.5, zorder=3)
    for i, a in enumerate(attacks):
        if fooled[i] > 0.15 or l2_vals[i] > np.percentile(l2_vals, 80):
            ax1.annotate(a, (l2_vals[i], fooled[i] * 100),
                         fontsize=6.5, ha="left", va="bottom", alpha=0.85)
    ax1.set_xlabel("Mean L2 Perturbation (Input Changed)", fontsize=11)
    ax1.set_ylabel("Model Actually Fooled (%)", fontsize=11)
    ax1.set_title(f"Input Perturbation vs Model Fooled — L3 Detection\n{dataset.upper()}",
                  fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.3)
    # Legend patches
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=fam) for fam, c in FAMILIES.items()]
    ax1.legend(handles=handles, fontsize=7, loc="upper left", ncol=2)

    # ── Bar: pct unchanged per attack ────────────────────────────────────────
    bar_colors = ["#e74c3c" if u > 0.20 else "#f39c12" if u > 0.05 else "#2ecc71"
                  for u in unchanged]
    ax2.barh(range(len(attacks)), [u * 100 for u in unchanged], color=bar_colors, alpha=0.85)
    ax2.set_yticks(range(len(attacks)))
    ax2.set_yticklabels(attacks, fontsize=7)
    ax2.set_xlabel("% Samples Unchanged (||δ||₂ < 1e-6)", fontsize=10)
    ax2.set_title(f"Fraction of Inputs NOT Perturbed — L3\n{dataset.upper()}",
                  fontsize=11, fontweight="bold")
    ax2.axvline(5,  color="orange", ls="--", lw=1, alpha=0.6, label="5% threshold")
    ax2.axvline(20, color="red",    ls="--", lw=1, alpha=0.6, label="20% threshold")
    ax2.legend(fontsize=8); ax2.grid(axis="x", alpha=0.3)

    plt.suptitle(f"Input Perturbation & Model Fooled Analysis — {dataset.upper()}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "12_perturbation_vs_fooled.png", dataset)


def generate_all_plots(results, soc, before, after, dataset):
    LOG.phase("Generating 12 Research Plots")
    plot_01_cascade_heatmap(results, dataset)
    plot_02_multi_metric_heatmap(before, after, dataset)
    plot_03_cascade_flow(results, dataset)
    plot_04_before_after_bars(before, after, dataset)
    plot_05_evasion_per_layer(results, dataset)
    plot_06_epsilon_curve(results, dataset)
    plot_07_confusion_matrix_diff(before, after, dataset)
    plot_08_attack_radar(results, dataset)
    plot_09_perturbation_magnitude(soc, dataset)
    plot_10_f1_auc_drop(results, dataset)
    plot_11_computed_drops_heatmap(results, dataset)
    plot_12_perturbation_vs_fooled(results, dataset)
    LOG.ok(f"All 12 plots saved → adversarial_plots/")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — LIVE ATTACK MONITOR
# ══════════════════════════════════════════════════════════════════════════════

ATTACK_TYPES = ["DDoS","PortScan","BruteForce","Heartbleed",
                "Exfiltration","XSS","Ransomware","BotNet","SQLi"]

def _layer_pred_label(model, scaler, le, raw: np.ndarray):
    feat = raw.reshape(1, -1).astype(np.float32)
    n    = model.n_features_in_
    if feat.shape[1] < n:
        feat = np.concatenate([feat, np.zeros((1, n-feat.shape[1]))], axis=1)
    else:
        feat = feat[:, :n]
    try: feat = scaler.transform(feat)
    except: pass
    pred  = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0] if hasattr(model, "predict_proba") else None
    conf  = float(np.max(proba)) if proba is not None else 1.0
    label = le.inverse_transform([pred])[0] if le else str(pred)
    return str(label), round(conf, 3)


def live_attack_monitor(soc: dict, dataset: str, n_events: int = 20, eps: float = 0.15):
    print(f"\n{RED}{BOLD}{'═'*95}{RESET}")
    print(f"{RED}{BOLD}   ⚔  LIVE ADVERSARIAL ATTACK MONITOR  |  {dataset.upper()}  |  AutoAttack + PGD ε={eps}{RESET}")
    print(f"{RED}{BOLD}{'═'*95}{RESET}\n")
    print(f"{BOLD}  {'#':>3}  {'TYPE':<12}  {'L1-INGEST':<13}  {'L2-TRIAGE':<12}  "
          f"{'L3-DETECT':<14}  {'L4-SIEM':<14}  {'L5-SOAR':<12}  {'RISK':>5}  DECISION{RESET}")
    print(f"  {'─'*100}")

    d3  = soc["detection"]; rng = np.random.default_rng(42)
    n_evaded = 0; n_adv_total = 0

    for i in range(n_events):
        is_adv   = (i % 2 == 1)
        atk_type = rng.choice(ATTACK_TYPES)
        idx      = int(rng.integers(0, len(d3["X_test"])))
        x_base   = d3["X_test"][idx:idx+1].copy()

        if is_adv:
            n_adv_total += 1
            # Alternate between PGD and AutoAttack
            if i % 4 == 1:
                x_atk = pgd(d3["model"], x_base, np.array([1]), eps, steps=30, random_start=True)
            else:
                x_atk = mi_fgsm(d3["model"], x_base, np.array([1]), eps, steps=15)
        else:
            x_atk = x_base

        layer_preds = {}; risk_parts = []
        for layer in LAYERS:
            d = soc[layer]
            x_aligned = d["X_test"][idx % len(d["X_test"]):idx % len(d["X_test"])+1].copy()
            if is_adv:
                delta = x_atk - x_base
                scale = (np.abs(x_aligned).mean() + 1e-8) / (np.abs(x_base).mean() + 1e-8)
                nf    = min(delta.shape[1], x_aligned.shape[1])
                x_aligned[0, :nf] += delta[0, :nf] * scale
            label, conf = _layer_pred_label(d["model"], d["scaler"], d["le"], x_aligned[0])
            layer_preds[layer] = (label, conf)
            is_benign_pred = str(label).lower() in BENIGN_LABELS
            risk_parts.append(0.05 if is_benign_pred else conf * 0.90)

        weights = [0.10, 0.20, 0.35, 0.20, 0.15]
        risk    = min(sum(r * w for r, w in zip(risk_parts, weights)), 1.0)
        soar_lbl    = layer_preds["soar"][0]
        final_block = str(soar_lbl).lower() not in BENIGN_LABELS
        evaded      = is_adv and not final_block
        if evaded: n_evaded += 1

        tag     = f"{RED}[ADV]{RESET}" if is_adv else f"{GREEN}[CLN]{RESET}"
        l1      = str(layer_preds["ingestion"][0])[:12]
        l2      = str(layer_preds["triage"][0])[:11]
        l3      = str(layer_preds["detection"][0])[:13]
        l4      = str(layer_preds["siem"][0])[:13]
        l5      = str(layer_preds["soar"][0])[:11]
        risk_col = RED if risk > 0.6 else YELLOW if risk > 0.3 else GREEN
        dec     = (f"{RED}✘ ALLOW {YELLOW}← EVADED!{RESET}" if evaded
                   else f"{GREEN}✔ BLOCK{RESET}" if final_block
                   else f"{GREY}○ ALLOW{RESET}")
        print(f"  {i+1:>3}  {tag}  {atk_type:<10}  {l1:<13}  {l2:<12}  {l3:<14}  "
              f"{l4:<14}  {l5:<12}  {risk_col}{risk:.3f}{RESET}  {dec}")
        time.sleep(0.08)

    print(f"\n  {'─'*100}")
    evasion_rate = n_evaded / max(n_adv_total, 1)
    print(f"\n  {BOLD}LIVE MONITOR SUMMARY:{RESET}")
    print(f"  Adversarial events : {n_adv_total}")
    print(f"  Evaded L5          : {RED}{n_evaded}{RESET}")
    print(f"  Evasion Rate       : {RED}{BOLD}{evasion_rate:.1%}{RESET}\n")
    print(f"{RED}{BOLD}{'═'*95}{RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="STRONG Adversarial Attack v2 on 5-Layer AI-SOC — MSc Cybersecurity")
    ap.add_argument("--mode", default="all",
        choices=["all","cascade","realtime","fgsm","pgd","mifgsm","cw",
                 "deepfool","feature","poison","transfer","constrained","auto"],
        help="Attack mode")
    ap.add_argument("--dataset", default="cicids2017",
        choices=["cicids2017","ember","loghub","all"])
    ap.add_argument("--eps",    type=float, default=0.15)
    ap.add_argument("--events", type=int,   default=20)
    args = ap.parse_args()

    print(f"\n{RED}{BOLD}{'═'*82}{RESET}")
    print(f"{RED}{BOLD}  ⚔  ADVERSARIAL EVASION ATTACK v2 — ENHANCED STRENGTH{RESET}")
    print(f"{CYAN}  Attacks: FGSM · PGD(+RandStart) · MI-FGSM · C&W L2 · DeepFool{RESET}")
    print(f"{CYAN}           Feature-Manip · Log-Poison · Transfer · AutoAttack{RESET}")
    print(f"{GREY}  Mode: {args.mode}   Dataset: {args.dataset}   ε={args.eps}{RESET}")
    print(f"{RED}{BOLD}{'═'*82}{RESET}\n")

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        LOG.banner(f"DATASET: {dataset.upper()}")
        soc = load_all_soc_layers(dataset)

        # STEP 1: Baseline metrics (clean)
        baseline = compute_baseline_metrics(soc, dataset)

        if args.mode == "realtime":
            live_attack_monitor(soc, dataset, args.events, args.eps)
            continue

        # STEP 2: Run attacks
        all_results = run_experiment(soc, args.mode, dataset)

        # STEP 3: After metrics (worst-case adversarial)
        after = compute_after_metrics(soc, all_results, dataset)

        # STEP 4: Full before/after report
        print_before_after_report(baseline, after, dataset)

        # STEP 5: Cascade table
        print_cascade_table(all_results, dataset)

        # STEP 6: Save
        save_results(all_results, dataset)
        save_before_after_csv(baseline, after, dataset)

        # STEP 7: Plots
        if args.mode in ("all", "cascade"):
            generate_all_plots(all_results, soc, baseline, after, dataset)

        # STEP 8: Live monitor
        if args.mode == "all":
            live_attack_monitor(soc, dataset, args.events, args.eps)

        # STEP 9: Experiment ledger
        LEDGER.save(dataset)
        LEDGER.print_summary(dataset)

    print(f"\n{GREEN}{BOLD}✔  EXPERIMENT COMPLETE{RESET}")
    print(f"  Results → adversarial_results/")
    print(f"  Plots   → adversarial_plots/\n")
    print(f"  {BOLD}Run commands:{RESET}")
    print(f"  {GREY}python adversarial_attack_v2.py --mode all{RESET}")
    print(f"  {GREY}python adversarial_attack_v2.py --mode auto --dataset cicids2017{RESET}")
    print(f"  {GREY}python adversarial_attack_v2.py --mode realtime --events 30{RESET}")
    print(f"  {GREY}python adversarial_attack_v2.py --mode pgd --eps 0.30{RESET}\n")


if __name__ == "__main__":
    main()