"""
Layer 3 — DETECTION  (Agentic upgrade)
────────────────────────────────────────────────────────────────────────────────
What's new vs baseline:
  • ReasoningAgent  — NL interpretation of MLP outputs, anomaly findings
  • AdversarialDetector  — detects evasion attempts against the neural net
  • Adaptive strategy  — when evasion is detected, falls back to Isolation Forest
    score as primary signal rather than MLP confidence (adversarial robustness)
  • FeedbackBus  — analyst corrections tune future risk weighting
  • AgentMemory  — zero-day pattern tracking across events
────────────────────────────────────────────────────────────────────────────────
"""
import time, argparse
import numpy as np
import warnings
from sklearn.neural_network  import MLPClassifier
from sklearn.ensemble        import IsolationForest
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from p1_soc_utils import (
    Logger, SOCEvent, DataLoader, MetricsEngine, Store, PlotEngine,
    DriftDetector, AgentMemory, FeedbackBus, ReasoningAgent, AdversarialDetector,
    DATASETS, MODEL_ROOT, PLOT_ROOT, MITRE_MAP,
    BENIGN_LABELS, RESET, BOLD,
)

LAYER = "detection"

MLP_BASE = dict(
    hidden_layer_sizes=(512, 256, 128, 64),
    activation="relu", solver="adam", alpha=0.0001,
    batch_size=512, learning_rate="adaptive",
    learning_rate_init=0.001, max_iter=500,
    early_stopping=True, validation_fraction=0.10,
    n_iter_no_change=20, tol=1e-5, random_state=42,
)
PARAM_GRID = {
    "hidden_layer_sizes": [(256, 128, 64), (512, 256, 128, 64)],
    "alpha"             : [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.005],
}

THREAT_LABELS = {
    "ddos","portscan","port scan","bruteforce","brute force","exfiltration",
    "infiltration","botnet","heartbleed","slowloris","goldeneye","hulk",
    "rudy","malware","ransomware","trojan","backdoor","injection","xss",
    "sql injection","web attack","lateral_movement","c2","beacon",
    "dos","dosgoldeneye","doshulk","dosrudy","dosslowloris",
    "ftp-patator","ssh-patator","1","attack","anomaly","malicious",
    "syslog_attack","auth_failure","suspicious",
    "2","3","4","5","6","7","8","9","10",
    "true","positive","1.0",
}


class DetectionLayer:
    def __init__(self, memory: AgentMemory = None,
                 feedback: FeedbackBus    = None):
        self.models    = {}
        self.anomaly   = {}
        self.scalers   = {}
        self.encoders  = {}
        self.les       = {}
        self.feat_names= {}
        self.drifters  = {}
        self.log       = Logger(LAYER)

        # ── Agentic components ───────────────────────────────────────────
        self.memory    = memory   or AgentMemory()
        self.feedback  = feedback or FeedbackBus()
        self.agent     = ReasoningAgent(LAYER)
        self.adv_det   = AdversarialDetector()

        # Track how often adversarial fallback is triggered per dataset
        self._adv_fallback_count: dict = {}

    # ═══════════════════════════════════════════════════════════════════════
    #  TRAIN
    # ═══════════════════════════════════════════════════════════════════════
    def train(self, dataset: str, tune: bool = True) -> dict:
        log = Logger(LAYER, dataset)
        log.banner(
            f"LAYER 3 — DETECTION  │  MLP 512→256→128→64 + Adversarial Awareness  │  "
            f"{dataset.upper()}")
        log.info("Architecture: Input → 512 → 256 → 128 → 64 → Output (ReLU + Adam)")

        loader = DataLoader(log)
        res    = loader.load(dataset, LAYER)
        if res[0] is None:
            log.err(f"No data for {dataset}/{LAYER}"); return {}

        X_tr, y_tr, X_te, y_te, sc, enc, le, feat_names = res
        n_cls = len(np.unique(y_tr))
        log.info(f"Classes ({n_cls}): {le.classes_.tolist()}")

        self.adv_det.fit_baseline(X_tr)
        log.ok("AdversarialDetector baseline fitted on neural-network training data")

        # ── Hyperparameter Tuning ────────────────────────────────────────
        if tune:
            log.section("Hyperparameter Tuning (GridSearchCV) …")
            base = MLPClassifier(max_iter=100, random_state=42,
                                 early_stopping=True, validation_fraction=0.10)
            gs   = GridSearchCV(base, PARAM_GRID, cv=3,
                                scoring="f1_weighted", n_jobs=-1, verbose=0)
            t0   = time.time()
            gs.fit(X_tr, y_tr)
            best_p = gs.best_params_
            log.ok(f"Best: {best_p}  │  Score: {gs.best_score_:.4f}  "
                   f"│  {time.time()-t0:.1f}s")
            model_params = {**MLP_BASE, **best_p}
        else:
            model_params = MLP_BASE

        # ── Train ────────────────────────────────────────────────────────
        log.section("Training MLP Neural Network …")
        model = MLPClassifier(**model_params)
        t0    = time.time()
        model.fit(X_tr, y_tr)
        log.ok(f"Done {time.time()-t0:.1f}s  │  Epochs={model.n_iter_}  │  "
               f"Loss={model.loss_:.6f}")

        # ── Anomaly Detector ─────────────────────────────────────────────
        log.section("Training secondary Anomaly Detector …")
        iso = IsolationForest(n_estimators=150, contamination=0.05,
                              random_state=42, n_jobs=-1)
        iso.fit(X_tr)
        log.ok("Isolation Forest fitted for zero-day / adversarial fallback")

        # ── Evaluate ─────────────────────────────────────────────────────
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        metrics = MetricsEngine.compute(y_te, y_pred, y_proba, le)
        MetricsEngine.print_table(metrics, LAYER, dataset)

        lp       = le.inverse_transform(y_pred)
        n_threat = sum(1 for l in lp if str(l).lower() in THREAT_LABELS
                       or str(l).lower() not in BENIGN_LABELS)
        log.info(f"Threats detected: {n_threat:,}/{len(y_pred):,} "
                 f"({100*n_threat/len(y_pred):.1f}%)")

        # ── K-Fold CV ────────────────────────────────────────────────────
        log.section("5-Fold Cross Validation …")
        cv_res = MetricsEngine.kfold_cv(
            MLPClassifier, model_params, X_tr, y_tr, k=5, log=log)

        # ── Loss Convergence Plot ────────────────────────────────────────
        self._loss_curve(model, LAYER, dataset)

        # ── Standard Plots ───────────────────────────────────────────────
        log.section("Generating evaluation plots …")
        PlotEngine.roc(y_te, y_proba, le, LAYER, dataset)
        PlotEngine.pr_curve(y_te, y_proba, le, LAYER, dataset)
        PlotEngine.confusion(y_te, y_pred, le, LAYER, dataset)
        PlotEngine.cv_results(cv_res, LAYER, dataset)
        PlotEngine.learning_curve(
            MLPClassifier,
            {**model_params, "max_iter":50, "n_iter_no_change":5},
            X_tr, y_tr, LAYER, dataset)
        log.ok(f"Plots → ai_soc_plots/{LAYER}/{dataset}_*.png")

        # ── Drift baseline ───────────────────────────────────────────────
        self.drifters[dataset] = DriftDetector()
        self.drifters[dataset].fit(y_proba)

        # ── MITRE ────────────────────────────────────────────────────────
        self._print_mitre(le, y_pred, log)

        # ── Save ─────────────────────────────────────────────────────────
        Store.save(LAYER, dataset, model, sc, enc, metrics, feat_names, cv_res)
        joblib.dump(iso, MODEL_ROOT/LAYER/f"{dataset}_isoforest.joblib")
        log.ok(f"Saved → ai_soc_models/{LAYER}/{dataset}_model.joblib")

        self.models[dataset]     = model
        self.anomaly[dataset]    = iso
        self.scalers[dataset]    = sc
        self.encoders[dataset]   = enc
        self.les[dataset]        = le
        self.feat_names[dataset] = feat_names
        return {"metrics": metrics, "cv_results": cv_res}

    def train_all(self, tune: bool = True) -> dict:
        all_m = {}
        for ds in DATASETS:
            m = self.train(ds, tune=tune)
            if m: all_m[ds] = m
        self._summary(all_m)
        return all_m

    # ═══════════════════════════════════════════════════════════════════════
    #  LOAD
    # ═══════════════════════════════════════════════════════════════════════
    def load(self, dataset: str):
        model, sc, enc, le, fn  = Store.load(LAYER, dataset)
        self.models[dataset]    = model
        self.scalers[dataset]   = sc
        self.encoders[dataset]  = enc
        self.les[dataset]       = le
        self.feat_names[dataset]= fn
        iso_p = MODEL_ROOT/LAYER/f"{dataset}_isoforest.joblib"
        if iso_p.exists():
            self.anomaly[dataset] = joblib.load(iso_p)
        self.log.ok(f"Detection loaded: {dataset}")

    # ═══════════════════════════════════════════════════════════════════════
    #  REAL-TIME INFERENCE  (agentic)
    # ═══════════════════════════════════════════════════════════════════════
    def infer(self, event: SOCEvent) -> SOCEvent:
        ds = event.dataset
        if ds not in self.models:
            try:   self.load(ds)
            except FileNotFoundError:
                event.record(LAYER, "unknown", 0.0, 0.5); return event

        label, conf, proba = Store.infer(
            self.models[ds], self.scalers[ds], self.les[ds], event.raw)

        # ── Adversarial check ────────────────────────────────────────────
        adv = self.adv_det.check(event.raw, proba, label, event.event_id)
        event.record_adversarial(LAYER, adv)

        # ── Adversarial strategy adaptation ─────────────────────────────
        # When evasion is detected, distrust MLP classification and rely on
        # the Isolation Forest anomaly score as the primary risk signal.
        used_fallback = False
        if adv["is_adversarial"] and ds in self.anomaly:
            self._adv_fallback_count[ds] = self._adv_fallback_count.get(ds,0) + 1
            self.log.warn(
                f"[DETECTION] Adversarial input detected — switching to "
                f"IsolationForest fallback (fallback #{self._adv_fallback_count[ds]})")
            self.feedback.emit(LAYER, event.event_id, "adversarial_detected",
                               adv["score"], adv["hint"])
            # Override label/conf with anomaly-based values
            feat = Store.align(event.raw.reshape(1,-1).astype(np.float32),
                               self.models[ds].n_features_in_)
            try:   feat = self.scalers[ds].transform(feat)
            except Exception: pass
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    iso_pred = int(self.anomaly[ds].predict(feat)[0])
                    iso_score= float(self.anomaly[ds].decision_function(feat)[0])
            except Exception:
                iso_pred, iso_score = -1, -0.5

            if iso_pred == -1:
                label       = f"{label}[ADV-ANOMALY]"
                conf        = min(conf + 0.20, 1.0)   # inflate confidence for the threat
            used_fallback = True

        # ── Standard anomaly check ───────────────────────────────────────
        is_anomaly = False
        if ds in self.anomaly and not used_fallback:
            feat = Store.align(event.raw.reshape(1,-1).astype(np.float32),
                               self.models[ds].n_features_in_)
            try:   feat = self.scalers[ds].transform(feat)
            except Exception: pass
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    is_anomaly = int(self.anomaly[ds].predict(feat)[0]) == -1
            except Exception:
                is_anomaly = False

        # ── L2 severity context ──────────────────────────────────────────
        l2_sev    = str(event.pipeline.get("triage",{}).get("prediction","")).lower()
        sev_boost = 0.10 if l2_sev in {"high","critical","emergency"} else 0.0

        # ── Feedback-driven weight adjustment ────────────────────────────
        fb_adj = self.feedback.adjustment(LAYER)

        is_threat    = (str(label).lower().rstrip("[anomaly]") in THREAT_LABELS or
                        str(label).lower() not in BENIGN_LABELS or is_anomaly)
        risk_contrib = float(np.clip(
            conf * (0.95 if is_threat else 0.04)
            + sev_boost + (0.15 if is_anomaly else 0)
            + fb_adj,
            0.0, 1.0))
        event.feedback_applied[LAYER] = fb_adj

        if is_anomaly and not used_fallback:
            label = f"{label}[ANOMALY]"

        event.record(LAYER, label, conf, risk_contrib)

        # ── Agent memory push ────────────────────────────────────────────
        self.memory.push(event.event_id, LAYER, {
            "prediction": label, "confidence": conf,
            "risk_contrib": risk_contrib,
            "is_anomaly": is_anomaly,
            "adversarial_fallback": used_fallback,
        })
        if is_anomaly or used_fallback:
            self.memory.push_evasion(event.event_id,
                                     adv.get("hint","zero-day anomaly"),
                                     adv.get("score",0.5))

        # ── Reasoning trace ──────────────────────────────────────────────
        reasoning = self.agent.explain(
            event_id    = event.event_id,
            prediction  = label,
            confidence  = conf,
            risk_score  = event.risk_score,
            pipeline_ctx= event.pipeline,
            mitre        = event.mitre,
            is_anomaly   = is_anomaly or used_fallback,
            evasion_hint = adv.get("hint",""),
        )
        if used_fallback:
            reasoning += (
                f"\n  🔄 STRATEGY ADAPTED: MLP classification distrusted due to "
                f"adversarial signal. IsolationForest used as primary detector. "
                f"Fallback count for {ds}: {self._adv_fallback_count[ds]}."
            )
        event.record_reasoning(LAYER, reasoning)
        self.log.reasoning(reasoning)
        self.log.event(event.event_id, LAYER, label, conf, event.risk_score)
        return event

    # ═══════════════════════════════════════════════════════════════════════
    #  ANALYST FEEDBACK
    # ═══════════════════════════════════════════════════════════════════════
    def analyst_correct(self, event_id: str, original: str,
                        corrected: str, analyst: str = "analyst"):
        self.memory.push_feedback(event_id, original, corrected, analyst)
        signal = ("false_positive" if corrected.lower() in BENIGN_LABELS
                  else "confirmed_threat")
        self.feedback.emit(LAYER, event_id, signal, 1.0, f"{original} → {corrected}")
        self.log.info(f"Analyst correction: {original} → {corrected} "
                      f"(adj={self.feedback.adjustment(LAYER):+.3f})")

    # ═══════════════════════════════════════════════════════════════════════
    #  DEMO
    # ═══════════════════════════════════════════════════════════════════════
    def run_demo(self, dataset: str, n: int = 10):
        self.log.banner(f"L3 DETECTION AGENT DEMO  │  {dataset.upper()}")
        try:   self.load(dataset)
        except FileNotFoundError as e:
            self.log.err(str(e)); return
        loader = DataLoader(self.log)
        res    = loader.load(dataset, LAYER)
        if res[0] is None: return
        X_te   = res[2]
        if self.adv_det.baseline_mean is None:
            self.adv_det.fit_baseline(X_te)
        idxs = np.random.default_rng(42).choice(len(X_te), min(n,len(X_te)), replace=False)
        for i, idx in enumerate(idxs):
            ev = SOCEvent(X_te[idx], dataset, f"DET-{i+1:04d}")
            self.infer(ev); time.sleep(0.05)
        print(self.agent.summarise_session(self.memory))

    def _loss_curve(self, model, layer, dataset):
        pdir = PLOT_ROOT / layer; pdir.mkdir(parents=True, exist_ok=True)
        if not hasattr(model, "loss_curve_"):
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(model.loss_curve_, label="Training Loss",   color="#F44336")
        if hasattr(model, "validation_scores_") and model.validation_scores_:
            ax.plot(model.validation_scores_, label="Val Accuracy", color="#4CAF50")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss / Accuracy")
        ax.set_title(f"MLP Training Convergence — {layer.upper()} / {dataset.upper()}", fontsize=12)
        ax.legend(); ax.grid(alpha=0.3)
        out = pdir / f"{dataset}_loss_curve.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    def _print_mitre(self, le, y_pred, log):
        if le is None: return
        mapped = set()
        for pi in np.unique(y_pred):
            lbl = str(le.inverse_transform([pi])[0]).lower()
            if lbl in MITRE_MAP:
                t, tech, tid = MITRE_MAP[lbl]
                mapped.add(f"  {tid:<12} {t:<22} {tech}")
        if mapped:
            log.section("MITRE ATT&CK Mappings:")
            for m in sorted(mapped): print(m)

    def _summary(self, all_m: dict):
        self.log.banner("DETECTION — SUMMARY")
        print(f"  {BOLD}{'Dataset':<14} {'Acc':>7} {'F1':>7} {'AUC':>7} {'TPR':>7} {'FPR':>7}{RESET}")
        print(f"  {'─'*52}")
        for ds, payload in all_m.items():
            m = payload.get("metrics", payload)
            print(f"  {ds:<14} {m['accuracy']:>7.4f} {m['f1_score']:>7.4f} "
                  f"{(m.get('auc_roc') or 0):>7.4f} {m['tpr']:>7.4f} {m['fpr']:>7.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",    choices=["train","realtime"], default="train")
    ap.add_argument("--dataset", choices=DATASETS+["all"],     default="all")
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    L3   = DetectionLayer()
    if args.mode == "train":
        if args.dataset == "all": L3.train_all(tune=not args.no_tune)
        else: L3.train(args.dataset, tune=not args.no_tune)
    else:
        ds = "cicids2017" if args.dataset == "all" else args.dataset
        L3.run_demo(ds)