"""
Layer 1 — INGESTION  (Agentic upgrade)
────────────────────────────────────────────────────────────────────────────────
What's new vs baseline:
  • ReasoningAgent  — NL explanation of every inference decision
  • AdversarialDetector  — flags crafted/evasion inputs at ingest time
  • FeedbackBus integration  — adjusts risk weight based on analyst corrections
  • AgentMemory  — persists cross-event context for session-level awareness
  • Dynamic risk thresholds  — adjusted per feedback bus signal
────────────────────────────────────────────────────────────────────────────────
"""
import time, json, argparse
import numpy as np
from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib
from p1_soc_utils import (
    Logger, SOCEvent, DataLoader, MetricsEngine, Store, PlotEngine,
    DriftDetector, AgentMemory, FeedbackBus, ReasoningAgent, AdversarialDetector,
    DATASETS, LAYER_META, MODEL_ROOT, RESULT_ROOT,
    MITRE_MAP, BENIGN_LABELS, RESET, BOLD, GREEN, GREY,
)

LAYER = "ingestion"

RF_BASE = dict(
    n_estimators=300, max_depth=25, min_samples_split=4,
    min_samples_leaf=2, max_features="sqrt",
    class_weight="balanced", oob_score=True,
    n_jobs=-1, random_state=42,
)
PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth"   : [15, 20, 25],
    "max_features": ["sqrt", "log2"],
}


class IngestionLayer:
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

    # ═══════════════════════════════════════════════════════════════════════
    #  TRAIN
    # ═══════════════════════════════════════════════════════════════════════
    def train(self, dataset: str, tune: bool = True) -> dict:
        log = Logger(LAYER, dataset)
        log.banner(
            f"LAYER 1 — INGESTION  │  Random Forest + Reasoning Agent  │  {dataset.upper()}")

        loader = DataLoader(log)
        res    = loader.load(dataset, LAYER)
        if res[0] is None:
            log.err(f"No data for {dataset}/{LAYER}"); return {}

        X_tr, y_tr, X_te, y_te, sc, enc, le, feat_names = res
        log.info(f"Classes: {le.classes_.tolist()}")

        # Fit adversarial detector baseline on training data
        self.adv_det.fit_baseline(X_tr)
        log.ok("AdversarialDetector baseline fitted on training distribution")

        # ── Hyperparameter Tuning ────────────────────────────────────────
        if tune:
            log.section("Hyperparameter Tuning (GridSearchCV 3-fold) …")
            base   = RandomForestClassifier(class_weight="balanced",
                                            oob_score=False, n_jobs=-1, random_state=42)
            cv_tune= StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            gs     = GridSearchCV(base, PARAM_GRID, cv=cv_tune,
                                  scoring="f1_weighted", n_jobs=-1, verbose=0)
            t0     = time.time()
            gs.fit(X_tr, y_tr)
            best_p = gs.best_params_
            log.ok(f"Best params: {best_p}  │  CV score: {gs.best_score_:.4f}  "
                   f"│  Time: {time.time()-t0:.1f}s")
            model_params = {**RF_BASE, **best_p}
        else:
            model_params = RF_BASE

        # ── Train final model ────────────────────────────────────────────
        log.section("Training Final Random Forest …")
        model = RandomForestClassifier(**model_params)
        t0    = time.time()
        model.fit(X_tr, y_tr)
        log.ok(f"Done {time.time()-t0:.1f}s  │  OOB={model.oob_score_:.4f}  │  "
               f"Trees={model.n_estimators}")

        # ── Anomaly Detection ────────────────────────────────────────────
        log.section("Training Anomaly Detector (Isolation Forest) …")
        iso = IsolationForest(n_estimators=200, contamination=0.05,
                              random_state=42, n_jobs=-1)
        iso.fit(X_tr)
        log.ok("Isolation Forest fitted — detects zero-day / unknown attacks")

        # ── Evaluate ─────────────────────────────────────────────────────
        log.section("Evaluating on test set …")
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        metrics = MetricsEngine.compute(y_te, y_pred, y_proba, le)
        MetricsEngine.print_table(metrics, LAYER, dataset)

        n_anom = int((iso.predict(X_te) == -1).sum())
        log.info(f"Anomalies detected in test: {n_anom}/{len(X_te)} "
                 f"({100*n_anom/len(X_te):.1f}%)")

        # ── K-Fold CV ────────────────────────────────────────────────────
        log.section("5-Fold Cross Validation …")
        cv_res = MetricsEngine.kfold_cv(
            RandomForestClassifier, model_params, X_tr, y_tr, k=5, log=log)

        # ── Plots ────────────────────────────────────────────────────────
        log.section("Generating plots …")
        PlotEngine.roc(y_te, y_proba, le, LAYER, dataset)
        PlotEngine.pr_curve(y_te, y_proba, le, LAYER, dataset)
        PlotEngine.confusion(y_te, y_pred, le, LAYER, dataset)
        PlotEngine.feature_importance(model, feat_names, LAYER, dataset)
        PlotEngine.cv_results(cv_res, LAYER, dataset)
        PlotEngine.learning_curve(
            RandomForestClassifier, {**model_params, "oob_score":False, "n_jobs":-1},
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
        self.log.ok(f"Ingestion loaded: {dataset}")

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
        if adv["is_adversarial"]:
            self.memory.push_evasion(event.event_id, adv["hint"], adv["score"])
            self.feedback.emit(LAYER, event.event_id, "adversarial_detected",
                               adv["score"], adv["hint"])

        # ── Anomaly check ────────────────────────────────────────────────
        is_anomaly = False
        if ds in self.anomaly:
            feat = Store.align(event.raw.reshape(1,-1).astype(np.float32),
                               self.models[ds].n_features_in_)
            try:   feat = self.scalers[ds].transform(feat)
            except Exception: pass
            is_anomaly = int(self.anomaly[ds].predict(feat)[0]) == -1

        # ── Feedback-driven risk adjustment ─────────────────────────────
        fb_adj    = self.feedback.adjustment(LAYER)
        is_benign = str(label).lower() in BENIGN_LABELS and not is_anomaly
        base_risk = conf * (0.05 if is_benign else 0.90)
        if is_anomaly:
            base_risk = max(base_risk, 0.70)
            label     = f"{label}[ANOMALY]"
        # Adversarial events get a risk boost regardless of label
        if adv["is_adversarial"]:
            base_risk = max(base_risk, 0.65)
        risk_contrib = float(np.clip(base_risk + fb_adj, 0.0, 1.0))
        event.feedback_applied[LAYER] = fb_adj

        event.record(LAYER, label, conf, risk_contrib)

        # ── Agent memory push ────────────────────────────────────────────
        self.memory.push(event.event_id, LAYER, {
            "prediction": label, "confidence": conf,
            "risk_contrib": risk_contrib, "is_anomaly": is_anomaly,
            "adversarial": adv["is_adversarial"],
        })

        # ── Reasoning trace ──────────────────────────────────────────────
        reasoning = self.agent.explain(
            event_id   = event.event_id,
            prediction = label,
            confidence = conf,
            risk_score = event.risk_score,
            pipeline_ctx = event.pipeline,
            mitre        = event.mitre,
            is_anomaly   = is_anomaly,
            evasion_hint = adv.get("hint",""),
        )
        event.record_reasoning(LAYER, reasoning)
        self.log.reasoning(reasoning)
        self.log.event(event.event_id, LAYER, label, conf, event.risk_score)
        return event

    # ═══════════════════════════════════════════════════════════════════════
    #  ANALYST FEEDBACK  — online correction
    # ═══════════════════════════════════════════════════════════════════════
    def analyst_correct(self, event_id: str, original_label: str,
                        corrected_label: str, analyst: str = "analyst"):
        """
        Called when a human analyst corrects a label.
        Updates feedback bus so future risk weights adapt.
        """
        self.memory.push_feedback(event_id, original_label, corrected_label, analyst)
        signal = ("false_positive" if corrected_label.lower() in BENIGN_LABELS
                  else "confirmed_threat")
        self.feedback.emit(LAYER, event_id, signal, 1.0,
                           f"{original_label} → {corrected_label}")
        self.log.info(f"Analyst correction recorded: {original_label} → "
                      f"{corrected_label} (bus adjustment={self.feedback.adjustment(LAYER):+.3f})")

    # ═══════════════════════════════════════════════════════════════════════
    #  DEMO
    # ═══════════════════════════════════════════════════════════════════════
    def run_demo(self, dataset: str, n: int = 10):
        self.log.banner(f"L1 INGESTION AGENT DEMO  │  {dataset.upper()}")
        try:   self.load(dataset)
        except FileNotFoundError as e:
            self.log.err(str(e)); return
        loader = DataLoader(self.log)
        res    = loader.load(dataset, LAYER)
        if res[0] is None: return
        X_te   = res[2]
        # Fit adversarial baseline from test data if not already set
        if self.adv_det.baseline_mean is None:
            self.adv_det.fit_baseline(X_te)
        idxs = np.random.default_rng(42).choice(len(X_te), min(n,len(X_te)), replace=False)
        for i, idx in enumerate(idxs):
            ev = SOCEvent(X_te[idx], dataset, f"ING-{i+1:04d}")
            self.infer(ev); time.sleep(0.05)
        print(self.agent.summarise_session(self.memory))

    def _print_mitre(self, le, y_pred, log):
        if le is None: return
        mapped = set()
        for pred_idx in np.unique(y_pred):
            label = str(le.inverse_transform([pred_idx])[0]).lower()
            if label in MITRE_MAP:
                t, tech, tid = MITRE_MAP[label]
                mapped.add(f"  {tid:<12} {t:<22} {tech}")
        if mapped:
            log.section("MITRE ATT&CK Mappings detected:")
            for m in sorted(mapped): print(f"  {GREY}{m}{RESET}")

    def _summary(self, all_m: dict):
        self.log.banner("INGESTION — SUMMARY")
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
    L1   = IngestionLayer()
    if args.mode == "train":
        if args.dataset == "all": L1.train_all(tune=not args.no_tune)
        else: L1.train(args.dataset, tune=not args.no_tune)
    else:
        ds = "cicids2017" if args.dataset == "all" else args.dataset
        L1.run_demo(ds)