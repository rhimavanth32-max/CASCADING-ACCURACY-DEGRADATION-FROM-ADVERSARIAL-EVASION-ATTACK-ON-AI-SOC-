"""
Layer 2 — TRIAGE  (Agentic upgrade)
────────────────────────────────────────────────────────────────────────────────
What's new vs baseline:
  • ReasoningAgent  — explains severity prioritisation decisions in NL
  • Temporal clustering  — groups bursts of similar events to amplify risk
  • FeedbackBus integration  — analyst corrections shift severity weights
  • AgentMemory  — tracks session-level threat patterns for context boost
  • Adaptive SEV_WEIGHT  — adjusted at runtime by feedback bus signal
────────────────────────────────────────────────────────────────────────────────
"""
import time, argparse
from collections import deque
import numpy as np
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from p1_soc_utils import (
    Logger, SOCEvent, DataLoader, MetricsEngine, Store, PlotEngine,
    DriftDetector, AgentMemory, FeedbackBus, ReasoningAgent, AdversarialDetector,
    DATASETS, MODEL_ROOT, MITRE_MAP, BENIGN_LABELS, RESET, BOLD,
)

LAYER = "triage"

GB_BASE = dict(
    n_estimators=200, learning_rate=0.08, max_depth=7,
    subsample=0.85, min_samples_split=4, min_samples_leaf=2,
    max_features="sqrt", n_iter_no_change=20,
    validation_fraction=0.10, tol=1e-4, random_state=42,
)
PARAM_GRID = {
    "n_estimators"  : [100, 200],
    "learning_rate" : [0.05, 0.08, 0.10],
    "max_depth"     : [5, 7],
}

# Base severity weights — adjusted dynamically by feedback bus
BASE_SEV_WEIGHT = {
    "low":0.05,"info":0.05,"0":0.05,"normal":0.05,
    "medium":0.40,"moderate":0.40,
    "high":0.75,"elevated":0.75,
    "critical":0.95,"emergency":0.98,
}

# Temporal cluster window: number of recent events considered
CLUSTER_WINDOW = 15


class TriageLayer:
    def __init__(self, memory: AgentMemory = None,
                 feedback: FeedbackBus    = None):
        self.models    = {}
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

        # Temporal cluster tracker per dataset
        self._cluster_window: dict = {}   # dataset → deque of recent labels

    # ═══════════════════════════════════════════════════════════════════════
    #  TRAIN
    # ═══════════════════════════════════════════════════════════════════════
    def train(self, dataset: str, tune: bool = True) -> dict:
        log = Logger(LAYER, dataset)
        log.banner(
            f"LAYER 2 — TRIAGE  │  Gradient Boosting + Temporal Clustering  │  {dataset.upper()}")

        loader = DataLoader(log)
        res    = loader.load(dataset, LAYER)
        if res[0] is None:
            log.err(f"No data for {dataset}/{LAYER}"); return {}

        X_tr, y_tr, X_te, y_te, sc, enc, le, feat_names = res
        log.info(f"Priority classes: {le.classes_.tolist()}")

        self.adv_det.fit_baseline(X_tr)
        log.ok("AdversarialDetector baseline fitted")

        # ── Hyperparameter Tuning ────────────────────────────────────────
        if tune:
            log.section("Hyperparameter Tuning (GridSearchCV) …")
            base = GradientBoostingClassifier(random_state=42)
            gs   = GridSearchCV(base, PARAM_GRID, cv=3,
                                scoring="f1_weighted", n_jobs=-1, verbose=0)
            t0   = time.time()
            gs.fit(X_tr, y_tr)
            best_p = gs.best_params_
            log.ok(f"Best: {best_p}  │  Score: {gs.best_score_:.4f}  "
                   f"│  {time.time()-t0:.1f}s")
            model_params = {**GB_BASE, **best_p}
        else:
            model_params = GB_BASE

        # ── Train ────────────────────────────────────────────────────────
        log.section("Training Gradient Boosting …")
        model = GradientBoostingClassifier(**model_params)
        t0    = time.time()
        model.fit(X_tr, y_tr)
        log.ok(f"Done {time.time()-t0:.1f}s  │  Iters={model.n_estimators_}")

        # ── Evaluate ─────────────────────────────────────────────────────
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        metrics = MetricsEngine.compute(y_te, y_pred, y_proba, le)
        MetricsEngine.print_table(metrics, LAYER, dataset)

        if le is not None:
            lp = le.inverse_transform(y_pred)
            vals, cts = np.unique(lp, return_counts=True)
            log.info("Test severity dist: " + "  ".join(f"{v}={c}" for v,c in zip(vals,cts)))

        # ── K-Fold CV ────────────────────────────────────────────────────
        log.section("5-Fold Cross Validation …")
        cv_res = MetricsEngine.kfold_cv(
            GradientBoostingClassifier, model_params, X_tr, y_tr, k=5, log=log)

        # ── Plots ────────────────────────────────────────────────────────
        log.section("Generating plots …")
        PlotEngine.roc(y_te, y_proba, le, LAYER, dataset)
        PlotEngine.pr_curve(y_te, y_proba, le, LAYER, dataset)
        PlotEngine.confusion(y_te, y_pred, le, LAYER, dataset)
        PlotEngine.feature_importance(model, feat_names, LAYER, dataset)
        PlotEngine.cv_results(cv_res, LAYER, dataset)
        PlotEngine.learning_curve(
            GradientBoostingClassifier, model_params, X_tr, y_tr, LAYER, dataset)
        log.ok(f"Plots → ai_soc_plots/{LAYER}/{dataset}_*.png")

        self.drifters[dataset] = DriftDetector()
        self.drifters[dataset].fit(y_proba)
        self._print_mitre(le, y_pred, log)

        Store.save(LAYER, dataset, model, sc, enc, metrics, feat_names, cv_res)
        log.ok(f"Saved → ai_soc_models/{LAYER}/{dataset}_model.joblib")

        self.models[dataset]     = model
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
        self.log.ok(f"Triage loaded: {dataset}")

    # ═══════════════════════════════════════════════════════════════════════
    #  REAL-TIME INFERENCE  (agentic)
    # ═══════════════════════════════════════════════════════════════════════
    def infer(self, event: SOCEvent) -> SOCEvent:
        ds = event.dataset
        if ds not in self.models:
            try:   self.load(ds)
            except FileNotFoundError:
                event.record(LAYER, "medium", 0.0, 0.40); return event

        label, conf, proba = Store.infer(
            self.models[ds], self.scalers[ds], self.les[ds], event.raw)

        # ── Adversarial check ────────────────────────────────────────────
        adv = self.adv_det.check(event.raw, proba, label, event.event_id)
        event.record_adversarial(LAYER, adv)
        if adv["is_adversarial"]:
            self.feedback.emit(LAYER, event.event_id, "adversarial_detected",
                               adv["score"], adv["hint"])

        # ── Temporal clustering boost ────────────────────────────────────
        if ds not in self._cluster_window:
            self._cluster_window[ds] = deque(maxlen=CLUSTER_WINDOW)
        win = self._cluster_window[ds]
        win.append(str(label).lower())
        threat_frac = sum(1 for l in win if l not in BENIGN_LABELS) / max(len(win),1)
        cluster_boost = 0.12 * threat_frac   # up to +0.12 when all recent are threats

        # ── Feedback-adapted severity weight ─────────────────────────────
        fb_adj    = self.feedback.adjustment(LAYER)
        sev_key   = str(label).lower()
        base_sev  = BASE_SEV_WEIGHT.get(sev_key, 0.40)

        # L1 context boost
        l1_pred   = event.pipeline.get("ingestion",{}).get("prediction","")
        l1_boost  = 0.10 if str(l1_pred).lower() not in BENIGN_LABELS else 0.0

        # Adversarial severity override
        adv_boost = 0.20 if adv["is_adversarial"] else 0.0

        risk_contrib = float(np.clip(
            conf * base_sev + l1_boost + cluster_boost + adv_boost + fb_adj,
            0.0, 1.0))
        event.feedback_applied[LAYER] = fb_adj

        event.record(LAYER, label, conf, risk_contrib)

        # ── Agent memory push ────────────────────────────────────────────
        self.memory.push(event.event_id, LAYER, {
            "prediction": label, "confidence": conf,
            "risk_contrib": risk_contrib,
            "cluster_boost": cluster_boost,
            "adversarial": adv["is_adversarial"],
        })

        # ── Reasoning trace ──────────────────────────────────────────────
        reasoning = self.agent.explain(
            event_id    = event.event_id,
            prediction  = label,
            confidence  = conf,
            risk_score  = event.risk_score,
            pipeline_ctx= event.pipeline,
            mitre        = event.mitre,
            evasion_hint = adv.get("hint",""),
        )
        # Append temporal context note
        if cluster_boost > 0.05:
            reasoning += (f"\n  ⚡ Temporal cluster: {threat_frac:.0%} of last "
                          f"{len(win)} events were threats (boost={cluster_boost:.3f})")
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
        self.log.banner(f"L2 TRIAGE AGENT DEMO  │  {dataset.upper()}")
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
            ev = SOCEvent(X_te[idx], dataset, f"TRG-{i+1:04d}")
            self.infer(ev); time.sleep(0.05)
        print(self.agent.summarise_session(self.memory))

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
        self.log.banner("TRIAGE — SUMMARY")
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
    L2   = TriageLayer()
    if args.mode == "train":
        if args.dataset == "all": L2.train_all(tune=not args.no_tune)
        else: L2.train(args.dataset, tune=not args.no_tune)
    else:
        ds = "cicids2017" if args.dataset == "all" else args.dataset
        L2.run_demo(ds)