"""
Layer 4 — SIEM  (Agentic upgrade)
────────────────────────────────────────────────────────────────────────────────
What's new vs baseline:
  • ReasoningAgent  — NL correlation narrative (multi-stage attack chains)
  • Campaign correlator  — detects kill-chain stage sequences across events
  • FeedbackBus  — analyst validation of correlated incidents shifts weights
  • AgentMemory  — cross-event context for identifying APT campaigns
  • Adversarial awareness  — flags events that span multiple evasion signals
────────────────────────────────────────────────────────────────────────────────
"""
import time, argparse
from collections import deque
import numpy as np
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from p1_soc_utils import (
    Logger, SOCEvent, DataLoader, MetricsEngine, Store, PlotEngine,
    DriftDetector, AgentMemory, FeedbackBus, ReasoningAgent, AdversarialDetector,
    DATASETS, MITRE_MAP, BENIGN_LABELS, RESET, BOLD,
)

LAYER      = "siem"
WINDOW_SZ  = 100

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

INCIDENT_LABELS = {
    "incident","correlated_alert","apt","campaign","lateral_movement",
    "data_exfiltration","c2_activity","1","attack","anomaly","high","critical",
    "correlated","alert","malformed","suspicious","syslog_attack","auth_failure",
}

# MITRE kill-chain stage ordering (simplified)
KILL_CHAIN_ORDER = [
    {"recon", "portscan", "port scan", "discovery"},
    {"initial_access", "bruteforce", "brute force", "infiltration", "heartbleed",
     "ftp-patator", "ssh-patator", "web attack", "sql injection", "xss"},
    {"execution", "malware", "trojan", "injection"},
    {"persistence", "backdoor"},
    {"lateral_movement"},
    {"c2", "beacon", "botnet"},
    {"exfiltration", "data_exfiltration"},
    {"impact", "ransomware", "ddos"},
]

def _kill_chain_stage(label: str) -> int:
    """Return the kill-chain stage index for a label, or -1 if benign."""
    l = label.lower()
    for i, stage_set in enumerate(KILL_CHAIN_ORDER):
        if any(s in l for s in stage_set):
            return i
    return -1


class SIEMLayer:
    def __init__(self, memory: AgentMemory = None,
                 feedback: FeedbackBus    = None):
        self.models    = {}
        self.scalers   = {}
        self.encoders  = {}
        self.les       = {}
        self.feat_names= {}
        self.drifters  = {}
        self._windows  = {ds: deque(maxlen=WINDOW_SZ) for ds in DATASETS}
        self.log       = Logger(LAYER)

        # ── Agentic components ───────────────────────────────────────────
        self.memory    = memory   or AgentMemory()
        self.feedback  = feedback or FeedbackBus()
        self.agent     = ReasoningAgent(LAYER)
        self.adv_det   = AdversarialDetector()

        # Campaign tracker: dataset → deque of (event_id, kill_chain_stage, label)
        self._campaign: dict = {ds: deque(maxlen=30) for ds in DATASETS}

    # ═══════════════════════════════════════════════════════════════════════
    #  TRAIN
    # ═══════════════════════════════════════════════════════════════════════
    def train(self, dataset: str, tune: bool = True) -> dict:
        log = Logger(LAYER, dataset)
        log.banner(
            f"LAYER 4 — SIEM  │  RF + Sliding Window + Campaign Correlator  │  {dataset.upper()}")

        loader = DataLoader(log)
        res    = loader.load(dataset, LAYER)
        if res[0] is None:
            log.err(f"No data for {dataset}/{LAYER}"); return {}

        X_tr, y_tr, X_te, y_te, sc, enc, le, feat_names = res
        log.info(f"Correlation classes: {le.classes_.tolist()}")

        self.adv_det.fit_baseline(X_tr)
        log.ok("AdversarialDetector baseline fitted")

        # ── Tuning ───────────────────────────────────────────────────────
        if tune:
            log.section("Hyperparameter Tuning …")
            base = RandomForestClassifier(class_weight="balanced",
                                          oob_score=False, n_jobs=-1, random_state=42)
            gs   = GridSearchCV(base, PARAM_GRID, cv=3,
                                scoring="f1_weighted", n_jobs=-1, verbose=0)
            t0   = time.time()
            gs.fit(X_tr, y_tr)
            best_p = gs.best_params_
            log.ok(f"Best: {best_p}  │  {gs.best_score_:.4f}  │  {time.time()-t0:.1f}s")
            model_params = {**RF_BASE, **best_p}
        else:
            model_params = RF_BASE

        log.section("Training Random Forest …")
        model = RandomForestClassifier(**model_params)
        t0    = time.time()
        model.fit(X_tr, y_tr)
        log.ok(f"Done {time.time()-t0:.1f}s  │  OOB={model.oob_score_:.4f}")

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        metrics = MetricsEngine.compute(y_te, y_pred, y_proba, le)
        MetricsEngine.print_table(metrics, LAYER, dataset)

        n_inc = sum(1 for l in le.inverse_transform(y_pred)
                    if str(l).lower() in INCIDENT_LABELS
                    or str(l).lower() not in BENIGN_LABELS)
        log.info(f"Incidents: {n_inc:,}/{len(y_pred):,} ({100*n_inc/len(y_pred):.1f}%)")

        log.section("5-Fold Cross Validation …")
        cv_res = MetricsEngine.kfold_cv(
            RandomForestClassifier, model_params, X_tr, y_tr, k=5, log=log)

        log.section("Generating plots …")
        PlotEngine.roc(y_te, y_proba, le, LAYER, dataset)
        PlotEngine.pr_curve(y_te, y_proba, le, LAYER, dataset)
        PlotEngine.confusion(y_te, y_pred, le, LAYER, dataset)
        PlotEngine.feature_importance(model, feat_names, LAYER, dataset)
        PlotEngine.cv_results(cv_res, LAYER, dataset)
        PlotEngine.learning_curve(
            RandomForestClassifier, {**model_params,"oob_score":False,"n_jobs":-1},
            X_tr, y_tr, LAYER, dataset)
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

    def train_all(self, tune=True):
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
        self.log.ok(f"SIEM loaded: {dataset}")

    # ═══════════════════════════════════════════════════════════════════════
    #  CAMPAIGN CORRELATION  — kill-chain stage tracker
    # ═══════════════════════════════════════════════════════════════════════
    def _correlate_campaign(self, ds: str, event_id: str,
                            label: str) -> tuple[float, str]:
        """
        Check if the current event advances a multi-stage kill chain.
        Returns (campaign_boost, campaign_narrative).
        """
        stage = _kill_chain_stage(label)
        campaign = self._campaign[ds]
        campaign.append((event_id, stage, label))

        if stage < 0:
            return 0.0, ""

        # Look for ascending kill-chain stages within the window
        stages = [s for _, s, _ in campaign if s >= 0]
        if len(stages) < 2:
            return 0.0, ""

        # Count how many unique ascending stages we've seen
        unique_asc = sorted(set(stages))
        progression = len(unique_asc)

        if progression >= 4:
            boost = 0.30
            narr  = (f"Multi-stage APT campaign detected: "
                     f"{progression} kill-chain stages observed "
                     f"({' → '.join(str(s) for s in unique_asc)}). "
                     f"High confidence of coordinated attack.")
        elif progression >= 2:
            boost = 0.12
            narr  = (f"Kill-chain progression: {progression} stages seen "
                     f"({' → '.join(str(s) for s in unique_asc)}). "
                     f"Possible early-stage campaign.")
        else:
            boost = 0.0
            narr  = ""

        return boost, narr

    # ═══════════════════════════════════════════════════════════════════════
    #  REAL-TIME INFERENCE  (agentic)
    # ═══════════════════════════════════════════════════════════════════════
    def infer(self, event: SOCEvent) -> SOCEvent:
        ds = event.dataset
        if ds not in self.models:
            try:   self.load(ds)
            except FileNotFoundError:
                event.record(LAYER, "unknown", 0.0, 0.5); return event

        # Sliding window deviation
        window = self._windows[ds]
        if len(window) >= 5:
            wa  = np.stack(list(window))
            dim = min(wa.shape[1], event.raw.shape[0])
            dev = float(np.abs(event.raw[:dim] - wa[:,:dim].mean(0)).mean())
            dev = min(dev / 10.0, 0.20)
        else:
            dev = 0.0
        window.append(event.raw)

        label, conf, proba = Store.infer(
            self.models[ds], self.scalers[ds], self.les[ds], event.raw)

        # ── Adversarial check ────────────────────────────────────────────
        adv = self.adv_det.check(event.raw, proba, label, event.event_id)
        event.record_adversarial(LAYER, adv)
        if adv["is_adversarial"]:
            self.feedback.emit(LAYER, event.event_id, "adversarial_detected",
                               adv["score"], adv["hint"])

        # ── Campaign correlation ─────────────────────────────────────────
        campaign_boost, campaign_narr = self._correlate_campaign(ds, event.event_id, label)

        # ── Cross-layer threat count ─────────────────────────────────────
        n_threat = sum(
            1 for lyr in ["ingestion","triage","detection"]
            if str(event.pipeline.get(lyr,{}).get("prediction","")).lower()
               not in BENIGN_LABELS)
        ctx_boost = n_threat * 0.05

        # ── Feedback adjustment ──────────────────────────────────────────
        fb_adj = self.feedback.adjustment(LAYER)

        is_incident  = (str(label).lower() in INCIDENT_LABELS or
                        str(label).lower() not in BENIGN_LABELS)
        adv_boost    = 0.15 if adv["is_adversarial"] else 0.0
        risk_contrib = float(np.clip(
            conf*(0.90 if is_incident else 0.06) + dev + ctx_boost
            + campaign_boost + adv_boost + fb_adj,
            0.0, 1.0))
        event.feedback_applied[LAYER] = fb_adj

        event.record(LAYER, label, conf, risk_contrib)

        # ── Agent memory push ────────────────────────────────────────────
        self.memory.push(event.event_id, LAYER, {
            "prediction": label, "confidence": conf,
            "risk_contrib": risk_contrib,
            "campaign_boost": campaign_boost,
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
        if campaign_narr:
            reasoning += f"\n  🔗 CAMPAIGN CORRELATION: {campaign_narr}"
        if dev > 0.05:
            reasoning += f"\n  📈 Sliding-window deviation={dev:.3f} (temporal anomaly)"
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
        self.log.banner(f"L4 SIEM AGENT DEMO  │  {dataset.upper()}")
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
            ev = SOCEvent(X_te[idx], dataset, f"SIM-{i+1:04d}")
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
        self.log.banner("SIEM — SUMMARY")
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
    L4   = SIEMLayer()
    if args.mode == "train":
        if args.dataset == "all": L4.train_all(tune=not args.no_tune)
        else: L4.train(args.dataset, tune=not args.no_tune)
    else:
        ds = "cicids2017" if args.dataset == "all" else args.dataset
        L4.run_demo(ds)