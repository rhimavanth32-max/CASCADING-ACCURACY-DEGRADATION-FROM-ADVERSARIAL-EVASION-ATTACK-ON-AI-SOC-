"""
Layer 5 — SOAR  (Agentic upgrade)
────────────────────────────────────────────────────────────────────────────────
What's new vs baseline:
  • ReasoningAgent  — full NL incident narrative + decision justification
  • Adaptive playbook generation  — playbook steps selected dynamically
    based on MITRE mappings, adversarial flags, and campaign context
  • FeedbackBus  — analyst accept/reject of playbooks refines future actions
  • AgentMemory  — session-level incident log for pattern-aware escalation
  • Confidence-gated escalation  — low-confidence SOAR decisions always page
    a human analyst rather than acting autonomously
────────────────────────────────────────────────────────────────────────────────
"""
import time, json, argparse
from pathlib import Path
import numpy as np
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from p1_soc_utils import (
    Logger, SOCEvent, DataLoader, MetricsEngine, Store, PlotEngine,
    DriftDetector, AgentMemory, FeedbackBus, ReasoningAgent, AdversarialDetector,
    DATASETS, RESULT_ROOT, MITRE_MAP, BENIGN_LABELS,
    RESET, BOLD, GREEN, YELLOW, RED,
)

LAYER = "soar"

GB_BASE = dict(
    n_estimators=250, learning_rate=0.07, max_depth=8,
    subsample=0.85, min_samples_split=4, min_samples_leaf=2,
    max_features="sqrt", n_iter_no_change=25,
    validation_fraction=0.10, tol=1e-5, random_state=42,
)
PARAM_GRID = {
    "n_estimators"  : [150, 250],
    "learning_rate" : [0.05, 0.07, 0.10],
    "max_depth"     : [6, 8],
}

ACTION_RISK = {
    "allow":0.0, "safe":0.0, "normal":0.0, "benign":0.0, "0":0.0,
    "monitor":0.20, "log":0.15,
    "patch_required":0.45, "patch":0.45,
    "escalate":0.65, "alert":0.60, "warn":0.50,
    "block":0.85, "quarantine":0.80,
    "isolate":0.95, "critical":1.0,
}

# Base playbooks — extended dynamically by agent
BASE_PLAYBOOK = {
    "allow"         : ["✅ Log to SIEM audit trail",
                       "Update traffic baseline"],
    "monitor"       : ["👁️  Add to watchlist (24h)",
                       "Increase log verbosity",
                       "Schedule analyst review in 4h"],
    "patch_required": ["🔧 Create patch ticket (P3)",
                       "Notify system owner",
                       "Apply compensating controls",
                       "Re-scan asset in 48h"],
    "escalate"      : ["📟 Page on-call analyst",
                       "Open P2 incident ticket",
                       "Preserve network pcap evidence",
                       "Notify team lead"],
    "block"         : ["🚫 Block source IP in firewall",
                       "Update WAF rule set",
                       "Notify network security team",
                       "Open P2 incident ticket"],
    "quarantine"    : ["🔒 Quarantine host from network",
                       "Block all outbound connections",
                       "Open P1 incident ticket",
                       "Initiate forensic collection"],
    "isolate"       : ["🔴 Network isolate endpoint IMMEDIATELY",
                       "Open P1 critical incident",
                       "Start forensic memory collection",
                       "Notify CISO within 15 minutes",
                       "Preserve disk image",
                       "Engage IR response team"],
    "critical"      : ["🚨 FULL IR ACTIVATION",
                       "Isolate entire affected network segment",
                       "Notify CISO + Legal + Board",
                       "Engage external IR retainer",
                       "Preserve all evidence (chain-of-custody)",
                       "Notify regulatory body if PII breach",
                       "Open bridge call with C-suite"],
}

# MITRE-driven playbook additions
MITRE_PLAYBOOK_EXTRAS = {
    "T1486": ["💾 Verify offline backups integrity",
               "Scan for additional encrypted file paths"],
    "T1071": ["📡 Block C2 domains at DNS/proxy layer",
               "Analyse beacon timing and jitter pattern"],
    "T1041": ["📤 Block exfil IP/port on perimeter firewall",
               "Audit DLP logs for data volume anomalies"],
    "T1110": ["🔑 Force password reset for targeted accounts",
               "Enable account lockout after 3 failed attempts"],
    "T1498": ["📶 Activate DDoS mitigation (scrubbing centre)",
               "Contact upstream ISP for rate-limiting"],
    "T1190": ["🌐 Patch vulnerable endpoint within 4h",
               "Review WAF logs for exploit payload signatures"],
}

# Confidence threshold below which human escalation is mandatory
AUTONOMOUS_CONF_THRESHOLD = 0.65


class SOARLayer:
    def __init__(self, memory: AgentMemory = None,
                 feedback: FeedbackBus    = None):
        self.models       = {}
        self.scalers      = {}
        self.encoders     = {}
        self.les          = {}
        self.feat_names   = {}
        self.drifters     = {}
        self.incident_log = []
        self.log          = Logger(LAYER)

        # ── Agentic components ───────────────────────────────────────────
        self.memory    = memory   or AgentMemory()
        self.feedback  = feedback or FeedbackBus()
        self.agent     = ReasoningAgent(LAYER)
        self.adv_det   = AdversarialDetector()

        # Playbook feedback tracker: action → (accepted, rejected)
        self._pb_feedback: dict = {}

    # ═══════════════════════════════════════════════════════════════════════
    #  TRAIN
    # ═══════════════════════════════════════════════════════════════════════
    def train(self, dataset: str, tune: bool = True) -> dict:
        log = Logger(LAYER, dataset)
        log.banner(
            f"LAYER 5 — SOAR  │  Gradient Boosting + Adaptive Playbook Agent  │  {dataset.upper()}")

        loader = DataLoader(log)
        res    = loader.load(dataset, LAYER)
        if res[0] is None:
            log.err(f"No data for {dataset}/{LAYER}"); return {}

        X_tr, y_tr, X_te, y_te, sc, enc, le, feat_names = res
        log.info(f"Response actions: {le.classes_.tolist()}")

        self.adv_det.fit_baseline(X_tr)
        log.ok("AdversarialDetector baseline fitted")

        if tune:
            log.section("Hyperparameter Tuning …")
            base = GradientBoostingClassifier(random_state=42)
            gs   = GridSearchCV(base, PARAM_GRID, cv=3,
                                scoring="f1_weighted", n_jobs=-1, verbose=0)
            t0   = time.time()
            gs.fit(X_tr, y_tr)
            best_p = gs.best_params_
            log.ok(f"Best: {best_p}  │  {gs.best_score_:.4f}  │  {time.time()-t0:.1f}s")
            model_params = {**GB_BASE, **best_p}
        else:
            model_params = GB_BASE

        log.section("Training Gradient Boosting …")
        model = GradientBoostingClassifier(**model_params)
        t0    = time.time()
        model.fit(X_tr, y_tr)
        log.ok(f"Done {time.time()-t0:.1f}s  │  Iters={model.n_estimators_}")

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        metrics = MetricsEngine.compute(y_te, y_pred, y_proba, le)
        MetricsEngine.print_table(metrics, LAYER, dataset)

        lp  = le.inverse_transform(y_pred)
        vals, cts = np.unique(lp, return_counts=True)
        sorted_acts = sorted(zip(vals,cts),
                             key=lambda x: ACTION_RISK.get(str(x[0]).lower(), 0.5),
                             reverse=True)
        log.info("Response action distribution (test):")
        for act, cnt in sorted_acts:
            log.info(f"  {str(act):<22} {cnt:>5,}  ({100*cnt/len(y_pred):.1f}%)")

        log.section("5-Fold Cross Validation …")
        cv_res = MetricsEngine.kfold_cv(
            GradientBoostingClassifier, model_params, X_tr, y_tr, k=5, log=log)

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
        self.log.ok(f"SOAR loaded: {dataset}")

    # ═══════════════════════════════════════════════════════════════════════
    #  ADAPTIVE PLAYBOOK GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    def _build_playbook(self, action: str, mitre: dict,
                        adversarial_flags: dict, conf: float) -> list:
        """
        Dynamically compose a playbook:
          1. Base steps for the action
          2. MITRE-specific additions based on mapped techniques
          3. Adversarial response steps if evasion was detected
          4. Mandatory human review step if confidence is below threshold
        """
        pb = list(BASE_PLAYBOOK.get(action,
                                    ["Review event manually",
                                     "Update detection rules"]))

        # Add MITRE-specific steps
        tids_seen = set()
        for m in mitre.values():
            tid = m.get("tid","")
            if tid and tid not in tids_seen:
                tids_seen.add(tid)
                for step in MITRE_PLAYBOOK_EXTRAS.get(tid, []):
                    if step not in pb:
                        pb.append(step)

        # Adversarial response
        any_adv = any(v.get("is_adversarial") for v in adversarial_flags.values())
        if any_adv:
            adv_steps = [
                "🧬 Collect adversarial sample for ML retraining pipeline",
                "📋 Submit evasion indicator to threat-intel sharing platform",
                "🔁 Flag model for expedited drift check",
            ]
            for s in adv_steps:
                if s not in pb: pb.append(s)

        # Low-confidence mandatory human review
        if conf < AUTONOMOUS_CONF_THRESHOLD:
            review_step = (f"🧑 MANDATORY ANALYST REVIEW — SOAR confidence "
                           f"{conf:.2f} < threshold {AUTONOMOUS_CONF_THRESHOLD:.2f}. "
                           f"Do not execute block/isolate autonomously.")
            if review_step not in pb:
                pb.insert(0, review_step)

        return pb

    # ═══════════════════════════════════════════════════════════════════════
    #  REAL-TIME INFERENCE  (agentic)
    # ═══════════════════════════════════════════════════════════════════════
    def infer(self, event: SOCEvent) -> SOCEvent:
        ds = event.dataset
        if ds not in self.models:
            try:   self.load(ds)
            except FileNotFoundError:
                event.record(LAYER, "escalate", 0.0, 0.65)
                event.decide(); return event

        label, conf, proba = Store.infer(
            self.models[ds], self.scalers[ds], self.les[ds], event.raw)

        # ── Adversarial check ────────────────────────────────────────────
        adv = self.adv_det.check(event.raw, proba, label, event.event_id)
        event.record_adversarial(LAYER, adv)
        if adv["is_adversarial"]:
            # Override to escalate when SOAR itself is being evaded
            if ACTION_RISK.get(str(label).lower(), 0.5) < 0.65:
                label = "escalate"
                self.log.warn(
                    f"[SOAR] Adversarial signal at response layer — "
                    f"overriding action to 'escalate' for human review")
            self.feedback.emit(LAYER, event.event_id, "adversarial_detected",
                               adv["score"], adv["hint"])

        # ── Confidence-gated action ──────────────────────────────────────
        if conf < AUTONOMOUS_CONF_THRESHOLD:
            self.log.warn(
                f"[SOAR] Low confidence ({conf:.3f}) — gating autonomous action, "
                f"mandatory analyst review inserted into playbook")
            self.feedback.emit(LAYER, event.event_id, "low_confidence",
                               conf, f"action={label}")

        # ── Feedback-driven risk adjustment ─────────────────────────────
        fb_adj       = self.feedback.adjustment(LAYER)
        action_weight= ACTION_RISK.get(str(label).lower(), 0.50)
        risk_contrib = float(np.clip(conf * action_weight + fb_adj, 0.0, 1.0))
        event.feedback_applied[LAYER] = fb_adj

        event.record(LAYER, label, conf, risk_contrib)
        event.decide()

        # ── Adaptive playbook ────────────────────────────────────────────
        playbook = self._build_playbook(
            action          = str(label).lower(),
            mitre           = event.mitre,
            adversarial_flags = event.adversarial,
            conf            = conf,
        )
        event.pipeline[LAYER]["playbook"] = playbook

        # ── Agent memory push ────────────────────────────────────────────
        self.memory.push(event.event_id, LAYER, {
            "prediction": label, "confidence": conf,
            "risk_contrib": risk_contrib,
            "action": event.final_action,
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
        # Append playbook summary to reasoning
        reasoning += (
            f"\n  📋 ADAPTIVE PLAYBOOK ({len(playbook)} steps):\n"
            + "\n".join(f"     {i+1}. {s}" for i, s in enumerate(playbook[:5]))
        )
        if len(playbook) > 5:
            reasoning += f"\n     ... and {len(playbook)-5} more steps"
        reasoning += f"\n  🎯 FINAL DECISION: {event.final_action}"
        event.record_reasoning(LAYER, reasoning)
        self.log.reasoning(reasoning)

        # ── Incident report ──────────────────────────────────────────────
        event.pipeline[LAYER]["incident_report"] = {
            "incident_id"        : f"INC-{event.event_id}",
            "timestamp"          : event.timestamp,
            "dataset"            : event.dataset,
            "risk_score"         : event.risk_score,
            "soar_action"        : str(label),
            "confidence"         : round(conf, 4),
            "final_decision"     : event.final_action,
            "escalated"          : event.escalate,
            "playbook_steps"     : playbook,
            "mitre_mappings"     : event.mitre,
            "adversarial_flags"  : {k: v.get("is_adversarial",False)
                                    for k,v in event.adversarial.items()},
            "feedback_adjustments": event.feedback_applied,
            "agent_reasoning"    : {k: v[-200:] for k,v in
                                    event.agent_reasoning.items()},
            "layer_summary"      : {lyr: d.get("prediction","n/a")
                                    for lyr, d in event.pipeline.items()},
        }
        self.incident_log.append(event.to_dict())
        self.log.event(event.event_id, LAYER, label, conf,
                       event.risk_score, action=event.final_action)
        return event

    # ═══════════════════════════════════════════════════════════════════════
    #  ANALYST PLAYBOOK FEEDBACK
    # ═══════════════════════════════════════════════════════════════════════
    def analyst_accept_playbook(self, event_id: str, action: str):
        """Analyst validates the playbook — reinforce action selection."""
        acc, rej = self._pb_feedback.get(action, (0, 0))
        self._pb_feedback[action] = (acc+1, rej)
        self.feedback.emit(LAYER, event_id, "confirmed_threat", 1.0,
                           f"playbook accepted for action={action}")
        self.log.ok(f"Playbook accepted: action={action} "
                    f"(accepted={acc+1}, rejected={rej})")

    def analyst_reject_playbook(self, event_id: str, action: str,
                                corrected_action: str = "escalate"):
        """Analyst rejects the playbook — penalise this action mapping."""
        acc, rej = self._pb_feedback.get(action, (0, 0))
        self._pb_feedback[action] = (acc, rej+1)
        self.memory.push_feedback(event_id, action, corrected_action)
        self.feedback.emit(LAYER, event_id, "false_positive", 1.0,
                           f"playbook rejected: {action} → {corrected_action}")
        self.log.warn(f"Playbook rejected: action={action} "
                      f"(accepted={acc}, rejected={rej+1}) "
                      f"corrected={corrected_action}")

    def analyst_correct(self, event_id: str, original: str,
                        corrected: str, analyst: str = "analyst"):
        self.memory.push_feedback(event_id, original, corrected, analyst)
        signal = ("false_positive" if corrected.lower() in BENIGN_LABELS
                  else "confirmed_threat")
        self.feedback.emit(LAYER, event_id, signal, 1.0, f"{original} → {corrected}")
        self.log.info(f"Analyst correction: {original} → {corrected} "
                      f"(adj={self.feedback.adjustment(LAYER):+.3f})")

    # ═══════════════════════════════════════════════════════════════════════
    #  DRIFT CHECK
    # ═══════════════════════════════════════════════════════════════════════
    def check_drift(self, dataset: str, new_proba: np.ndarray):
        if dataset in self.drifters:
            result = self.drifters[dataset].check(new_proba, LAYER, dataset, self.log)
            if result.get("psi", 0) > 0.20:
                self.feedback.emit(LAYER, "drift-check", "drift",
                                   result["psi"], f"PSI={result['psi']:.4f}")
            return result
        return {}

    # ═══════════════════════════════════════════════════════════════════════
    #  SAVE INCIDENT LOG
    # ═══════════════════════════════════════════════════════════════════════
    def save_incident_log(self):
        p = Path(RESULT_ROOT) / LAYER / "incident_log.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.incident_log, indent=2, default=str))
        self.log.ok(f"Incident log → {p}")

        # Also save feedback bus state
        fb_p = Path(RESULT_ROOT) / LAYER / "feedback_bus.json"
        fb_p.write_text(json.dumps(self.feedback.summary(), indent=2, default=str))
        self.log.ok(f"Feedback bus → {fb_p}")

    # ═══════════════════════════════════════════════════════════════════════
    #  DEMO
    # ═══════════════════════════════════════════════════════════════════════
    def run_demo(self, dataset: str, n: int = 10):
        self.log.banner(f"L5 SOAR AGENT DEMO  │  {dataset.upper()}")
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
            ev = SOCEvent(X_te[idx], dataset, f"SOA-{i+1:04d}")
            self.infer(ev)
            pb = ev.pipeline[LAYER].get("playbook",[])
            if pb: print(f"    Playbook: {' → '.join(pb[:3])}")
            if ev.mitre: print(f"    MITRE: {list(ev.mitre.values())}")
            print()
            time.sleep(0.08)
        self.save_incident_log()
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
        self.log.banner("SOAR — SUMMARY")
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
    L5   = SOARLayer()
    if args.mode == "train":
        if args.dataset == "all": L5.train_all(tune=not args.no_tune)
        else: L5.train(args.dataset, tune=not args.no_tune)
    else:
        ds = "cicids2017" if args.dataset == "all" else args.dataset
        L5.run_demo(ds)