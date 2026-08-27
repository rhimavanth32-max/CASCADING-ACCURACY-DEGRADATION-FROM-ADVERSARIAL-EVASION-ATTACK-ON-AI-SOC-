import os, json, time, warnings
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy  as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, average_precision_score,
    precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils           import resample

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("processed_data")
MODEL_ROOT  = Path("ai_soc_models")
RESULT_ROOT = Path("ai_soc_results")
PLOT_ROOT   = Path("ai_soc_plots")
LOG_ROOT    = Path("ai_soc_logs")

CICIDS_DIR  = DATA_ROOT / "cicids2017"
EMBER_DIR   = DATA_ROOT / "ember"
LOGHUB_DIR  = DATA_ROOT / "loghub"

# ── Dataset / layer file map ──────────────────────────────────────────────────
LAYER_FILES = {
    "cicids2017": {
        "ingestion": ("ingestion_train.csv", "ingestion_test.csv"),
        "triage"   : ("triage_train.csv",    "triage_test.csv"),
        "detection": ("detection_train.csv",  "detection_test.csv"),
        "siem"     : ("siem_train.csv",       "siem_test.csv"),
        "soar"     : ("soar_train.csv",       "soar_test.csv"),
    },
    "ember": {
        "ingestion": ("ember_train.csv", "ember_test.csv"),
        "triage"   : ("ember_train.csv", "ember_test.csv"),
        "detection": ("ember_train.csv", "ember_test.csv"),
        "siem"     : ("ember_train.csv", "ember_test.csv"),
        "soar"     : ("ember_train.csv", "ember_test.csv"),
    },
    "loghub": {
        "ingestion": ("ingestion_train.csv", "ingestion_test.csv"),
        "triage"   : ("triage_train.csv",    "triage_test.csv"),
        "detection": ("detection_train.csv",  "detection_test.csv"),
        "siem"     : ("siem_train.csv",       "siem_test.csv"),
        "soar"     : ("soar_train.csv",       "soar_test.csv"),
    },
}
DATASET_DIRS = {"cicids2017": CICIDS_DIR, "ember": EMBER_DIR, "loghub": LOGHUB_DIR}
DATASETS     = ["cicids2017", "ember", "loghub"]
LAYERS       = ["ingestion", "triage", "detection", "siem", "soar"]

LAYER_META = {
    "ingestion": {"algo": "Random Forest",      "color": "\033[94m"},
    "triage"   : {"algo": "Gradient Boosting",  "color": "\033[93m"},
    "detection": {"algo": "MLP Neural Network", "color": "\033[91m"},
    "siem"     : {"algo": "Random Forest",      "color": "\033[95m"},
    "soar"     : {"algo": "Gradient Boosting",  "color": "\033[92m"},
}

LABEL_CANDIDATES = [
    "label","Label","LABEL","class","Class","CLASS","target","Target",
    "attack_type","Attack","category","y","Y","tag","Tag","type","Type",
    "response_action","priority","alert_priority","correlation_label",
    "soar_action","threat_label","event_type",
]

# ── MITRE ATT&CK Mapping ──────────────────────────────────────────────────────
MITRE_MAP = {
    "ddos"              : ("Impact",             "Network Denial of Service",        "T1498"),
    "dosslowloris"      : ("Impact",             "Endpoint Denial of Service",       "T1499"),
    "dosgoldeneye"      : ("Impact",             "Endpoint Denial of Service",       "T1499"),
    "doshulk"           : ("Impact",             "Endpoint Denial of Service",       "T1499"),
    "dosrudy"           : ("Impact",             "Endpoint Denial of Service",       "T1499"),
    "portscan"          : ("Discovery",          "Network Service Discovery",        "T1046"),
    "port scan"         : ("Discovery",          "Network Service Discovery",        "T1046"),
    "bruteforce"        : ("Credential Access",  "Brute Force",                      "T1110"),
    "brute force"       : ("Credential Access",  "Brute Force",                      "T1110"),
    "ftp-patator"       : ("Credential Access",  "Brute Force: Password Spraying",   "T1110.003"),
    "ssh-patator"       : ("Credential Access",  "Brute Force: Password Spraying",   "T1110.003"),
    "exfiltration"      : ("Exfiltration",       "Exfiltration Over C2 Channel",     "T1041"),
    "infiltration"      : ("Initial Access",     "Exploit Public-Facing Application","T1190"),
    "botnet"            : ("Command & Control",  "Application Layer Protocol",       "T1071"),
    "heartbleed"        : ("Initial Access",     "Exploit Public-Facing Application","T1190"),
    "web attack"        : ("Initial Access",     "Exploit Public-Facing Application","T1190"),
    "sql injection"     : ("Initial Access",     "Exploit Public-Facing Application","T1190"),
    "xss"               : ("Initial Access",     "Drive-by Compromise",              "T1189"),
    "lateral_movement"  : ("Lateral Movement",   "Remote Services",                  "T1021"),
    "c2"                : ("Command & Control",  "Application Layer Protocol",       "T1071"),
    "beacon"            : ("Command & Control",  "Web Protocols",                    "T1071.001"),
    "ransomware"        : ("Impact",             "Data Encrypted for Impact",        "T1486"),
    "malware"           : ("Execution",          "User Execution: Malicious File",   "T1204.002"),
    "trojan"            : ("Execution",          "User Execution: Malicious File",   "T1204.002"),
    "backdoor"          : ("Persistence",        "Server Software Component",        "T1505"),
    "injection"         : ("Privilege Escalation","Exploitation for Privilege Esc.", "T1068"),
    "correlated_alert"  : ("Discovery",          "Network Service Discovery",        "T1046"),
    "incident"          : ("Impact",             "Service Stop",                     "T1489"),
    "monitor"           : ("Discovery",          "System Information Discovery",     "T1082"),
    "escalate"          : ("Privilege Escalation","Valid Accounts",                  "T1078"),
    "block"             : ("Defense Evasion",    "Impair Defenses",                  "T1562"),
    "isolate"           : ("Impact",             "Service Stop",                     "T1489"),
    "suspicious"        : ("Discovery",          "System Network Config Discovery",  "T1016"),
    "malformed"         : ("Defense Evasion",    "Obfuscated Files or Information",  "T1027"),
    "syslog_attack"     : ("Collection",         "Data from Local System",           "T1005"),
    "auth_failure"      : ("Credential Access",  "Brute Force",                      "T1110"),
    "anomaly"           : ("Discovery",          "Network Sniffing",                 "T1040"),
    "error"             : ("Defense Evasion",    "Indicator Removal",                "T1070"),
}

BENIGN_LABELS = {"benign","normal","0","safe","clean","legitimate","allow",
                 "low","false_positive","ham","0.0","info","false","negative"}

NUMERIC_LABEL_MAP = {
    "0": "benign", "0.0": "benign", "1": "attack", "1.0": "attack",
    "2": "ddos",   "3": "portscan", "4": "bruteforce",
    "5": "exfiltration", "6": "malware",
}

def resolve_label(label: str) -> str:
    return NUMERIC_LABEL_MAP.get(str(label).strip(), str(label))

# ── Adversarial evasion signatures ───────────────────────────────────────────
# Feature patterns that suggest crafted / evasion traffic
EVASION_SIGNATURES = {
    "low_ttl_high_payload": "Possible TTL-manipulation evasion",
    "fragment_overlap"    : "IP fragmentation overlap (evasion)",
    "port_hop"            : "Port-hopping C2 channel detected",
    "mimicry_benign"      : "Statistical mimicry of benign traffic",
    "low_entropy_payload" : "Encrypted/obfuscated payload with benign headers",
}

# ── Colours ───────────────────────────────────────────────────────────────────
RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"
GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; GREY="\033[37m"
MAGENTA="\033[95m"; BLUE="\033[94m"


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT MEMORY  — cross-layer context store with sliding history
# ══════════════════════════════════════════════════════════════════════════════
class AgentMemory:
    """
    Shared working memory for reasoning agents.
    Stores per-event context + global feedback signals from every layer.
    """
    def __init__(self, maxlen: int = 500):
        self._events: deque = deque(maxlen=maxlen)
        self._feedback: deque = deque(maxlen=200)   # analyst corrections
        self._threat_counts: dict = {}
        self._evasion_flags: deque = deque(maxlen=100)

    def push(self, event_id: str, layer: str, context: dict):
        self._events.append({
            "event_id": event_id, "layer": layer,
            "ts": datetime.now().isoformat(), **context
        })
        label = str(context.get("prediction","")).lower()
        if label not in BENIGN_LABELS:
            self._threat_counts[label] = self._threat_counts.get(label, 0) + 1

    def push_evasion(self, event_id: str, sig: str, score: float):
        self._evasion_flags.append({
            "event_id": event_id, "signature": sig,
            "score": score, "ts": datetime.now().isoformat()
        })

    def push_feedback(self, event_id: str, original: str, corrected: str,
                      analyst: str = "analyst"):
        """Record analyst correction for feedback-driven adaptation."""
        self._feedback.append({
            "event_id": event_id, "original": original,
            "corrected": corrected, "analyst": analyst,
            "ts": datetime.now().isoformat()
        })

    def recent_threats(self, n: int = 10) -> list:
        return [e for e in list(self._events)[-n:]
                if str(e.get("prediction","")).lower() not in BENIGN_LABELS]

    def top_threats(self, k: int = 5) -> list:
        return sorted(self._threat_counts.items(),
                      key=lambda x: x[1], reverse=True)[:k]

    def recent_evasions(self, n: int = 5) -> list:
        return list(self._evasion_flags)[-n:]

    def feedback_corrections(self) -> list:
        return list(self._feedback)

    def feedback_bias(self) -> dict:
        """Return label → correction-rate map to adjust confidence."""
        bias = {}
        for fb in self._feedback:
            orig = fb["original"]
            bias[orig] = bias.get(orig, 0) + 1
        total = len(self._feedback) or 1
        return {k: v/total for k, v in bias.items()}

    def to_dict(self) -> dict:
        return {
            "recent_events"   : list(self._events)[-20:],
            "threat_counts"   : self._threat_counts,
            "evasion_flags"   : list(self._evasion_flags)[-10:],
            "feedback_count"  : len(self._feedback),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK BUS  — pipeline-wide feedback loop
# ══════════════════════════════════════════════════════════════════════════════
class FeedbackBus:
    """
    Collects per-layer signals (confidence deltas, drift alerts, analyst
    corrections) and exposes adjustment factors so each layer can adapt
    its risk thresholds dynamically.
    """
    def __init__(self):
        self._signals: deque = deque(maxlen=300)
        self._layer_adjustments: dict = {l: 0.0 for l in LAYERS}

    def emit(self, layer: str, event_id: str, signal_type: str,
             value: float, note: str = ""):
        self._signals.append({
            "layer": layer, "event_id": event_id,
            "type": signal_type, "value": value,
            "note": note, "ts": datetime.now().isoformat()
        })
        # Incremental adjustment: drift/fp signals lower the layer's
        # risk contribution weight; tp confirms raise it slightly
        if signal_type == "false_positive":
            self._layer_adjustments[layer] = max(
                self._layer_adjustments[layer] - 0.02, -0.15)
        elif signal_type == "confirmed_threat":
            self._layer_adjustments[layer] = min(
                self._layer_adjustments[layer] + 0.01,  0.10)
        elif signal_type == "drift":
            self._layer_adjustments[layer] = max(
                self._layer_adjustments[layer] - 0.05, -0.20)

    def adjustment(self, layer: str) -> float:
        return self._layer_adjustments.get(layer, 0.0)

    def recent(self, n: int = 20) -> list:
        return list(self._signals)[-n:]

    def summary(self) -> dict:
        return {
            "total_signals"     : len(self._signals),
            "layer_adjustments" : dict(self._layer_adjustments),
            "recent_signals"    : self.recent(5),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  REASONING AGENT  — NL explanation + decision rationale per layer
# ══════════════════════════════════════════════════════════════════════════════
class ReasoningAgent:
    """
    Interprets ML predictions and generates natural-language reasoning.
    Each layer instantiates its own agent with a role description.
    """
    ROLE_PROMPTS = {
        "ingestion": (
            "You are the Ingestion Analyst. Your job is to assess whether "
            "raw network events show signs of known attack patterns or "
            "structural anomalies before they enter the pipeline."
        ),
        "triage": (
            "You are the Triage Analyst. You prioritise events by severity "
            "using context from ingestion, statistical deviation, and "
            "temporal clustering of similar events."
        ),
        "detection": (
            "You are the Detection Analyst. You interpret deep neural network "
            "outputs to classify specific attack types and flag zero-day "
            "anomalies that fall outside the training distribution."
        ),
        "siem": (
            "You are the SIEM Correlation Analyst. You correlate events across "
            "time windows and pipeline layers to identify multi-stage attacks, "
            "lateral movement, and coordinated campaigns."
        ),
        "soar": (
            "You are the SOAR Response Orchestrator. You determine the "
            "automated response action, generate a prioritised playbook, "
            "and escalate to human analysts when confidence is insufficient."
        ),
    }

    def __init__(self, layer: str):
        self.layer = layer
        self.role  = self.ROLE_PROMPTS.get(layer, "You are a SOC analyst.")

    def explain(self, event_id: str, prediction: str, confidence: float,
                risk_score: float, pipeline_ctx: dict,
                mitre: dict, is_anomaly: bool = False,
                evasion_hint: str = "") -> str:
        """
        Generate a structured natural-language reasoning trace for the event.
        This is the 'explainability' output analysts see.
        """
        pred_clean = resolve_label(prediction)
        is_threat  = pred_clean.lower() not in BENIGN_LABELS

        # Build cross-layer context sentence
        ctx_parts = []
        for lyr in ["ingestion","triage","detection","siem"]:
            ldata = pipeline_ctx.get(lyr, {})
            if ldata:
                ctx_parts.append(
                    f"{lyr.upper()}={resolve_label(str(ldata.get('prediction','?')))}"
                    f"(conf={ldata.get('confidence',0):.2f})"
                )
        ctx_str = " → ".join(ctx_parts) if ctx_parts else "No prior layers"

        # MITRE context
        mitre_strs = []
        for lyr, m in mitre.items():
            if m:
                mitre_strs.append(
                    f"{m.get('tid','?')} {m.get('technique','?')} [{m.get('tactic','?')}]"
                )
        mitre_str = "; ".join(mitre_strs) if mitre_strs else "None mapped"

        # Adversarial hint
        adv_note = (f"\n  ⚠  ADVERSARIAL SIGNAL: {evasion_hint}" if evasion_hint else "")

        # Confidence qualifier
        if confidence >= 0.90:
            conf_qual = "very high confidence"
        elif confidence >= 0.70:
            conf_qual = "moderate-to-high confidence"
        elif confidence >= 0.50:
            conf_qual = "moderate confidence"
        else:
            conf_qual = "low confidence — recommend analyst review"

        # Anomaly note
        anom_note = (" [ANOMALY — outside training distribution]" if is_anomaly else "")

        reasoning = (
            f"\n{'─'*70}\n"
            f"  [{self.layer.upper()} AGENT]  Event: {event_id}\n"
            f"  Role: {self.role}\n"
            f"  ──\n"
            f"  Prediction : {pred_clean}{anom_note}\n"
            f"  Confidence : {confidence:.4f}  ({conf_qual})\n"
            f"  Risk Score : {risk_score:.4f}\n"
            f"  Pipeline   : {ctx_str}\n"
            f"  MITRE      : {mitre_str}"
            f"{adv_note}\n"
            f"  ──\n"
            f"  Reasoning  : "
        )

        if not is_threat:
            reasoning += (
                f"The event exhibits characteristics consistent with benign traffic. "
                f"Prediction '{pred_clean}' at {conf_qual} aligns with baseline behaviour. "
                f"No cross-layer escalation signals detected. Logging for audit trail."
            )
        else:
            reasoning += (
                f"The event is classified as '{pred_clean}' at {conf_qual}. "
            )
            if len(ctx_parts) > 1:
                reasoning += (
                    f"Cross-layer agreement ({ctx_str}) strengthens the threat assessment. "
                )
            if mitre_strs:
                reasoning += (
                    f"MITRE ATT&CK mapping ({mitre_str}) indicates this technique "
                    f"is associated with {list(mitre.values())[0].get('tactic','unknown tactic')}. "
                )
            if is_anomaly:
                reasoning += (
                    f"The Isolation Forest flags this as statistically anomalous — "
                    f"possible zero-day or previously unseen variant. "
                )
            if evasion_hint:
                reasoning += (
                    f"Adversarial evasion signature detected: {evasion_hint}. "
                    f"Standard classification confidence may be deflated by crafted features. "
                )
            reasoning += (
                f"Risk contribution to pipeline score: {risk_score:.4f}."
            )

        reasoning += f"\n{'─'*70}"
        return reasoning

    def summarise_session(self, memory: "AgentMemory") -> str:
        """Produce an end-of-session analyst summary from agent memory."""
        top = memory.top_threats(5)
        evasions = memory.recent_evasions(3)
        fb = memory.feedback_corrections()

        lines = [
            f"\n{'═'*70}",
            f"  [{self.layer.upper()} AGENT]  SESSION SUMMARY",
            f"  Top threat types observed:",
        ]
        for label, cnt in top:
            lines.append(f"    • {label:<25} {cnt:>4} events")
        if evasions:
            lines.append("  Adversarial evasion signals:")
            for ev in evasions:
                lines.append(f"    ⚠  [{ev['event_id']}] {ev['signature']} (score={ev['score']:.3f})")
        if fb:
            lines.append(f"  Analyst feedback corrections: {len(fb)}")
        lines.append(f"{'═'*70}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  ADVERSARIAL DETECTOR  — evasion-intent detection
# ══════════════════════════════════════════════════════════════════════════════
class AdversarialDetector:
    """
    Heuristic + statistical checks that run on raw feature vectors to detect
    crafted inputs designed to evade ML classifiers.

    Checks implemented:
      1. Feature-space boundary proximity  (near decision boundary = suspicious)
      2. Statistical outlier in a benign-labelled cluster  (mimicry attack)
      3. High-variance feature with near-zero model confidence  (obfuscation)
      4. Sudden feature distribution shift  (adversarial perturbation)
    """
    def __init__(self, baseline_mean: np.ndarray = None,
                 baseline_std: np.ndarray  = None):
        self.baseline_mean = baseline_mean
        self.baseline_std  = baseline_std
        self._history: deque = deque(maxlen=200)

    def fit_baseline(self, X: np.ndarray):
        self.baseline_mean = X.mean(axis=0)
        self.baseline_std  = X.std(axis=0) + 1e-9

    def check(self, x: np.ndarray, proba: np.ndarray,
              label: str, event_id: str) -> dict:
        """
        Returns dict with keys: is_adversarial (bool), score (float),
        hint (str), checks (list of triggered rules).
        """
        checks    = []
        score     = 0.0
        hint      = ""

        # 1. Decision-boundary proximity: max_prob low + not benign
        max_p = float(proba.max()) if proba is not None else 1.0
        if max_p < 0.55 and label.lower() not in BENIGN_LABELS:
            checks.append("boundary_proximity")
            score += 0.30
            hint   = "low-confidence threat near decision boundary"

        # 2. Statistical deviation from known benign baseline
        if self.baseline_mean is not None:
            dim = min(len(x), len(self.baseline_mean))
            z   = np.abs(x[:dim] - self.baseline_mean[:dim]) / self.baseline_std[:dim]
            # Paradox: very LOW z-score on a flagged event = mimicry
            if label.lower() not in BENIGN_LABELS and float(z.mean()) < 0.5:
                checks.append("benign_mimicry")
                score += 0.40
                hint  += (" | Threat traffic statistically resembles benign baseline"
                          " — possible mimicry attack")

        # 3. High feature variance + low confidence
        fvar = float(np.var(x))
        if fvar > 50.0 and max_p < 0.60:
            checks.append("high_variance_low_conf")
            score += 0.20
            hint  += " | High feature variance with low model confidence — obfuscation?"

        # 4. Temporal distribution shift
        self._history.append(x.copy())
        if len(self._history) >= 20:
            hist = np.stack(list(self._history)[-20:])
            dim  = min(x.shape[0], hist.shape[1])
            cur  = x[:dim]
            mu   = hist[:, :dim].mean(0)
            sd   = hist[:, :dim].std(0) + 1e-9
            z2   = np.abs(cur - mu) / sd
            if float(z2.mean()) > 3.5:
                checks.append("distribution_shift")
                score += 0.25
                hint  += " | Sudden feature distribution shift — adversarial perturbation?"

        is_adv = score >= 0.40
        return {
            "is_adversarial": is_adv,
            "score"         : min(score, 1.0),
            "hint"          : hint.strip(" |"),
            "checks"        : checks,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self, layer="pipeline", dataset=""):
        self.layer   = layer
        self.dataset = dataset
        self.color   = LAYER_META.get(layer, {}).get("color", GREY)
        self._buf    = []

    def _ts(self): return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def banner(self, msg, w=74):
        line = "═" * w
        print(f"\n{self.color}{BOLD}{line}\n  {msg}\n{line}{RESET}\n")

    def section(self, msg):
        print(f"\n{self.color}{BOLD}▶  {msg}{RESET}")

    def ok(self,   msg): self._log("✔", GREEN,  msg)
    def warn(self, msg): self._log("⚠", YELLOW, msg)
    def err(self,  msg): self._log("✖", RED,    msg)
    def info(self, msg): self._log("→", GREY,   msg)

    def _log(self, sym, col, msg):
        self._buf.append(f"[{self._ts()}] {sym} {msg}")
        print(f"  {col}{sym}  {msg}{RESET}")

    def event(self, eid, layer, pred, conf, risk, action=""):
        c      = LAYER_META.get(layer, {}).get("color", GREY)
        suffix = f"  → {BOLD}{action}{RESET}" if action else ""
        print(f"  [{self._ts()}] {BOLD}{eid}{RESET}  "
              f"{c}[{layer.upper():<9}]{RESET}  "
              f"pred={BOLD}{str(pred):<22}{RESET}  "
              f"conf={conf:.4f}  risk={risk:.4f}{suffix}")

    def reasoning(self, text: str):
        """Print agent reasoning trace (cyan)."""
        print(f"{CYAN}{text}{RESET}")

    def save_log(self):
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        p = LOG_ROOT / f"{self.layer}_{self.dataset}_{datetime.now():%Y%m%d_%H%M%S}.log"
        p.write_text("\n".join(self._buf))


# ══════════════════════════════════════════════════════════════════════════════
#  SOC EVENT  — enhanced with agent_reasoning + adversarial fields
# ══════════════════════════════════════════════════════════════════════════════
class SOCEvent:
    """Carrier object flowing through all 5 layers."""
    LAYER_WEIGHTS = {
        "ingestion": 0.10, "triage": 0.20, "detection": 0.35,
        "siem": 0.20,      "soar":   0.15,
    }

    def __init__(self, raw: np.ndarray, dataset: str, eid: str = None):
        self.event_id          = eid or f"EVT-{int(time.time()*1000)%999999:06d}"
        self.timestamp         = datetime.now().isoformat()
        self.dataset           = dataset
        self.raw               = raw
        self.pipeline          = {}
        self.risk_score        = 0.0
        self.final_action      = None
        self.escalate          = False
        self.mitre             = {}
        self.agent_reasoning   = {}   # layer → NL explanation
        self.adversarial       = {}   # layer → adversarial check result
        self.feedback_applied  = {}   # layer → feedback adjustment applied

    def record(self, layer: str, pred, conf: float, risk_contrib: float):
        self.pipeline[layer] = {
            "prediction"  : str(pred),
            "confidence"  : round(float(conf),  4),
            "risk_contrib": round(float(risk_contrib), 4),
        }
        total_w = total_r = 0.0
        for lyr, data in self.pipeline.items():
            w = self.LAYER_WEIGHTS.get(lyr, 0.2)
            total_w += w
            total_r += w * data["risk_contrib"]
        self.risk_score = round(total_r / total_w if total_w else 0.0, 4)

        # MITRE mapping
        key = str(pred).lower().replace("[anomaly]","").strip()
        if key in MITRE_MAP:
            self.mitre[layer] = dict(zip(
                ["tactic","technique","tid"], MITRE_MAP[key]))

    def record_reasoning(self, layer: str, text: str):
        self.agent_reasoning[layer] = text

    def record_adversarial(self, layer: str, result: dict):
        self.adversarial[layer] = result

    def decide(self):
        soar = str(self.pipeline.get("soar",{}).get("prediction","")).lower()
        r    = self.risk_score
        # Escalate if any layer detected adversarial evasion
        adv_detected = any(
            v.get("is_adversarial", False) for v in self.adversarial.values()
        )
        if r > 0.70 or any(x in soar for x in ["block","isolate","critical"]) or adv_detected:
            self.final_action = "🔴 BLOCK & ISOLATE — P1 Incident raised"
            self.escalate     = True
        elif r > 0.40 or any(x in soar for x in ["escalate","alert","warn","medium"]):
            self.final_action = "🟡 ALERT — Escalated to SOC analyst"
            self.escalate     = True
        else:
            self.final_action = "🟢 ALLOW — Logged for audit"
            self.escalate     = False
        return self.final_action

    def to_dict(self):
        return {
            "event_id"        : self.event_id,
            "timestamp"       : self.timestamp,
            "dataset"         : self.dataset,
            "pipeline"        : self.pipeline,
            "risk_score"      : self.risk_score,
            "final_action"    : self.final_action,
            "escalate"        : self.escalate,
            "mitre"           : self.mitre,
            "agent_reasoning" : self.agent_reasoning,
            "adversarial"     : self.adversarial,
            "feedback_applied": self.feedback_applied,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER  (unchanged — oversampling for class imbalance)
# ══════════════════════════════════════════════════════════════════════════════
class DataLoader:
    def __init__(self, log: Logger = None):
        self.log = log or Logger()

    def find_label(self, df: pd.DataFrame) -> str:
        for c in LABEL_CANDIDATES:
            if c in df.columns:
                return c
        self.log.warn(f"Label not found — using last col: '{df.columns[-1]}'")
        return df.columns[-1]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        return df

    def _oversample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y
        max_count = counts.max()
        X_parts, y_parts = [X], [y]
        for cls, cnt in zip(classes, counts):
            if cnt < max_count:
                deficit = max_count - cnt
                idx     = np.where(y == cls)[0]
                extra_i = np.random.choice(idx, deficit, replace=True)
                X_parts.append(X[extra_i])
                y_parts.append(y[extra_i])
        return np.vstack(X_parts), np.concatenate(y_parts)

    def encode_fit(self, df: pd.DataFrame, label_col: str,
                   oversample: bool = True):
        df    = self.clean(df)
        y_raw = df[label_col].astype(str)
        le    = LabelEncoder()
        y     = le.fit_transform(y_raw)
        df.drop(columns=[label_col], inplace=True)

        encoders = {}
        for col in df.select_dtypes(include=["object","category"]).columns:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col].astype(str))
            encoders[col] = enc

        df.fillna(df.median(numeric_only=True), inplace=True)
        df.fillna(0, inplace=True)
        X      = df.values.astype(np.float32)
        scaler = StandardScaler()
        X      = scaler.fit_transform(X)

        classes, counts = np.unique(y, return_counts=True)
        imbalance_ratio = counts.max() / (counts.min() + 1e-9)
        if oversample and imbalance_ratio > 1.5 and len(classes) > 1:
            self.log.info(f"Class imbalance (ratio={imbalance_ratio:.1f}) — oversampling")
            X, y = self._oversample(X, y)
            self.log.info(f"After oversampling: {len(X):,} samples")

        encoders["__label__"] = le
        feature_names = list(df.columns)
        return X, y, scaler, encoders, le, feature_names

    def encode_transform(self, df: pd.DataFrame, label_col: str,
                         scaler, encoders: dict):
        df    = self.clean(df)
        y_raw = df[label_col].astype(str)
        le    = encoders["__label__"]
        known = set(le.classes_)
        y_raw = y_raw.apply(lambda v: v if v in known else le.classes_[0])
        y     = le.transform(y_raw)
        df.drop(columns=[label_col], inplace=True)

        for col in df.select_dtypes(include=["object","category"]).columns:
            enc = encoders.get(col)
            if enc:
                known_c = set(enc.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda v: v if v in known_c else enc.classes_[0])
                df[col] = enc.transform(df[col])
            else:
                df[col] = 0

        df.fillna(df.median(numeric_only=True), inplace=True)
        df.fillna(0, inplace=True)
        X = df.values.astype(np.float32)
        try:
            X = scaler.transform(X)
        except Exception:
            pass
        return X, y

    def load(self, dataset: str, layer: str):
        base           = DATASET_DIRS[dataset]
        train_f, test_f= LAYER_FILES[dataset][layer]
        train_path     = base / train_f
        test_path      = base / test_f

        if not train_path.exists():
            self.log.warn(f"Not found: {train_path}")
            return (None,) * 8

        try:
            tr = pd.read_csv(train_path, low_memory=False)
        except Exception as e:
            self.log.err(f"Cannot read {train_path}: {e}")
            return (None,) * 8

        if tr.empty:
            self.log.warn(f"Empty file: {train_path}")
            return (None,) * 8

        lc    = self.find_label(tr)
        n_cls = tr[lc].nunique()
        self.log.info(f"File: {train_path.name}  │  Rows: {len(tr):,}  │  "
                      f"Features: {tr.shape[1]-1}  │  Classes: {n_cls}  │  "
                      f"Label: '{lc}'")

        X_tr, y_tr, sc, enc, le, feat_names = self.encode_fit(tr, lc)

        if test_path.exists():
            try:
                te = pd.read_csv(test_path, low_memory=False)
                if lc in te.columns:
                    X_te, y_te = self.encode_transform(te, lc, sc, enc)
                else:
                    raise ValueError("label not in test file")
            except Exception as ex:
                self.log.warn(f"Test load issue ({ex}) — using 80/20 split")
                s = int(len(X_tr) * 0.8)
                X_te, y_te = X_tr[s:], y_tr[s:]
                X_tr, y_tr = X_tr[:s], y_tr[:s]
        else:
            self.log.warn("No test file — 80/20 split")
            s = int(len(X_tr) * 0.8)
            X_te, y_te = X_tr[s:], y_tr[s:]
            X_tr, y_tr = X_tr[:s], y_tr[:s]

        self.log.info(f"Train: {len(X_tr):,}  │  Test: {len(X_te):,}")
        return X_tr, y_tr, X_te, y_te, sc, enc, le, feat_names


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class MetricsEngine:
    @staticmethod
    def compute(y_true, y_pred, y_proba, le=None) -> dict:
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_true, y_pred)
        tn   = int(cm[0,0]) if cm.shape[0]>1 else 0
        fp   = int(cm[0,1]) if cm.shape[1]>1 else 0
        fn   = int(cm[1,0]) if cm.shape[0]>1 else 0
        tp   = int(cm[1,1]) if cm.shape[0]>1 and cm.shape[1]>1 else 0
        tpr  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        fpr  = fp/(fp+tn) if (fp+tn)>0 else 0.0
        try:
            if y_proba is not None and y_proba.shape[1]>1:
                auc_roc = roc_auc_score(y_true, y_proba,
                                        multi_class="ovr", average="weighted")
                avg_prec= average_precision_score(
                    y_true,
                    y_proba[:,1] if y_proba.shape[1]==2 else y_proba.max(1),
                    average="weighted")
            else:
                auc_roc = avg_prec = None
        except Exception:
            auc_roc = avg_prec = None
        return dict(accuracy=acc, precision=prec, recall=rec, f1_score=f1,
                    tpr=tpr, fpr=fpr, auc_roc=auc_roc, avg_precision=avg_prec,
                    confusion_matrix=cm.tolist())

    @staticmethod
    def print_table(m: dict, layer: str, dataset: str):
        c = LAYER_META.get(layer,{}).get("color", GREY)
        print(f"\n  {c}{BOLD}{'─'*52}")
        print(f"  Metrics — {layer.upper()} / {dataset.upper()}")
        print(f"  {'─'*52}{RESET}")
        for k in ["accuracy","precision","recall","f1_score","tpr","fpr"]:
            bar_len = int(m[k]*20)
            bar = "█"*bar_len + "░"*(20-bar_len)
            print(f"  {k:<14} {m[k]:>7.4f}  {c}{bar}{RESET}")
        if m.get("auc_roc"):
            print(f"  {'auc_roc':<14} {m['auc_roc']:>7.4f}")
        print()

    @staticmethod
    def kfold_cv(model_cls, params: dict, X, y, k=5, log=None) -> dict:
        skf   = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        accs, precs, recs, f1s, tprs, fprs = [],[],[],[],[],[]
        for fold, (tr_i, val_i) in enumerate(skf.split(X, y)):
            mdl = model_cls(**params)
            mdl.fit(X[tr_i], y[tr_i])
            yp  = mdl.predict(X[val_i])
            m   = MetricsEngine.compute(y[val_i], yp, None)
            accs.append(m["accuracy"]); precs.append(m["precision"])
            recs.append(m["recall"]);   f1s.append(m["f1_score"])
            tprs.append(m["tpr"]);      fprs.append(m["fpr"])
            if log:
                log.info(f"Fold {fold+1}/{k}: acc={m['accuracy']:.4f} "
                         f"f1={m['f1_score']:.4f}")
        def _s(arr): return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        return dict(accuracy=_s(accs), precision=_s(precs), recall=_s(recs),
                    f1_score=_s(f1s), tpr=_s(tprs), fpr=_s(fprs))


# ══════════════════════════════════════════════════════════════════════════════
#  STORE  (save / load / infer helpers — unchanged API)
# ══════════════════════════════════════════════════════════════════════════════
class Store:
    @staticmethod
    def save(layer, dataset, model, sc, enc, metrics, feat_names, cv_res=None):
        mdir = MODEL_ROOT / layer; mdir.mkdir(parents=True, exist_ok=True)
        rdir = RESULT_ROOT/ layer; rdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model,  mdir/f"{dataset}_model.joblib")
        joblib.dump(sc,     mdir/f"{dataset}_scaler.joblib")
        joblib.dump(enc,    mdir/f"{dataset}_encoders.joblib")
        joblib.dump(feat_names, mdir/f"{dataset}_feat_names.joblib")
        payload = {"metrics": metrics, "cv_results": cv_res or {},
                   "feat_names": feat_names,
                   "timestamp": datetime.now().isoformat()}
        (rdir/f"{dataset}_metrics.json").write_text(
            json.dumps(payload, indent=2, default=str))

    @staticmethod
    def load(layer, dataset):
        mdir = MODEL_ROOT / layer
        model     = joblib.load(mdir/f"{dataset}_model.joblib")
        sc        = joblib.load(mdir/f"{dataset}_scaler.joblib")
        enc       = joblib.load(mdir/f"{dataset}_encoders.joblib")
        le        = enc.get("__label__")
        feat_names= joblib.load(mdir/f"{dataset}_feat_names.joblib")
        return model, sc, enc, le, feat_names

    @staticmethod
    def align(x: np.ndarray, n_features: int) -> np.ndarray:
        if x.shape[1] < n_features:
            pad = np.zeros((x.shape[0], n_features - x.shape[1]), dtype=x.dtype)
            x   = np.hstack([x, pad])
        return x[:, :n_features]

    @staticmethod
    def infer(model, sc, le, raw: np.ndarray):
        x = raw.reshape(1,-1).astype(np.float32)
        x = Store.align(x, model.n_features_in_)
        try:   x = sc.transform(x)
        except Exception: pass
        pred  = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
        conf  = float(proba.max())
        label = str(le.inverse_transform([pred])[0]) if le else str(pred)
        return label, conf, proba


# ══════════════════════════════════════════════════════════════════════════════
#  DRIFT DETECTOR  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class DriftDetector:
    def __init__(self, bins: int = 10):
        self.bins      = bins
        self.baseline  = None
        self.history   = []

    def fit(self, proba: np.ndarray):
        p = proba.mean(axis=0) if proba.ndim > 1 else proba
        self.baseline = np.clip(p, 1e-9, 1)

    def check(self, proba: np.ndarray, layer: str, dataset: str, log) -> dict:
        if self.baseline is None:
            return {}
        p   = proba.mean(axis=0) if proba.ndim > 1 else proba
        p   = np.clip(p, 1e-9, 1)
        n   = min(len(p), len(self.baseline))
        psi = float(np.sum((p[:n]-self.baseline[:n]) *
                            np.log(p[:n]/(self.baseline[:n]+1e-9))))
        entry = {"psi": psi, "ts": datetime.now().isoformat(),
                 "layer": layer, "dataset": dataset}
        self.history.append(entry)
        if psi > 0.20:
            log.warn(f"DRIFT ALERT PSI={psi:.4f} — consider retraining {layer}/{dataset}")
        elif psi > 0.10:
            log.warn(f"Drift monitor PSI={psi:.4f}")
        return entry


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT ENGINE  (unchanged — all existing plots preserved)
# ══════════════════════════════════════════════════════════════════════════════
class PlotEngine:
    @staticmethod
    def _dir(layer: str) -> Path:
        p = PLOT_ROOT / layer; p.mkdir(parents=True, exist_ok=True); return p

    @staticmethod
    def roc(y_te, y_proba, le, layer, dataset):
        pdir = PlotEngine._dir(layer)
        n_cls= len(np.unique(y_te))
        fig, ax = plt.subplots(figsize=(8,6))
        colours = plt.cm.tab10(np.linspace(0, 0.9, min(n_cls,10)))
        if y_proba is not None and n_cls >= 2:
            for i in range(min(n_cls, y_proba.shape[1], 10)):
                try:
                    mask = (y_te == i)
                    if mask.sum() == 0: continue
                    fpr_c, tpr_c, _ = roc_curve(mask.astype(int), y_proba[:,i])
                    a = auc(fpr_c, tpr_c)
                    lbl_name = (le.inverse_transform([i])[0]
                                if le is not None else str(i))
                    ax.plot(fpr_c, tpr_c, lw=1.5,
                            label=f"{lbl_name} (AUC={a:.3f})",
                            color=colours[i % len(colours)])
                except Exception:
                    continue
        ax.plot([0,1],[0,1],"k--",lw=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curves — {layer.upper()} / {dataset.upper()}", fontsize=12)
        ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
        out = pdir/f"{dataset}_roc.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    @staticmethod
    def pr_curve(y_te, y_proba, le, layer, dataset):
        pdir = PlotEngine._dir(layer)
        n_cls= len(np.unique(y_te))
        fig, ax = plt.subplots(figsize=(8,6))
        colours = plt.cm.tab10(np.linspace(0,0.9, min(n_cls,10)))
        if y_proba is not None and n_cls >= 2:
            for i in range(min(n_cls, y_proba.shape[1], 10)):
                try:
                    mask = (y_te == i)
                    if mask.sum()==0: continue
                    p_, r_, _ = precision_recall_curve(mask.astype(int), y_proba[:,i])
                    ap = average_precision_score(mask.astype(int), y_proba[:,i])
                    lbl_name = (le.inverse_transform([i])[0]
                                if le is not None else str(i))
                    ax.plot(r_, p_, lw=1.5,
                            label=f"{lbl_name} (AP={ap:.3f})",
                            color=colours[i % len(colours)])
                except Exception:
                    continue
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall — {layer.upper()} / {dataset.upper()}", fontsize=12)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        out = pdir/f"{dataset}_pr_curve.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    @staticmethod
    def confusion(y_te, y_pred, le, layer, dataset):
        pdir = PlotEngine._dir(layer)
        cm   = confusion_matrix(y_te, y_pred)
        labels = (le.classes_.tolist() if le is not None
                  else [str(i) for i in range(cm.shape[0])])
        fig_h = max(5, cm.shape[0]*0.55+2)
        fig, ax = plt.subplots(figsize=(fig_h+2, fig_h))
        sns.heatmap(cm, annot=(cm.shape[0]<=20), fmt="d",
                    cmap="Blues", ax=ax,
                    xticklabels=labels[:cm.shape[1]],
                    yticklabels=labels[:cm.shape[0]],
                    linewidths=0.3 if cm.shape[0]<=20 else 0)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {layer.upper()} / {dataset.upper()}", fontsize=12)
        plt.xticks(fontsize=8, rotation=45, ha="right")
        plt.yticks(fontsize=8, rotation=0)
        out = pdir/f"{dataset}_confusion.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    @staticmethod
    def feature_importance(model, feat_names, layer, dataset, top_n=20):
        pdir = PlotEngine._dir(layer)
        if not hasattr(model, "feature_importances_"):
            return None
        fi = model.feature_importances_
        n  = min(top_n, len(fi))
        idx= np.argsort(fi)[::-1][:n]
        names = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in idx]
        vals  = fi[idx]
        fig, ax = plt.subplots(figsize=(9, 0.4*n+2))
        colours = plt.cm.RdYlGn(np.linspace(0.3,0.9,n))
        ax.barh(range(n), vals[::-1], color=colours)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel("Feature Importance (Gini)", fontsize=10)
        ax.set_title(f"Top {n} Features — {layer.upper()} / {dataset.upper()}", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        out = pdir/f"{dataset}_feature_importance.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return str(out)

    @staticmethod
    def learning_curve(model_cls, params, X, y, layer, dataset, n_points=6):
        pdir  = PlotEngine._dir(layer)
        sizes = np.linspace(0.10,1.0,n_points)
        tr_scores, val_scores = [], []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for frac in sizes:
            n = max(int(len(X)*frac), 20)
            Xi, yi = X[:n], y[:n]
            fold_tr, fold_val = [], []
            for tr_i, val_i in skf.split(Xi, yi):
                mdl = model_cls(**params)
                mdl.fit(Xi[tr_i], yi[tr_i])
                fold_tr.append(accuracy_score(yi[tr_i], mdl.predict(Xi[tr_i])))
                fold_val.append(accuracy_score(yi[val_i], mdl.predict(Xi[val_i])))
            tr_scores.append(np.mean(fold_tr))
            val_scores.append(np.mean(fold_val))
        fig, ax = plt.subplots(figsize=(7,5))
        ns = [int(len(X)*s) for s in sizes]
        ax.plot(ns, tr_scores,  "o-", label="Training",   color="#2196F3")
        ax.plot(ns, val_scores, "s-", label="Validation", color="#F44336")
        ax.fill_between(ns, tr_scores, val_scores, alpha=0.1, color="purple")
        ax.set_xlabel("Training Samples", fontsize=11)
        ax.set_ylabel("Accuracy",         fontsize=11)
        ax.set_title(f"Learning Curve — {layer.upper()} / {dataset.upper()}", fontsize=12)
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim([0,1.05])
        out = pdir/f"{dataset}_learning_curve.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return str(out)

    @staticmethod
    def cv_results(cv, layer, dataset):
        pdir   = PlotEngine._dir(layer)
        keys   = ["accuracy","precision","recall","f1_score","tpr","fpr"]
        means  = [cv[k]["mean"] for k in keys]
        stds   = [cv[k]["std"]  for k in keys]
        labels = ["Accuracy","Precision","Recall","F1","TPR","FPR"]
        colours= ["#4CAF50","#2196F3","#FF9800","#9C27B0","#00BCD4","#F44336"]
        fig, ax = plt.subplots(figsize=(9,5))
        bars = ax.bar(labels, means, yerr=stds, capsize=5,
                      color=colours, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{val:.3f}", ha="center", fontsize=9)
        ax.set_ylim([0,1.15]); ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"5-Fold CV Results — {layer.upper()} / {dataset.upper()}", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        out = pdir/f"{dataset}_cv_results.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return str(out)

    @staticmethod
    def drift_history(history, layer, dataset):
        if not history: return None
        pdir = PlotEngine._dir(layer)
        psi_v = [h["psi"] for h in history]
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(psi_v, "o-", color="#9C27B0", lw=2)
        ax.axhline(0.10, color="orange", ls="--", lw=1, label="Monitor (0.10)")
        ax.axhline(0.20, color="red",    ls="--", lw=1, label="Retrain (0.20)")
        ax.fill_between(range(len(psi_v)), psi_v, alpha=0.15, color="#9C27B0")
        ax.set_xlabel("Check #"); ax.set_ylabel("PSI")
        ax.set_title(f"Model Drift (PSI) — {layer.upper()} / {dataset.upper()}", fontsize=12)
        ax.legend(); ax.grid(alpha=0.3)
        out = pdir/f"{dataset}_drift.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return str(out)

    @staticmethod
    def master_heatmap(all_metrics):
        PLOT_ROOT.mkdir(parents=True, exist_ok=True)
        rows, idx = [], []
        for ds, layers in all_metrics.items():
            for lyr, payload in layers.items():
                m = payload.get("metrics", payload)
                rows.append([m.get("accuracy",0), m.get("precision",0),
                              m.get("recall",0),   m.get("f1_score",0),
                              m.get("tpr",0),       m.get("fpr",0),
                              m.get("auc_roc") or 0])
                idx.append(f"{ds[:3].upper()}/{lyr[:4].upper()}")
        df  = pd.DataFrame(rows, index=idx,
                           columns=["Acc","Prec","Rec","F1","TPR","FPR","AUC"])
        fig, ax = plt.subplots(figsize=(10, max(4,len(df)*0.55)+1))
        sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGn",
                    linewidths=0.5, ax=ax, vmin=0, vmax=1,
                    cbar_kws={"label":"Score"})
        ax.set_title("AI-SOC Agentic — All Layers × All Datasets", fontsize=13)
        plt.xticks(fontsize=10); plt.yticks(fontsize=9, rotation=0)
        out = PLOT_ROOT/"MASTER_METRICS_HEATMAP.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return str(out)

    @staticmethod
    def radar_comparison(all_metrics, dataset):
        PLOT_ROOT.mkdir(parents=True, exist_ok=True)
        metrics_keys = ["accuracy","precision","recall","f1_score","tpr"]
        labels       = ["Accuracy","Precision","Recall","F1","TPR"]
        angles = np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
        colours = ["#2196F3","#4CAF50","#FF9800","#F44336","#9C27B0"]
        ds_data = all_metrics.get(dataset, {})
        for i, (lyr, payload) in enumerate(ds_data.items()):
            m    = payload.get("metrics", payload)
            vals = [m.get(k,0) for k in metrics_keys] + [m.get(metrics_keys[0],0)]
            ax.plot(angles, vals, "o-", lw=2, label=lyr.upper(),
                    color=colours[i%len(colours)])
            ax.fill(angles, vals, alpha=0.08, color=colours[i%len(colours)])
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
        ax.set_ylim(0,1)
        ax.set_title(f"Layer Comparison Radar — {dataset.upper()}", fontsize=12, pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), fontsize=9)
        ax.grid(alpha=0.3)
        out = PLOT_ROOT/f"{dataset}_radar.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        return str(out)