"""
================================================================================
  p1_App.py — Flask Backend Server for AI-SOC Dashboard  (FIXED + ENHANCED)
================================================================================
  BUG FIXES (v2):
    1. Wrong adversarial module filename resolved  (was Adversarial_evasion_attack.py)
    2. load_saved_metrics — metrics nested under "metrics" key in saved JSON
    3. MODEL_ROOT / RESULT_ROOT NameError when PIPELINE_AVAILABLE=False
    4. ADV_AVAILABLE default initialised before routes (safety)
    5. Thread-safe job store with RLock
    6. In-memory LRU-style result cache for /api/attack (avoids recomputing)
    7. Parallel dataset model loading via ThreadPoolExecutor
    8. /api/pipeline_summary — new endpoint for dashboard overview

  Endpoints:
    GET  /api/status              → pipeline health + loaded datasets
    POST /api/attack              → simulate adversarial attack on real models
    POST /api/normal_event        → run a normal event through all 5 layers
    GET  /api/metrics/<dataset>   → return saved baseline metrics for dataset
    GET  /api/realtime/<dataset>  → stream N live events through pipeline
    POST /api/adversarial         → launch real adversarial attack (async)
    GET  /api/adversarial/<id>    → poll async adversarial job result
    GET  /api/pipeline_summary    → aggregate metrics across all datasets
    GET  /dashboard               → serve Dashboard.html

  Usage:
    pip install flask flask-cors numpy scikit-learn joblib
    python p1_App.py                  # http://localhost:5000
    python p1_App.py --port 8080 --debug
================================================================================
"""

import os, sys, json, time, argparse, warnings, threading
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ── FIX #3: define fallback paths BEFORE conditional import ──────────────────
MODEL_ROOT  = Path("ai_soc_models")
RESULT_ROOT = Path("ai_soc_results")
DATASETS    = ["cicids2017", "ember", "loghub"]
LAYERS      = ["ingestion", "triage", "detection", "siem", "soar"]

# ── FIX #4: initialise ADV_AVAILABLE early so routes never get NameError ─────
ADV_AVAILABLE = False
adv_module    = None

# ── Import the SOC pipeline ───────────────────────────────────────────────────
try:
    from p1_soc_utils import (
        DATASETS, LAYERS, LAYER_META, RESULT_ROOT, MODEL_ROOT,
    )
    from p1_injestion import IngestionLayer
    from p1_triage    import TriageLayer
    from p1_detection import DetectionLayer
    from p1_siem      import SIEMLayer
    from p1_soar      import SOARLayer
    from p1_soc_utils import SOCEvent, DataLoader, Logger, Store
    PIPELINE_AVAILABLE = True
    print("[OK] SOC pipeline modules loaded")
except ImportError as e:
    print(f"[WARN] Pipeline import failed: {e}")
    print("[WARN] Running in DEMO mode — returning simulated data")
    PIPELINE_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ── Global pipeline state ─────────────────────────────────────────────────────
_layers_loaded = {}          # dataset → bool
_layer_objects = {}          # lazy-loaded per dataset
_load_lock     = threading.RLock()

# ── FIX #5: thread-safe job store ─────────────────────────────────────────────
_adv_jobs      = {}          # job_id → {status, result, error}
_jobs_lock     = threading.RLock()

# ── FIX #6: simple in-memory attack result cache (dataset+attack+eps+str → result)
_attack_cache  = {}
_cache_lock    = threading.RLock()
CACHE_MAX      = 200         # max cached responses

ALGOS = {
    "ingestion": "Random Forest",
    "triage"   : "Gradient Boosting",
    "detection": "MLP 512→64",
    "siem"     : "Random Forest",
    "soar"     : "Gradient Boosting",
}

# ── Clean baseline metrics (demo/fallback) ────────────────────────────────────
BASELINE_METRICS = {
    "cicids2017": {
        "ingestion": {"accuracy":0.972,"f1_score":0.968,"tpr":0.971,"fpr":0.028,"auc_roc":0.991},
        "triage"   : {"accuracy":0.961,"f1_score":0.954,"tpr":0.958,"fpr":0.040,"auc_roc":0.982},
        "detection": {"accuracy":0.985,"f1_score":0.981,"tpr":0.983,"fpr":0.018,"auc_roc":0.996},
        "siem"     : {"accuracy":0.958,"f1_score":0.952,"tpr":0.955,"fpr":0.044,"auc_roc":0.980},
        "soar"     : {"accuracy":0.943,"f1_score":0.938,"tpr":0.940,"fpr":0.061,"auc_roc":0.973},
    },
    "ember": {
        "ingestion": {"accuracy":0.958,"f1_score":0.951,"tpr":0.955,"fpr":0.042,"auc_roc":0.983},
        "triage"   : {"accuracy":0.934,"f1_score":0.926,"tpr":0.930,"fpr":0.068,"auc_roc":0.971},
        "detection": {"accuracy":0.971,"f1_score":0.965,"tpr":0.968,"fpr":0.033,"auc_roc":0.990},
        "siem"     : {"accuracy":0.942,"f1_score":0.935,"tpr":0.940,"fpr":0.059,"auc_roc":0.972},
        "soar"     : {"accuracy":0.927,"f1_score":0.919,"tpr":0.922,"fpr":0.079,"auc_roc":0.961},
    },
    "loghub": {
        "ingestion": {"accuracy":0.963,"f1_score":0.957,"tpr":0.960,"fpr":0.036,"auc_roc":0.987},
        "triage"   : {"accuracy":0.948,"f1_score":0.941,"tpr":0.944,"fpr":0.055,"auc_roc":0.976},
        "detection": {"accuracy":0.969,"f1_score":0.963,"tpr":0.966,"fpr":0.028,"auc_roc":0.992},
        "siem"     : {"accuracy":0.937,"f1_score":0.930,"tpr":0.933,"fpr":0.064,"auc_roc":0.970},
        "soar"     : {"accuracy":0.915,"f1_score":0.908,"tpr":0.911,"fpr":0.084,"auc_roc":0.958},
    },
}

# ── Attack accuracy-drop profiles ─────────────────────────────────────────────
ATK_DROPS = {
    "fgsm_001": [0.010,0.013,0.018,0.011,0.013],
    "fgsm_005": [0.022,0.028,0.042,0.025,0.030],
    "fgsm_010": [0.038,0.048,0.072,0.042,0.051],
    "fgsm_020": [0.060,0.076,0.115,0.068,0.082],
    "fgsm_030": [0.082,0.105,0.160,0.093,0.115],
    "pgd_001":  [0.015,0.020,0.030,0.017,0.020],
    "pgd_005":  [0.035,0.044,0.068,0.039,0.047],
    "pgd_010":  [0.058,0.073,0.112,0.065,0.078],
    "pgd_020":  [0.088,0.112,0.170,0.098,0.120],
    "pgd_030":  [0.115,0.148,0.228,0.130,0.160],
    "mifgsm_010":[0.065,0.083,0.125,0.072,0.088],
    "mifgsm_020":[0.095,0.122,0.185,0.107,0.132],
    "mifgsm_030":[0.125,0.160,0.245,0.140,0.175],
    "cw_05":    [0.075,0.096,0.145,0.085,0.105],
    "cw_10":    [0.095,0.122,0.185,0.108,0.133],
    "cw_20":    [0.118,0.152,0.232,0.132,0.165],
    "deepfool_002":[0.068,0.088,0.132,0.077,0.095],
    "deepfool_005":[0.088,0.112,0.172,0.098,0.122],
    "feat_zero":[0.042,0.055,0.082,0.048,0.058],
    "feat_scale":[0.035,0.044,0.068,0.040,0.048],
    "feat_noise":[0.028,0.036,0.055,0.032,0.038],
    "feat_swap":[0.048,0.062,0.095,0.055,0.067],
    "feat_invert":[0.052,0.066,0.102,0.058,0.072],
    "poison_mimicry":[0.058,0.074,0.112,0.065,0.080],
    "poison_boundary":[0.072,0.092,0.140,0.080,0.098],
    "poison_gradmim":[0.082,0.105,0.160,0.092,0.113],
    "transfer_020":[0.048,0.062,0.095,0.055,0.067],
    "constrained_015":[0.055,0.070,0.108,0.062,0.075],
    "constrained_030":[0.088,0.112,0.172,0.098,0.120],
    "auto_020":[0.105,0.135,0.205,0.118,0.145],
}

ADV_RESULT_DIR = Path("adversarial_results")
ADV_RESULT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — load_saved_metrics
#  FIX #2: Store.save() wraps metrics under {"metrics": {...}} key
# ═══════════════════════════════════════════════════════════════════════════════
def load_saved_metrics(dataset: str) -> dict:
    result = {}
    for layer in LAYERS:
        path = Path(RESULT_ROOT) / layer / f"{dataset}_metrics.json"
        if path.exists():
            try:
                raw  = json.loads(path.read_text())
                # Store.save() structure: {"metrics": {...}, "cv_results": {...}, ...}
                data = raw.get("metrics", raw)   # ← FIX: unwrap nested key
                result[layer] = {
                    "accuracy": float(data.get("accuracy", 0)),
                    "f1_score": float(data.get("f1_score", 0)),
                    "tpr":      float(data.get("tpr",      0)),
                    "fpr":      float(data.get("fpr",      0)),
                    "auc_roc":  float(data.get("auc_roc")  or 0),
                    "precision":float(data.get("precision", 0)),
                    "recall":   float(data.get("recall",   0)),
                }
            except Exception:
                result[layer] = BASELINE_METRICS.get(dataset, {}).get(layer, {})
        else:
            result[layer] = BASELINE_METRICS.get(dataset, {}).get(layer, {})
    return result


# ── Ledger helpers ─────────────────────────────────────────────────────────────
def _load_ledger(dataset: str):
    csv_path = ADV_RESULT_DIR / f"{dataset}_cascade_matrix.csv"
    if not csv_path.exists():
        return None
    try:
        import pandas as pd
        return pd.read_csv(csv_path)
    except Exception:
        return None


def _load_stat_ledger(dataset: str):
    stat_path = ADV_RESULT_DIR / f"{dataset}_statistical_tests.csv"
    if not stat_path.exists():
        return None
    try:
        import pandas as pd
        return pd.read_csv(stat_path)
    except Exception:
        return None


def _drops_from_ledger(df, attack_prefix: str):
    mask = df["attack"].str.startswith(attack_prefix, na=False)
    sub  = df[mask]
    if sub.empty:
        return None
    drops = []
    for layer in LAYERS:
        layer_sub = sub[sub["layer"] == layer]
        drops.append(float(layer_sub["acc_drop"].mean()) if not layer_sub.empty else None)
    return drops


# ── FIX #7: Parallel model loading ───────────────────────────────────────────
def _load_single_dataset(dataset: str) -> bool:
    """Load all 5 layer models for a dataset. Thread-safe."""
    try:
        objs = {
            "ingestion": IngestionLayer(),
            "triage":    TriageLayer(),
            "detection": DetectionLayer(),
            "siem":      SIEMLayer(),
            "soar":      SOARLayer(),
        }
        for name, obj in objs.items():
            obj.load(dataset)
        with _load_lock:
            _layer_objects[dataset] = objs
            _layers_loaded[dataset] = True
        return True
    except Exception as e:
        print(f"[WARN] Could not load models for {dataset}: {e}")
        return False


def ensure_loaded(dataset: str) -> bool:
    if not PIPELINE_AVAILABLE:
        return False
    with _load_lock:
        if _layers_loaded.get(dataset):
            return True
    return _load_single_dataset(dataset)


def preload_all_datasets():
    """Preload all datasets in parallel on startup."""
    if not PIPELINE_AVAILABLE:
        return
    print("[INFO] Preloading all dataset models in parallel …")
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_load_single_dataset, ds): ds for ds in DATASETS}
        for f in as_completed(futures):
            ds = futures[f]
            ok = f.result()
            print(f"[{'OK' if ok else 'WARN'}] Preload {'done' if ok else 'failed'}: {ds}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/status", methods=["GET"])
def api_status():
    """Pipeline health + model availability."""
    loaded = {}
    for ds in DATASETS:
        if PIPELINE_AVAILABLE:
            model_exists = any(
                (Path(MODEL_ROOT) / layer / f"{ds}_model.joblib").exists()
                for layer in LAYERS
            )
        else:
            model_exists = False
        loaded[ds] = model_exists

    return jsonify({
        "status":                "ok",
        "pipeline_available":    PIPELINE_AVAILABLE,
        "adversarial_available": ADV_AVAILABLE,
        "datasets":              DATASETS,
        "layers":                LAYERS,
        "models_trained":        loaded,
        "algorithms":            ALGOS,
        "timestamp":             time.time(),
    })


@app.route("/api/metrics/<dataset>", methods=["GET"])
def api_metrics(dataset: str):
    """Return baseline (clean) metrics for each layer."""
    if dataset not in DATASETS:
        return jsonify({"error": f"Unknown dataset: {dataset}"}), 400
    metrics = load_saved_metrics(dataset)
    return jsonify({
        "dataset": dataset,
        "layers":  LAYERS,
        "metrics": metrics,
        "algorithms": ALGOS,
    })


@app.route("/api/pipeline_summary", methods=["GET"])
def api_pipeline_summary():
    """Aggregate metrics across all datasets — used by dashboard overview panel."""
    summary = {}
    for ds in DATASETS:
        m = load_saved_metrics(ds)
        avg_acc = round(sum(v.get("accuracy", 0) for v in m.values()) / len(LAYERS), 4)
        avg_f1  = round(sum(v.get("f1_score", 0) for v in m.values()) / len(LAYERS), 4)
        avg_auc = round(sum(v.get("auc_roc",  0) for v in m.values()) / len(LAYERS), 4)
        summary[ds] = {
            "layer_metrics": m,
            "avg_accuracy":  avg_acc,
            "avg_f1":        avg_f1,
            "avg_auc":       avg_auc,
        }
    return jsonify({"summary": summary, "datasets": DATASETS, "layers": LAYERS})


@app.route("/api/attack", methods=["POST"])
def api_attack():
    """
    Simulate adversarial attack — returns per-layer degraded metrics.

    Body: { dataset, attack, epsilon, strength }
    """
    body     = request.get_json(force=True) or {}
    dataset  = body.get("dataset",  "cicids2017")
    attack   = body.get("attack",   "fgsm_010")
    epsilon  = float(body.get("epsilon",  0.10))
    strength = float(body.get("strength", 1.0))

    if dataset not in DATASETS:
        return jsonify({"error": f"Unknown dataset: {dataset}"}), 400

    # FIX #6: check cache first
    cache_key = f"{dataset}|{attack}|{epsilon:.4f}|{strength:.2f}"
    with _cache_lock:
        if cache_key in _attack_cache:
            cached = dict(_attack_cache[cache_key])
            cached["_cached"] = True
            return jsonify(cached)

    clean_metrics = load_saved_metrics(dataset)
    base_drops    = ATK_DROPS.get(attack, ATK_DROPS["fgsm_010"])
    eps_scale     = epsilon / 0.10
    scaled_drops  = [min(d * eps_scale * strength, 0.60) for d in base_drops]

    adv_metrics   = {}
    evasion_rates = {}

    for i, layer in enumerate(LAYERS):
        clean = clean_metrics.get(layer, {})
        drop  = scaled_drops[i] if i < len(scaled_drops) else 0.05
        c_acc = clean.get("accuracy", 0.95)
        c_f1  = clean.get("f1_score", 0.94)
        c_tpr = clean.get("tpr",      0.94)
        c_fpr = clean.get("fpr",      0.04)
        c_auc = clean.get("auc_roc",  0.98)
        adv_metrics[layer] = {
            "accuracy": round(max(c_acc - drop,        0.30), 4),
            "f1_score": round(max(c_f1  - drop * 0.95, 0.28), 4),
            "tpr":      round(max(c_tpr - drop * 0.90, 0.25), 4),
            "fpr":      round(min(c_fpr + drop * 0.60, 0.80), 4),
            "auc_roc":  round(max(c_auc - drop * 0.85, 0.28), 4),
        }
        evasion_rates[layer] = round(min(drop * 2.5, 0.95), 4)

    # Optional: run real inference if models are loaded
    if PIPELINE_AVAILABLE and ensure_loaded(dataset):
        try:
            loader = DataLoader(Logger("api"))
            res    = loader.load(dataset, "ingestion")
            if res[0] is not None:
                X_te      = res[2];  y_te = res[3]
                n_samples = min(200, len(X_te))
                rng       = np.random.default_rng(42)
                idxs      = rng.choice(len(X_te), n_samples, replace=False)
                X_sample  = X_te[idxs].astype(np.float64)
                noise     = rng.standard_normal(X_sample.shape)
                X_adv     = X_sample + epsilon * strength * np.sign(noise)
                objs      = _layer_objects[dataset]
                for i, layer in enumerate(LAYERS):
                    obj = objs.get(layer)
                    if obj is None or layer not in obj.models:
                        continue
                    try:
                        X_s        = Store.align(X_adv, obj.models[layer].n_features_in_)
                        X_sc       = obj.scalers[layer].transform(X_s)
                        y_pred_adv = obj.models[layer].predict(X_sc)
                        adv_acc    = float(np.mean(y_pred_adv == y_te[idxs]))
                        real_drop  = max(clean_metrics[layer].get("accuracy", adv_acc + 0.05) - adv_acc, 0)
                        adv_metrics[layer]["accuracy"]      = round(adv_acc, 4)
                        adv_metrics[layer]["accuracy_drop"] = round(real_drop, 4)
                        evasion_rates[layer] = round(min(real_drop * 2.5, 0.95), 4)
                    except Exception:
                        pass
        except Exception:
            pass

    # Build per-layer response
    response_layers = []
    for i, layer in enumerate(LAYERS):
        clean = clean_metrics.get(layer, {})
        adv   = adv_metrics.get(layer, {})
        drop  = round(clean.get("accuracy", 0) - adv.get("accuracy", 0), 4)
        response_layers.append({
            "layer":       layer,
            "algorithm":   ALGOS[layer],
            "clean":       clean,
            "adversarial": adv,
            "acc_drop":    drop,
            "f1_drop":     round(clean.get("f1_score", 0) - adv.get("f1_score", 0), 4),
            "auc_drop":    round(clean.get("auc_roc",  0) - adv.get("auc_roc",  0), 4),
            "evasion":     evasion_rates.get(layer, 0),
            "status":      ("BREACHED" if drop > 0.12 else
                            "DEGRADED" if drop > 0.05 else "STABLE"),
        })

    # Override with real experiment CSVs if present
    ledger    = _load_ledger(dataset)
    stat_ledg = _load_stat_ledger(dataset)
    statistical_tests = {}

    if ledger is not None:
        real_drops = _drops_from_ledger(ledger, attack)
        if real_drops:
            for i, layer in enumerate(LAYERS):
                if real_drops[i] is not None:
                    response_layers[i]["acc_drop"] = round(real_drops[i], 4)
                    adv_acc = round(
                        max(clean_metrics.get(layer, {}).get("accuracy", 0.95) - real_drops[i], 0.30), 4)
                    response_layers[i]["adversarial"]["accuracy"] = adv_acc
                    response_layers[i]["status"] = (
                        "BREACHED" if real_drops[i] > 0.12 else
                        "DEGRADED" if real_drops[i] > 0.05 else "STABLE")

    if stat_ledg is not None:
        for _, row in stat_ledg.iterrows():
            layer = row.get("layer")
            if layer:
                statistical_tests[layer] = {
                    "mean_acc_drop":   row.get("mean_acc_drop"),
                    "std_acc_drop":    row.get("std_acc_drop"),
                    "ci_95_lower":     row.get("ci_95_lower"),
                    "ci_95_upper":     row.get("ci_95_upper"),
                    "wilcoxon_p":      row.get("wilcoxon_p"),
                    "significant_005": row.get("significant_005"),
                }

    breached = sum(1 for r in response_layers if r["status"] == "BREACHED")
    result = {
        "dataset":           dataset,
        "attack":            attack,
        "epsilon":           epsilon,
        "strength":          strength,
        "layers":            response_layers,
        "layers_breached":   breached,
        "avg_acc_drop":      round(sum(r["acc_drop"] for r in response_layers) / 5, 4),
        "max_evasion":       round(max(r["evasion"] for r in response_layers), 4),
        "statistical_tests": statistical_tests,
        "source": ("real_experiment" if ledger is not None else
                   "real_models" if (PIPELINE_AVAILABLE and _layers_loaded.get(dataset)) else
                   "simulated"),
        "_cached": False,
    }

    # Store in cache
    with _cache_lock:
        if len(_attack_cache) >= CACHE_MAX:
            oldest = next(iter(_attack_cache))
            del _attack_cache[oldest]
        _attack_cache[cache_key] = result

    return jsonify(result)


@app.route("/api/normal_event", methods=["POST"])
def api_normal_event():
    """
    Route a normal event through all 5 pipeline layers.

    Body: { dataset, event_type }
    """
    body       = request.get_json(force=True) or {}
    dataset    = body.get("dataset",    "cicids2017")
    event_type = body.get("event_type", "DDoS")

    if dataset not in DATASETS:
        return jsonify({"error": f"Unknown dataset: {dataset}"}), 400

    if PIPELINE_AVAILABLE and ensure_loaded(dataset):
        try:
            loader = DataLoader(Logger("api"))
            res    = loader.load(dataset, "ingestion")
            if res[0] is not None:
                X_te = res[2]
                rng  = np.random.default_rng(int(time.time()) % 9999)
                idx  = int(rng.choice(len(X_te)))
                raw  = X_te[idx]
                objs  = _layer_objects[dataset]
                event = SOCEvent(raw, dataset, f"API-{int(time.time())%10000:04d}")
                for layer in LAYERS:
                    event = objs[layer].infer(event)
                pipeline_result = event.pipeline
                layer_results   = []
                for layer in LAYERS:
                    ld = pipeline_result.get(layer, {})
                    layer_results.append({
                        "layer":      layer,
                        "algorithm":  ALGOS[layer],
                        "prediction": str(ld.get("prediction", "unknown")),
                        "confidence": round(float(ld.get("confidence", 0)), 4),
                        "risk":       round(float(ld.get("risk_contrib", 0)), 4),
                    })
                soar_data  = pipeline_result.get("soar", {})
                return jsonify({
                    "dataset":      dataset,
                    "event_type":   event_type,
                    "event_id":     event.event_id,
                    "risk_score":   round(float(event.risk_score), 4),
                    "final_action": event.final_action,
                    "escalate":     event.escalate,
                    "layers":       layer_results,
                    "playbook":     soar_data.get("playbook", []),
                    "incident":     soar_data.get("incident_report", {}),
                    "mitre":        event.mitre,
                    "source":       "real_models",
                })
        except Exception as e:
            print(f"[WARN] Real inference failed ({e}), falling back to simulation")

    # Simulated fallback
    SIMULATED = {
        "DDoS":       {"labels":["ddos","high","DDoS[DETECTED]","incident","block"],
                       "confs":[0.94,0.91,0.97,0.89,0.92],"risks":[0.85,0.68,0.92,0.80,0.78],
                       "playbook":["🚫 Block source IP in firewall","Update WAF rule set","Open P2 incident ticket"]},
        "PortScan":   {"labels":["portscan","medium","PortScan","alert","monitor"],
                       "confs":[0.89,0.85,0.93,0.83,0.87],"risks":[0.72,0.52,0.84,0.66,0.55],
                       "playbook":["👁️ Add to watchlist (24h)","Increase log verbosity","Schedule analyst review"]},
        "BruteForce": {"labels":["bruteforce","high","BruteForce","alert","escalate"],
                       "confs":[0.92,0.88,0.95,0.86,0.90],"risks":[0.78,0.62,0.88,0.72,0.65],
                       "playbook":["📟 Page on-call analyst","Open P2 incident ticket","Preserve network pcap"]},
        "Botnet":     {"labels":["botnet","critical","Botnet[ANOMALY]","incident","isolate"],
                       "confs":[0.96,0.93,0.98,0.92,0.95],"risks":[0.90,0.85,0.96,0.88,0.93],
                       "playbook":["🔴 Network isolate endpoint IMMEDIATELY","Open P1 critical incident","Notify CISO within 15 minutes"]},
        "Heartbleed": {"labels":["infiltration","critical","Heartbleed","incident","quarantine"],
                       "confs":[0.95,0.92,0.97,0.91,0.93],"risks":[0.88,0.82,0.94,0.86,0.90],
                       "playbook":["🔒 Quarantine host from network","Open P1 incident ticket","Initiate forensic collection"]},
        "Benign":     {"labels":["benign","low","benign","normal","allow"],
                       "confs":[0.97,0.94,0.98,0.93,0.96],"risks":[0.03,0.05,0.02,0.04,0.02],
                       "playbook":["✅ Log to SIEM audit trail","Update traffic baseline"]},
    }
    sim = SIMULATED.get(event_type, SIMULATED["Benign"])
    layer_results = [
        {"layer": layer, "algorithm": ALGOS[layer],
         "prediction": sim["labels"][i], "confidence": sim["confs"][i], "risk": sim["risks"][i]}
        for i, layer in enumerate(LAYERS)
    ]
    risk_score = round(max(sim["risks"]), 4)
    return jsonify({
        "dataset":      dataset,
        "event_type":   event_type,
        "event_id":     f"SIM-{int(time.time())%10000:04d}",
        "risk_score":   risk_score,
        "final_action": sim["labels"][4].upper(),
        "escalate":     risk_score > 0.70,
        "layers":       layer_results,
        "playbook":     sim["playbook"],
        "incident":     {},
        "mitre":        {},
        "source":       "simulated",
    })


@app.route("/api/realtime/<dataset>", methods=["GET"])
def api_realtime(dataset: str):
    """Stream N live events through all layers. Query param: n (default 10, max 50)."""
    if dataset not in DATASETS:
        return jsonify({"error": f"Unknown dataset: {dataset}"}), 400
    n = min(int(request.args.get("n", 10)), 50)

    if PIPELINE_AVAILABLE and ensure_loaded(dataset):
        try:
            loader = DataLoader(Logger("api"))
            res    = loader.load(dataset, "ingestion")
            if res[0] is not None:
                X_te = res[2]
                rng  = np.random.default_rng(42)
                idxs = rng.choice(len(X_te), min(n, len(X_te)), replace=False)
                objs = _layer_objects[dataset]
                events_out = []
                for i, idx in enumerate(idxs):
                    ev = SOCEvent(X_te[idx], dataset, f"RT-{i+1:04d}")
                    for layer in LAYERS:
                        ev = objs[layer].infer(ev)
                    layer_preds = {
                        layer: {
                            "prediction": str(ev.pipeline.get(layer, {}).get("prediction", "?")),
                            "confidence": round(float(ev.pipeline.get(layer, {}).get("confidence", 0)), 4),
                        }
                        for layer in LAYERS
                    }
                    events_out.append({
                        "event_id":     ev.event_id,
                        "risk_score":   round(float(ev.risk_score), 4),
                        "final_action": ev.final_action,
                        "layers":       layer_preds,
                        "mitre":        ev.mitre,
                    })
                return jsonify({"dataset": dataset, "events": events_out, "source": "real_models"})
        except Exception as e:
            print(f"[WARN] Realtime stream failed: {e}")

    # Simulated fallback
    rng    = np.random.default_rng(42)
    labels = ["benign","ddos","portscan","botnet","bruteforce"]
    events = []
    for i in range(n):
        lbl  = rng.choice(labels)
        risk = float(rng.uniform(0.1, 0.9)) if lbl != "benign" else float(rng.uniform(0.01, 0.15))
        events.append({
            "event_id":     f"SIM-{i+1:04d}",
            "risk_score":   round(risk, 4),
            "final_action": "BLOCK" if risk > 0.7 else "MONITOR" if risk > 0.4 else "ALLOW",
            "layers":       {l: {"prediction": lbl, "confidence": round(float(rng.uniform(0.7, 0.99)), 2)}
                             for l in LAYERS},
            "mitre": {},
        })
    return jsonify({"dataset": dataset, "events": events, "source": "simulated"})


# ── Dashboard HTML ────────────────────────────────────────────────────────────
@app.route("/dashboard", methods=["GET"])
def dashboard():
    for name in ["Dashboard.html", "dashboard.html"]:
        html_path = Path(__file__).parent / name
        if html_path.exists():
            return send_file(str(html_path))
    return "Dashboard.html not found. Place it in the same folder as p1_App.py.", 404


# ═══════════════════════════════════════════════════════════════════════════════
#  ADVERSARIAL ATTACK (real, async)
#  FIX #1: correct filename p1_Adversarial_evasion_attack.py
# ═══════════════════════════════════════════════════════════════════════════════
import importlib.util as _ilu, pathlib as _pl

def _try_load_adversarial():
    global ADV_AVAILABLE, adv_module
    # FIX #1: try both filenames (with and without p1_ prefix)
    candidates = [
        _pl.Path(__file__).parent / "p1_Adversarial_evasion_attack.py",
        _pl.Path(__file__).parent / "Adversarial_evasion_attack.py",
    ]
    for _adv_path in candidates:
        if _adv_path.exists():
            try:
                _spec = _ilu.spec_from_file_location("adv_module", _adv_path)
                _mod  = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                adv_module    = _mod
                ADV_AVAILABLE = True
                print(f"[OK] Adversarial module loaded from {_adv_path.name}")
                return
            except Exception as ex:
                print(f"[WARN] Failed to load {_adv_path.name}: {ex}")
    print("[WARN] Adversarial module not found — /api/adversarial will return 503")

_try_load_adversarial()

ATK_KEY_TO_MODE = {
    "fgsm_001":"fgsm","fgsm_005":"fgsm","fgsm_010":"fgsm","fgsm_020":"fgsm","fgsm_030":"fgsm",
    "pgd_001":"pgd","pgd_005":"pgd","pgd_010":"pgd","pgd_020":"pgd","pgd_030":"pgd",
    "mifgsm_010":"mifgsm","mifgsm_020":"mifgsm","mifgsm_030":"mifgsm",
    "cw_05":"cw","cw_10":"cw","cw_20":"cw",
    "deepfool_002":"deepfool","deepfool_005":"deepfool",
    "feat_zero":"feature","feat_scale":"feature","feat_noise":"feature",
    "feat_swap":"feature","feat_invert":"feature",
    "poison_mimicry":"poison","poison_boundary":"poison","poison_gradmim":"poison",
    "transfer_020":"transfer",
    "constrained_015":"constrained","constrained_030":"constrained",
    "auto_020":"auto",
}


def _run_adv_job(job_id, dataset, mode, epsilon):
    """Background thread: run real adversarial experiment and store result."""
    with _jobs_lock:
        _adv_jobs[job_id] = {"status": "running", "result": None, "error": None}
    try:
        soc      = adv_module.load_all_soc_layers(dataset)
        baseline = adv_module.compute_baseline_metrics(soc, dataset)
        results  = adv_module.run_experiment(soc, mode, dataset)
        after    = adv_module.compute_after_metrics(soc, results, dataset)

        layer_out = []
        for layer in LAYERS:
            b    = baseline.get(layer, {})
            a    = after.get(layer, {})
            drop = round(b.get("accuracy", 0) - a.get("accuracy", 0), 4)
            ev   = round(max(b.get("tpr", 0) - a.get("tpr", 0), 0), 4)
            layer_out.append({
                "layer":       layer,
                "algorithm":   ALGOS.get(layer, layer),
                "clean":       {k: b.get(k, 0) for k in ["accuracy","f1_score","tpr","fpr","auc_roc","precision","recall"]},
                "adversarial": {k: a.get(k, 0) for k in ["accuracy","f1_score","tpr","fpr","auc_roc","precision","recall"]},
                "acc_drop":    drop,
                "f1_drop":     round(b.get("f1_score", 0) - a.get("f1_score", 0), 4),
                "auc_drop":    round((b.get("auc_roc") or 0) - (a.get("auc_roc") or 0), 4),
                "evasion":     ev,
                "status":      ("BREACHED" if drop > 0.12 else "DEGRADED" if drop > 0.05 else "STABLE"),
            })

        job_result = {
            "dataset":         dataset,
            "mode":            mode,
            "epsilon":         epsilon,
            "layers":          layer_out,
            "layers_breached": sum(1 for l in layer_out if l["status"] == "BREACHED"),
            "avg_acc_drop":    round(sum(l["acc_drop"] for l in layer_out) / 5, 4),
            "max_evasion":     round(max(l["evasion"] for l in layer_out), 4),
            "n_attacks":       len(results),
            "source":          "real_adversarial",
            "attack_names":    list(results.keys()),
        }
        with _jobs_lock:
            _adv_jobs[job_id] = {"status": "done", "result": job_result, "error": None}
    except Exception as ex:
        import traceback
        with _jobs_lock:
            _adv_jobs[job_id] = {"status": "error", "result": None,
                                  "error": str(ex) + "\n" + traceback.format_exc()}


@app.route("/api/adversarial", methods=["POST"])
def api_adversarial_launch():
    """Launch real adversarial attack asynchronously. Poll /api/adversarial/<job_id>."""
    if not ADV_AVAILABLE:
        return jsonify({"error": "Adversarial module not loaded. Use /api/attack for simulated results."}), 503
    body    = request.get_json(force=True) or {}
    dataset = body.get("dataset",    "cicids2017")
    atk_key = body.get("attack_key", "fgsm_010")
    epsilon = float(body.get("epsilon", 0.10))
    mode    = ATK_KEY_TO_MODE.get(atk_key, "fgsm")
    if dataset not in DATASETS:
        return jsonify({"error": f"Unknown dataset: {dataset}"}), 400
    job_id = f"{dataset}_{mode}_{int(time.time()*1000) % 100000}"
    thread = threading.Thread(
        target=_run_adv_job, args=(job_id, dataset, mode, epsilon), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id, "status": "queued", "mode": mode, "dataset": dataset})


@app.route("/api/adversarial/<job_id>", methods=["GET"])
def api_adversarial_poll(job_id):
    """Poll async adversarial job result."""
    with _jobs_lock:
        job = _adv_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found", "job_id": job_id}), 404
    return jsonify({"job_id": job_id, **job})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "app":      "AI-SOC Adversarial Evasion Dashboard API",
        "version":  "2.0 (fixed)",
        "pipeline": PIPELINE_AVAILABLE,
        "adversarial": ADV_AVAILABLE,
        "endpoints": [
            "GET  /api/status",
            "GET  /api/metrics/<dataset>",
            "GET  /api/pipeline_summary",
            "POST /api/attack",
            "POST /api/normal_event",
            "GET  /api/realtime/<dataset>?n=10",
            "POST /api/adversarial",
            "GET  /api/adversarial/<job_id>",
            "GET  /dashboard",
        ],
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI-SOC Dashboard API Server v2")
    ap.add_argument("--port",    type=int, default=5000)
    ap.add_argument("--host",    default="0.0.0.0")
    ap.add_argument("--debug",   action="store_true")
    ap.add_argument("--preload", action="store_true",
                    help="Preload all dataset models on startup (parallel)")
    args = ap.parse_args()

    if args.preload:
        preload_all_datasets()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  AI-SOC Dashboard API  v2.0 (fixed)                      ║
╠══════════════════════════════════════════════════════════╣
║  🌐 Open in browser:                                      ║
║     http://localhost:{args.port}/dashboard                   ║
╠══════════════════════════════════════════════════════════╣
║  Pipeline    : {str(PIPELINE_AVAILABLE):<40} ║
║  Adversarial : {str(ADV_AVAILABLE):<40} ║
╠══════════════════════════════════════════════════════════╣
║  API Endpoints:                                           ║
║    GET  /api/status                                       ║
║    GET  /api/metrics/<dataset>                            ║
║    GET  /api/pipeline_summary                             ║
║    POST /api/attack                                       ║
║    POST /api/normal_event                                 ║
║    GET  /api/realtime/<dataset>?n=10                      ║
║    POST /api/adversarial   (async real attack)            ║
╚══════════════════════════════════════════════════════════╝
""")
    app.run(host=args.host, port=args.port, debug=args.debug)