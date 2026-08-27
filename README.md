# Cascading Accuracy Degradation from adversarial evasion attack on AI-Augmented SOC Pipelines

Adversarial robustness evaluation framework for AI-driven Security Operations Center (SOC) pipelines — measuring how adversarial attacks propagate and compound across multi-stage detection architectures, rather than evaluating models in isolation.

## Overview

Most adversarial ML research evaluates robustness at the level of a single model. This project instead evaluates a full **five-layer AI-SOC pipeline** — Detection, Ingestion, SIEM, SOAR, and Triage — to measure how adversarial perturbations at the entry point cascade and compound through downstream stages.

The project includes an attack engine, a Flask backend to serve results, and an interactive dashboard for exploring accuracy degradation across attack families and pipeline layers.

## Key Findings

- **Two distinct threat models emerge:**
  - *Bounded perturbation attacks* (gradient-based methods) cluster around a **30–35% accuracy drop**.
  - *Distributional substitution attacks* (the Poison-Mimicry family) achieve up to a **~95% accuracy drop** — a qualitatively more dangerous class of attack.
- Degradation is statistically significant across all five pipeline layers, with **Detection** showing the largest mean drop and **SOAR** the smallest.
- Nine adversarial attack families are benchmarked: **FGSM, PGD, MI-FGSM, C&W L2, DeepFool, Feature Manipulation, Log Poisoning/Mimicry, Transferability,** and **AutoAttack**.

## Datasets

- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — network intrusion flow data
- [EMBER](https://github.com/elastic/ember) — malware feature dataset
- [LogHub](https://github.com/logpai/loghub) — system/application log data

## Statistical Methodology

Results are validated with:
- Shapiro-Wilk normality testing
- Wilcoxon signed-rank test
- BCa (bias-corrected and accelerated) bootstrap confidence intervals
- Rank-biserial effect size
- Four-tier severity classification

## Project Structure

```
.
├── ai-soc/                 # Core pipeline code — attack engine, backend, dashboard
├── ai_soc_models/          # Models used by the AI-SOC pipeline
├── ai_soc_plots/           # Generated result plots (pipeline runs)
├── ai_soc_results/         # Generated result data (pipeline runs)
├── adversarial_plots/      # Generated plots (adversarial attack runs)
├── adversarial_results/    # Generated result data (adversarial attack runs)
├── trained_models/         # Saved model checkpoints
├── loghub/                 # LogHub dataset files
├── datapreprocessor.py     # Data preprocessing utilities
├── ember.py                # EMBER dataset loading / feature extraction
├── generate_datasets.py    # Dataset generation / assembly script
├── requirements.txt
├── .gitignore
└── .gitattributes
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Preparing the Data

```bash
python generate_datasets.py
python datapreprocessor.py
```

### Running the Pipeline

```bash
python ai-soc/app.py
```

> Replace `app.py` with the actual script that launches the attack engine / dashboard inside `ai-soc/` — let me know its name and I'll fill this in precisely.

## Dashboard

The dashboard visualizes accuracy degradation across all nine attack families and five pipeline layers, with adjustable epsilon (perturbation bound) up to 0.30 and per-attack danger rankings grounded in interpretable CICIDS2017 flow features (Flow Duration, Packet Length, Flow Bytes/s, etc.) rather than abstract mathematical notation.

## Authors

- **Himavanth** — JSS Science and Technology University, Dept. of Computer Science and Engineering (Cybersecurity)
- **Dr. Madhusudhan G.** — JSS STU, Cybersecurity Department

## Citation

If you use this work, please cite:

> *Cascading Accuracy Degradation in AI-SOC Pipelines* (ACIG), Himavanth — JSS Science and Technology University.

