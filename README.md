# 🏆 MLOps Model Tournament Pipeline

![CI](https://github.com/jumma786/mlops-model-tournament/actions/workflows/tournament.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.13-orange)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Bank%20Marketing-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Part of the MLOps Portfolio Series** — Project 1 of 10  
> A production-grade multi-model tournament pipeline that automatically trains, evaluates, and promotes the best ML model using MLflow experiment tracking and GitHub Actions CI/CD.

---

## 🎯 What This Project Does

Instead of manually comparing models in a notebook, this pipeline:

1. **Trains 5 models** — LogReg, Random Forest, XGBoost, LightGBM, CatBoost
2. **Logs everything to MLflow** — params, metrics, plots, and model artifacts per run
3. **Auto-selects the champion** by ROC-AUC score
4. **Gates promotion** — only registers the champion if AUC ≥ configurable threshold
5. **Runs automatically** on every push to `main` via GitHub Actions (+ weekly cron)
6. **Generates an HTML comparison report** as a CI artifact

The pipeline makes the model selection decision — no human needed unless AUC drops below threshold.

---

## 📊 Dataset

**UCI Bank Marketing** — Real dataset, 41,188 rows

| Property | Value |
|---|---|
| Source | Moro et al., 2014 — [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| Rows | 41,188 (real data) |
| Train / Test | 32,950 / 8,238 (80/20 split) |
| Features | 19 (after dropping `duration` — leakage risk) |
| Target | Term deposit subscription (binary: yes/no) |
| Class balance | 11.3% positive (imbalanced) |

**Why `duration` is dropped:** Call duration is unknown before the call is made — using it causes data leakage. Documented in `src/data/loader.py` and flagged as a warning at runtime. See: Moro et al. (2014).

> Dataset not included in repo. Download via Kaggle CLI:
> ```bash
> kaggle datasets download -d henriqueyamahata/bank-marketing -p data --unzip
> ```

---

## 🏗️ Architecture

```
mlops-model-tournament/
├── src/
│   ├── data/
│   │   └── loader.py          # Data loading, preprocessing, train/test split
│   ├── models/
│   │   └── contestants.py     # 5 model definitions (all sklearn-compatible)
│   ├── evaluation/
│   │   └── metrics.py         # Metrics, ROC curves, confusion matrices, HTML report
│   └── tournament.py          # Main runner — MLflow parent/child run orchestration
├── tests/
│   └── test_tournament.py     # 13 unit tests (data, models, metrics)
├── reports/                   # Auto-generated plots and HTML report
├── .github/
│   └── workflows/
│       └── tournament.yml     # CI: test → tournament → gate → artifact upload
├── requirements.txt
└── Makefile
```

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/jumma786/mlops-model-tournament.git
cd mlops-model-tournament
pip install -r requirements.txt

# Download dataset
kaggle datasets download -d henriqueyamahata/bank-marketing -p data --unzip

# Run tournament on real data
python src/tournament.py --data-path data/bank-additional-full.csv

# View results in MLflow UI
mlflow ui --backend-store-uri mlruns
# → Open http://localhost:5000
```

---

## 🔬 MLflow Experiment Structure

```
tournament-parent (run)
├── params: n_train, n_test, n_features, drop_duration
├── metrics: champion_auc, champion_f1
├── tags: champion=RandomForest
├── artifacts: tournament_comparison.png, tournament_report.html
│
├── LogisticRegression (nested run)
│   ├── metrics: roc_auc, f1, precision, recall, accuracy, train_time
│   ├── artifacts: roc_LogisticRegression.png, cm_LogisticRegression.png
│   └── model artifact
│
├── RandomForest (nested run)  ← champion registered to Model Registry
├── XGBoost (nested run)
├── LightGBM (nested run)
└── CatBoost (nested run)
```

---

## 📈 Results — Real Data (41,188 rows)

| Model | AUC | F1 | Precision | Recall | Accuracy | Train Time |
|---|---|---|---|---|---|---|
| **RandomForest 🏆** | **0.8174** | **0.5182** | 0.4368 | 0.6369 | 0.8666 | 4.0s |
| LightGBM | 0.8144 | 0.5023 | 0.4056 | 0.6595 | 0.8528 | 2.2s |
| CatBoost | 0.8139 | 0.4861 | 0.3879 | 0.6509 | 0.8450 | 1.4s |
| XGBoost | 0.8135 | 0.4965 | 0.4009 | 0.6519 | 0.8511 | 2.0s |
| LogisticRegression | 0.7959 | 0.4403 | 0.3315 | 0.6552 | 0.8123 | 0.1s |

**Champion:** Random Forest registered to MLflow Model Registry as `BankMarketingChampion v1`

> All four gradient boosting models cluster tightly between AUC 0.813–0.817 — meaningful separation will come from hyperparameter tuning in **Project 4 (Optuna)**. The value here is the *pipeline infrastructure*, not absolute metric numbers.

---

## ⚙️ Configuration

```bash
python src/tournament.py \
  --data-path data/bank-additional-full.csv \
  --experiment-name "bank-marketing-v2" \
  --min-auc 0.75
```

| Argument | Default | Description |
|---|---|---|
| `--data-path` | None | Path to real UCI CSV (semicolon-sep) |
| `--n-samples` | 5000 | Synthetic dataset size (if no data-path) |
| `--experiment-name` | mlops-model-tournament | MLflow experiment name |
| `--min-auc` | 0.0 | Minimum AUC to promote to registry |
| `--drop-duration` | True | Drop leakage feature |

---

## 🔄 CI/CD Pipeline

GitHub Actions runs on every push to `main`:

```
push to main
    ↓
[Unit Tests] — 13 tests across data, models, metrics
    ↓ (pass required)
[Tournament] — train 5 models, log to MLflow
    ↓
[AUC Gate] — fail pipeline if champion AUC < 0.50
    ↓ (pass required)
[Upload Artifacts] — reports/ and mlruns/ stored 30 days
```

Weekly cron (Monday 06:00 UTC) triggers the full tournament automatically.

---

## 🔗 MLOps Portfolio Series

| # | Project | Status |
|---|---|---|
| **1** | **Multi-Model Tournament Pipeline** | ✅ This repo |
| 2 | Scheduled Retraining + DVC + MLflow | 🔜 |
| 3 | Feature Engineering as Versioned Artifact | 🔜 |
| 4 | Hyperparameter Tuning with Optuna + MLflow | 🔜 |
| 5 | FastAPI + Docker + Cloud Run Deployment | 🔜 |
| 6 | Feature Store with Feast + Redis | 🔜 |
| 7 | Model Monitoring & Drift Detection | 🔜 |
| 8 | A/B Testing Framework | 🔜 |
| 9 | Airflow Pipeline Orchestration | 🔜 |
| 10 | Kubernetes ML Platform | 🔜 |

---

## 📝 Key MLOps Concepts Demonstrated

- **Experiment tracking** — every run logged with full reproducibility
- **Model versioning** — MLflow Model Registry with champion alias
- **Champion/challenger** — automated model comparison and promotion gate
- **CI/CD for ML** — GitHub Actions test → train → gate → artifact pipeline
- **Leakage discipline** — `duration` feature dropped with documented rationale
- **Class imbalance handling** — `class_weight="balanced"` and `scale_pos_weight` across all models

---

## 👤 Author

**Jumma Mohammad Teli** — Data Analyst & ML Engineer  
📍 Birmingham, UK  
🔗 [LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786)

---

*Part of a 10-project MLOps portfolio. Each project builds on the last — from experiment tracking to a full Kubernetes ML platform.*
