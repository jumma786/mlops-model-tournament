"""
MLOps Model Tournament — Main Runner
=====================================
Trains 5 models, logs everything to MLflow, selects champion by AUC,
registers champion to MLflow Model Registry.

Usage:
    python src/tournament.py
    python src/tournament.py --n-samples 10000 --experiment-name "bank-marketing-v2"
"""

import os
import sys
import time

# Allow MLflow file-based tracking store (local development)
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import argparse
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_data
from src.models.contestants import get_models
from src.evaluation.metrics import (
    compute_metrics, plot_roc_curve, plot_confusion_matrix,
    plot_tournament_comparison, generate_comparison_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "mlops-model-tournament"
REGISTRY_NAME   = "BankMarketingChampion"
REPORTS_DIR     = "reports"
CHAMPION_AUC_THRESHOLD = 0.0   # First run: promote any champion


def parse_args():
    p = argparse.ArgumentParser(description="MLOps Model Tournament")
    p.add_argument("--n-samples",       type=int,   default=5000)
    p.add_argument("--test-size",       type=float, default=0.2)
    p.add_argument("--random-state",    type=int,   default=42)
    p.add_argument("--experiment-name", type=str,   default=EXPERIMENT_NAME)
    p.add_argument("--drop-duration",   action="store_true", default=True,
                   help="Drop 'duration' feature (leakage risk)")
    p.add_argument("--min-auc",         type=float, default=CHAMPION_AUC_THRESHOLD,
                   help="Minimum AUC to promote to registry")
    p.add_argument("--data-path",       type=str,   default=None,
                   help="Path to real UCI CSV (semicolon-separated). If omitted, uses synthetic data.")
    return p.parse_args()


def run_tournament(args):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data(
        n_samples=args.n_samples,
        test_size=args.test_size,
        random_state=args.random_state,
        drop_duration=args.drop_duration,
        data_path=args.data_path,
    )

    # ── MLflow setup ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(args.experiment_name)

    models    = get_models(random_state=args.random_state)
    results   = {}
    run_ids   = {}

    # ── Parent run: wraps all child runs ───────────────────────────────────
    with mlflow.start_run(run_name="tournament-parent") as parent_run:
        mlflow.set_tag("tournament_type", "multi-model-comparison")
        mlflow.set_tag("dataset",         "bank-marketing-synthetic")
        mlflow.set_tag("n_contestants",   len(models))
        mlflow.log_param("n_train",       len(X_train))
        mlflow.log_param("n_test",        len(X_test))
        mlflow.log_param("n_features",    len(feature_names))
        mlflow.log_param("drop_duration", args.drop_duration)

        # ── Train each contestant ──────────────────────────────────────────
        for model_name, model in models.items():
            logger.info(f"\n{'='*50}\nTraining: {model_name}\n{'='*50}")

            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                run_ids[model_name] = child_run.info.run_id
                mlflow.set_tag("model_type", model_name)

                # Log hyperparams
                params = {}
                if hasattr(model, "get_params"):
                    params = {k: str(v) for k, v in model.get_params().items()}
                elif hasattr(model, "named_steps"):
                    params = {k: str(v) for k, v in model.named_steps["clf"].get_params().items()}
                mlflow.log_params({k: v for k, v in list(params.items())[:20]})

                # Train
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = round(time.time() - t0, 2)

                # Predict
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.decision_function(X_test)

                # Metrics
                metrics = compute_metrics(y_test, y_pred, y_prob)
                metrics["train_time_seconds"] = train_time
                results[model_name] = metrics

                mlflow.log_metrics(metrics)
                mlflow.log_param("training_time_s", train_time)

                logger.info(f"{model_name}: AUC={metrics['roc_auc']:.4f} | "
                            f"F1={metrics['f1']:.4f} | "
                            f"Precision={metrics['precision']:.4f} | "
                            f"Recall={metrics['recall']:.4f} | "
                            f"Train time={train_time}s")

                # Plots
                roc_path = plot_roc_curve(y_test, y_prob, model_name, REPORTS_DIR)
                cm_path  = plot_confusion_matrix(y_test, y_pred, model_name, REPORTS_DIR)
                mlflow.log_artifact(roc_path)
                mlflow.log_artifact(cm_path)

                # Log model
                try:
                    if "XGBoost" in model_name:
                        mlflow.xgboost.log_model(model, model_name)
                    elif "LightGBM" in model_name:
                        mlflow.lightgbm.log_model(model, model_name)
                    elif "CatBoost" in model_name:
                        mlflow.catboost.log_model(model, model_name)
                    else:
                        mlflow.sklearn.log_model(model, model_name)
                except Exception as e:
                    logger.warning(f"Model logging failed for {model_name}: {e}")
                    mlflow.sklearn.log_model(model, model_name)

        # ── Select champion ────────────────────────────────────────────────
        champion = max(results, key=lambda m: results[m]["roc_auc"])
        champion_auc = results[champion]["roc_auc"]
        logger.info(f"\n{'='*50}\n🏆 CHAMPION: {champion} (AUC={champion_auc:.4f})\n{'='*50}")

        # Log champion summary to parent run
        mlflow.log_param("champion_model",  champion)
        mlflow.log_metric("champion_auc",   champion_auc)
        mlflow.log_metric("champion_f1",    results[champion]["f1"])
        mlflow.set_tag("champion", champion)

        # ── Generate comparison artefacts ──────────────────────────────────
        comparison_path = plot_tournament_comparison(results, REPORTS_DIR)
        report_path     = generate_comparison_report(results, champion, REPORTS_DIR)
        mlflow.log_artifact(comparison_path)
        mlflow.log_artifact(report_path)

        # ── Register champion to Model Registry ───────────────────────────
        if champion_auc >= args.min_auc:
            logger.info(f"Registering champion to MLflow Model Registry as '{REGISTRY_NAME}'...")
            champion_run_id = run_ids[champion]
            try:
                model_uri = f"runs:/{champion_run_id}/{champion}"
                reg = mlflow.register_model(model_uri, REGISTRY_NAME)
                logger.info(f"Registered: {REGISTRY_NAME} v{reg.version}")
                client = mlflow.tracking.MlflowClient()
                client.set_registered_model_alias(REGISTRY_NAME, "champion", reg.version)
                client.update_registered_model(
                    REGISTRY_NAME,
                    description=f"Champion: {champion} | AUC={champion_auc:.4f}"
                )
            except Exception as e:
                logger.warning(f"Registry step failed (normal on first run without server): {e}")
        else:
            logger.warning(
                f"Champion AUC {champion_auc:.4f} below threshold {args.min_auc:.4f} — NOT promoted."
            )

    # ── Print final leaderboard ────────────────────────────────────────────
    print("\n" + "="*60)
    print("🏆 TOURNAMENT LEADERBOARD")
    print("="*60)
    print(f"{'Model':<22} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Acc':>7} {'Time(s)':>8}")
    print("-"*60)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
    for model_name, m in sorted_results:
        crown = " 🏆" if model_name == champion else ""
        print(f"{model_name:<22} {m['roc_auc']:>7.4f} {m['f1']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['accuracy']:>7.4f} "
              f"{m['train_time_seconds']:>7.1f}s{crown}")
    print("="*60)
    print(f"\nReports saved to: {REPORTS_DIR}/")
    print(f"MLflow UI: mlflow ui --backend-store-uri mlruns")

    return champion, results


if __name__ == "__main__":
    args = parse_args()
    champion, results = run_tournament(args)
