"""
Test suite for mlops-model-tournament.
Run: pytest tests/ -v --cov=src
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import generate_bank_marketing_data, preprocess, load_data
from src.models.contestants import get_models
from src.evaluation.metrics import compute_metrics


# ── Data tests ────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_generate_shape(self):
        df = generate_bank_marketing_data(n_samples=500)
        assert df.shape[0] == 500
        assert "y" in df.columns

    def test_positive_rate_reasonable(self):
        df = generate_bank_marketing_data(n_samples=2000)
        rate = df["y"].mean()
        assert 0.05 <= rate <= 0.40, f"Unexpected positive rate: {rate:.2%}"

    def test_preprocess_drops_duration(self):
        df = generate_bank_marketing_data(n_samples=200)
        df_processed = preprocess(df, drop_duration=True)
        assert "duration" not in df_processed.columns

    def test_preprocess_no_objects(self):
        df = generate_bank_marketing_data(n_samples=200)
        df_processed = preprocess(df)
        obj_cols = df_processed.select_dtypes(include="object").columns.tolist()
        assert len(obj_cols) == 0, f"Remaining object columns: {obj_cols}"

    def test_load_data_returns_correct_types(self):
        X_train, X_test, y_train, y_test, features = load_data(n_samples=500)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert len(features) > 0

    def test_train_test_split_sizes(self):
        X_train, X_test, y_train, y_test, _ = load_data(n_samples=1000, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.02

    def test_no_nulls_after_preprocess(self):
        df = generate_bank_marketing_data(n_samples=300)
        df_processed = preprocess(df)
        assert df_processed.isnull().sum().sum() == 0


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModels:
    @pytest.fixture(autouse=True)
    def setup(self):
        X_train, X_test, y_train, y_test, _ = load_data(n_samples=400, random_state=99)
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

    def test_all_five_models_present(self):
        models = get_models()
        assert len(models) == 5
        expected = {"LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"}
        assert set(models.keys()) == expected

    def test_all_models_fit_and_predict(self):
        models = get_models(random_state=42)
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            assert len(preds) == len(self.X_test), f"{name} prediction length mismatch"
            assert set(preds).issubset({0, 1}), f"{name} produced non-binary predictions"

    def test_all_models_predict_proba(self):
        models = get_models(random_state=42)
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(self.X_test)
                assert proba.shape == (len(self.X_test), 2), f"{name} proba shape wrong"
                assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), f"{name} proba doesn't sum to 1"


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_compute_metrics_keys(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8])
        metrics = compute_metrics(y_true, y_pred, y_prob)
        for key in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            assert key in metrics

    def test_perfect_classifier_metrics(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["roc_auc"] == 1.0
        assert metrics["accuracy"] == 1.0

    def test_metrics_in_valid_range(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        y_pred = (y_prob > 0.5).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"Metric {k}={v} out of range"
