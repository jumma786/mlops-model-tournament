"""
Model definitions for the tournament.
All 5 contestants: LogReg, Random Forest, XGBoost, LightGBM, CatBoost.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


def get_models(random_state: int = 42) -> dict:
    """
    Returns dict of model_name -> fitted-ready sklearn-compatible estimator.
    All models use default hyperparameters — tuning is Project 4's job.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
                C=1.0,
            ))
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),

        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=8,       # handles class imbalance (~11% positive)
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        ),

        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        ),

        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0,
        ),
    }

    return models
