"""
Data loading and preprocessing for the UCI Bank Marketing dataset.
Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_bank_marketing_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the UCI Bank Marketing schema.
    NOTE: This is synthetic data — clearly labelled as such.
    Replace with real UCI dataset by calling load_uci_bank_marketing() below.
    """
    np.random.seed(random_state)
    n = n_samples

    age = np.random.randint(18, 95, n)
    job = np.random.choice(
        ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
         "retired", "self-employed", "services", "student", "technician",
         "unemployed", "unknown"], n
    )
    marital = np.random.choice(["divorced", "married", "single", "unknown"], n)
    education = np.random.choice(
        ["basic.4y", "basic.6y", "basic.9y", "high.school",
         "illiterate", "professional.course", "university.degree", "unknown"], n
    )
    default = np.random.choice(["no", "yes", "unknown"], n, p=[0.79, 0.01, 0.20])
    housing = np.random.choice(["no", "yes", "unknown"], n, p=[0.45, 0.50, 0.05])
    loan = np.random.choice(["no", "yes", "unknown"], n, p=[0.82, 0.15, 0.03])
    contact = np.random.choice(["cellular", "telephone"], n, p=[0.63, 0.37])
    month = np.random.choice(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"], n
    )
    day_of_week = np.random.choice(["mon", "tue", "wed", "thu", "fri"], n)
    duration = np.random.exponential(250, n).astype(int)
    campaign = np.random.randint(1, 15, n)
    pdays = np.where(np.random.rand(n) < 0.13, np.random.randint(1, 30, n), 999)
    previous = np.random.randint(0, 7, n)
    poutcome = np.random.choice(["failure", "nonexistent", "success"], n, p=[0.10, 0.86, 0.04])
    emp_var_rate = np.random.choice([-3.4, -3.0, -2.9, -1.8, -1.7, 1.1, 1.4], n)
    cons_price_idx = np.random.uniform(92.2, 94.8, n).round(3)
    cons_conf_idx = np.random.uniform(-50.8, -26.9, n).round(1)
    euribor3m = np.random.uniform(0.6, 5.1, n).round(3)
    nr_employed = np.random.choice([4963.6, 5008.7, 5017.5, 5099.1, 5176.3, 5195.8, 5228.1], n)

    # Target: subscription — roughly 11% positive rate (mirrors real dataset)
    base_prob = 0.11
    prob = base_prob + (duration > 300) * 0.15 + (poutcome == "success") * 0.25
    prob = np.clip(prob, 0, 1)
    y = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan, "contact": contact,
        "month": month, "day_of_week": day_of_week, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome, "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m, "nr.employed": nr_employed, "y": y
    })

    logger.info(f"Generated synthetic Bank Marketing dataset: {df.shape}, "
                f"positive rate: {df['y'].mean():.1%}")
    logger.info("NOTE: Synthetic data — replace with real UCI dataset for production use.")
    return df


def preprocess(df: pd.DataFrame, drop_duration: bool = True) -> pd.DataFrame:
    """
    Encode categoricals, optionally drop 'duration' (leakage risk).
    duration is a known leakage feature — unknown before call is made.
    See: Moro et al., 2014 (UCI Bank Marketing paper).
    """
    df = df.copy()

    if drop_duration:
        logger.warning("Dropping 'duration' feature — leakage risk (unknown pre-call).")
        df = df.drop(columns=["duration"], errors="ignore")

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def load_data(
    n_samples: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_duration: bool = True,
    data_path: str = None,
):
    """
    Load, preprocess, and split data.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, sep=";")
        df["y"] = (df["y"] == "yes").astype(int)
    else:
        logger.info("No data path provided — generating synthetic dataset.")
        df = generate_bank_marketing_data(n_samples=n_samples, random_state=random_state)

    df = preprocess(df, drop_duration=drop_duration)

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Class balance (train): {y_train.mean():.1%} positive")

    return X_train, X_test, y_train, y_test, list(X.columns)
