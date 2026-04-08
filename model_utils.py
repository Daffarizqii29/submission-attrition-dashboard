from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "employee_data.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "attrition_model.joblib"
DEFAULT_METRICS_PATH = BASE_DIR / "model" / "model_metrics.json"


DROPPED_COLUMNS = {"Attrition", "EmployeeId", "EmployeeCount", "Over18", "StandardHours"}


def load_dataset(data_path: Path | str = DEFAULT_DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)
    labeled = df.dropna(subset=["Attrition"]).copy()
    labeled["Attrition"] = labeled["Attrition"].astype(int)
    return df, labeled


def build_training_bundle(labeled: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    features = [c for c in labeled.columns if c not in DROPPED_COLUMNS]
    X = labeled[features].copy()
    y = labeled["Attrition"].copy()

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        (
            "preprocess",
            preprocess,
        ),
        (
            "model",
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    best_threshold = 0.5
    best_f1 = -1.0
    for step in range(20, 81):
        threshold = step / 100
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_test, pred)
        if score > best_f1:
            best_threshold = threshold
            best_f1 = score

    pred = (proba >= best_threshold).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 3),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "threshold": round(float(best_threshold), 2),
    }
    bundle = {"pipeline": pipeline, "threshold": best_threshold, "features": features}
    return bundle, metrics


def get_model_bundle(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    data_path: Path | str = DEFAULT_DATA_PATH,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    persist_if_retrained: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    try:
        bundle = joblib.load(model_path)
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        return bundle, metrics, "loaded_from_file"
    except Exception:
        _, labeled = load_dataset(data_path)
        bundle, metrics = build_training_bundle(labeled)
        if persist_if_retrained:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, model_path)
            metrics_path.write_text(json.dumps(metrics, indent=2))
        return bundle, metrics, "retrained_from_dataset"
