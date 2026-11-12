"""Pipeline logic shared by the Airflow DAG, the CLI (training/train.py) and the unit tests.

Stages (Figure 2): validate -> preprocess -> train -> evaluate_and_log -> quality_gate.
Each stage is a plain function so it can be tested without Airflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("training")

TARGET = "target"
N_FEATURES = 30
MODEL_NAME = "iris-inference-model"  # registry name used by the deployed service (name kept generic)
EXPERIMENT = "breast-cancer-classifier"


class DataValidationError(Exception):
    """Dataset is missing, corrupted, or does not match its DVC-recorded version."""


class QualityGateError(Exception):
    """Model did not meet the quality threshold; promotion must not happen."""


def repo_root() -> Path:
    return Path(os.getenv("MLOPS_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True, check=False)  # noqa: S603,S607
    return out.stdout.strip()


def dataset_provenance(dataset: str = "data/dataset.csv") -> dict[str, Any]:
    """Provenance recorded on every MLflow run: DVC md5 + git commit + human version label."""
    pointer = repo_root() / f"{dataset}.dvc"
    md5 = None
    if pointer.exists():
        md5 = (yaml.safe_load(pointer.read_text())["outs"][0]).get("md5")
    last = _git("log", "-1", "--format=%H", "--", f"{dataset}.dvc")
    tags = sorted(t for t in _git("tag", "--points-at", last).split() if t.startswith("data-")) if last else []
    return {
        "dataset.path": dataset,
        "dataset.md5": md5,
        "dataset.version": tags[-1] if tags else (f"md5:{md5[:8]}" if md5 else None),
        "dataset.git_commit": _git("rev-parse", "HEAD") or None,
        "dataset.pointer_commit": last or None,
    }


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - DVC itself uses md5 for content addressing
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 1) validate ------------------------------------------------------------------------------
def validate_dataset(dataset: str = "data/dataset.csv", min_rows: int = 100) -> dict[str, Any]:
    path = repo_root() / dataset
    if not path.exists():
        raise DataValidationError(f"dataset {dataset} not found - run `dvc checkout`/`dvc pull`")
    prov = dataset_provenance(dataset)
    actual = _md5(path)
    if prov["dataset.md5"] and actual != prov["dataset.md5"]:
        raise DataValidationError(
            f"dataset content (md5 {actual[:8]}) differs from DVC-recorded version (md5 {prov['dataset.md5'][:8]}): "
            "data changed without `dvc add` + commit"
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns or df.shape[1] != N_FEATURES + 1:
        raise DataValidationError(f"schema mismatch: expected {N_FEATURES} features + '{TARGET}', got {df.shape[1]} columns")
    if len(df) < min_rows:
        raise DataValidationError(f"only {len(df)} rows (< {min_rows})")
    if df.isnull().any().any():
        raise DataValidationError(f"{int(df.isnull().sum().sum())} null values found")
    if df[TARGET].nunique() < 2:
        raise DataValidationError("target has a single class")
    log.info("validation ok: %d rows, md5 %s", len(df), actual[:8])
    return {"dataset": dataset, "rows": len(df), "md5": actual, "provenance": prov}


# 2) preprocess ----------------------------------------------------------------------------
def preprocess(dataset: str, work_dir: str, seed: int = 42) -> dict[str, str]:
    df = pd.read_csv(repo_root() / dataset)
    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=seed)
    out = Path(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    log.info("preprocess: train=%d test=%d", len(train), len(test))
    return {"train": str(out / "train.csv"), "test": str(out / "test.csv"), "n_train": len(train), "n_test": len(test)}  # type: ignore[dict-item]


# 3) train ---------------------------------------------------------------------------------
def train_model(train_csv: str, work_dir: str, n_estimators: int = 200, max_depth: int = 8, seed: int = 42) -> str:
    df = pd.read_csv(train_csv)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=1)),
        ]
    )
    model.fit(df.drop(columns=[TARGET]), df[TARGET])
    path = Path(work_dir) / "model.joblib"
    joblib.dump(model, path)
    log.info("trained model -> %s", path)
    return str(path)


# 4) evaluate + log to MLflow ---------------------------------------------------------------
def evaluate_and_log(
    model_path: str,
    test_csv: str,
    params: dict[str, Any],
    provenance: dict[str, Any],
    tracking_uri: str,
    experiment: str = EXPERIMENT,
    run_name: str | None = None,
) -> dict[str, Any]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    model = joblib.load(model_path)
    test = pd.read_csv(test_csv)
    x, y = test.drop(columns=[TARGET]), test[TARGET]
    pred = model.predict(x)
    metrics = {
        "f1": f1_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
    }
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({k: v for k, v in provenance.items() if v is not None})
        mlflow.set_tag("quality_gate", "pending")
        try:
            import mlflow.sklearn

            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:  # noqa: BLE001 - fall back to raw artifact so the run still carries the model
            mlflow.log_artifact(model_path)
        mlflow.log_text(json.dumps({"params": params, "metrics": metrics}, indent=2), "summary.json")
        run_id = run.info.run_id
    log.info("logged run %s metrics=%s", run_id, {k: round(v, 4) for k, v in metrics.items()})
    return {"run_id": run_id, "metrics": metrics}


# 5) quality gate --------------------------------------------------------------------------
def quality_gate(run_id: str, threshold: float, tracking_uri: str, metric: str = "f1", register: bool = True) -> dict[str, Any]:
    """Fail the pipeline (and block promotion) if `metric` < threshold. On pass, mark and register candidate."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    value = client.get_run(run_id).data.metrics.get(metric)
    if value is None or value < threshold:
        client.set_tag(run_id, "quality_gate", "failed")
        raise QualityGateError(f"{metric}={value} is below the gate {threshold}; model NOT promoted")
    client.set_tag(run_id, "quality_gate", "passed")
    result: dict[str, Any] = {"run_id": run_id, metric: value, "threshold": threshold, "promoted": False}
    if register:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        try:
            mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
            client.set_registered_model_alias(MODEL_NAME, "candidate", mv.version)
            result.update({"promoted": True, "model_version": mv.version, "alias": "candidate"})
        except Exception as exc:  # noqa: BLE001
            log.warning("model registration failed: %s", exc)
            result["registration_error"] = type(exc).__name__
    log.info("quality gate passed: %s", result)
    return result
