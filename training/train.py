"""CLI runner of the full pipeline (same functions the Airflow DAG calls)."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

from training import pipeline_steps as ps


def run(dataset: str, n_estimators: int, max_depth: int, tracking_uri: str, gate: float,
        experiment: str = ps.EXPERIMENT, run_name: str | None = None, export: bool = True) -> dict:
    work = tempfile.mkdtemp(prefix="train-")
    v = ps.validate_dataset(dataset)
    splits = ps.preprocess(dataset, work)
    model_path = ps.train_model(splits["train"], work, n_estimators, max_depth)
    ev = ps.evaluate_and_log(model_path, splits["test"], {"n_estimators": n_estimators, "max_depth": max_depth},
                             v["provenance"], tracking_uri, experiment, run_name)
    try:
        g = ps.quality_gate(ev["run_id"], gate, tracking_uri)
    finally:
        pass
    if export:  # only reached when the gate passed -> the exported model is what the K8s image bakes in
        out = ps.repo_root() / "models"
        out.mkdir(exist_ok=True)
        shutil.copy(model_path, out / "model.joblib")
    return {"validation": v, "evaluation": ev, "gate": g}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.csv")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--gate", type=float, default=float(os.getenv("F1_QUALITY_GATE", "0.93")))
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{Path('mlflow.db').resolve()}"))
    ap.add_argument("--run-name")
    a = ap.parse_args()
    out = run(a.dataset, a.n_estimators, a.max_depth, a.tracking_uri, a.gate, run_name=a.run_name)
    print({"run_id": out["evaluation"]["run_id"], "f1": round(out["evaluation"]["metrics"]["f1"], 4), "gate": out["gate"]})
