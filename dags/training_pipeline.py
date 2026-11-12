"""Airflow 3 training DAG: validate -> preprocess -> train -> evaluate/log (MLflow) -> quality gate.

Reliability design
  * validation + quality-gate failures raise AirflowFailException  -> NO retries (deterministic failures),
    and every downstream task is `upstream_failed`, so nothing is promoted;
  * train / evaluate are retried (transient failures: IO, tracking server hiccup);
  * the quality gate is the only place a model is registered and given the `candidate` alias.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, task

from training import pipeline_steps as ps

log = logging.getLogger("airflow.task")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
WORK_ROOT = os.getenv("MLOPS_WORK_DIR", "/tmp/mlops-agent")
GATE = float(os.getenv("F1_QUALITY_GATE", "0.93"))


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops", "retries": 2, "retry_delay": timedelta(seconds=20)},
    params={
        "dataset": Param("data/dataset.csv", type="string"),
        "n_estimators": Param(200, type="integer", minimum=1, maximum=2000),
        "max_depth": Param(8, type="integer", minimum=1, maximum=64),
    },
    tags=["mlops", "training"],
)
def training_pipeline():
    @task(retries=0)
    def validate_dataset(params=None):
        try:
            result = ps.validate_dataset(params["dataset"])
        except ps.DataValidationError as exc:
            log.error("dataset validation failed: %s", exc)
            raise AirflowFailException(str(exc)) from exc
        return {"dataset": params["dataset"], "provenance": result["provenance"], "rows": result["rows"]}

    @task
    def preprocess(validated, run_id=None):
        work = f"{WORK_ROOT}/{run_id}".replace(":", "_").replace("+", "_")
        splits = ps.preprocess(validated["dataset"], work)
        return {**splits, "work_dir": work, "provenance": validated["provenance"]}

    @task
    def train_model(prepared, params=None):
        path = ps.train_model(prepared["train"], prepared["work_dir"], params["n_estimators"], params["max_depth"])
        return {**prepared, "model_path": path}

    @task
    def evaluate_and_log(trained, params=None, run_id=None):
        ev = ps.evaluate_and_log(
            trained["model_path"],
            trained["test"],
            {"n_estimators": params["n_estimators"], "max_depth": params["max_depth"], "airflow_run_id": run_id},
            trained["provenance"],
            TRACKING_URI,
            run_name=f"airflow-{run_id}"[:100],
        )
        log.info("evaluation: %s", ev["metrics"])
        return ev["run_id"]

    @task(retries=0)
    def apply_model_quality_gate(mlflow_run_id):
        try:
            return ps.quality_gate(mlflow_run_id, GATE, TRACKING_URI)
        except ps.QualityGateError as exc:
            log.error("quality gate failed: %s", exc)
            raise AirflowFailException(str(exc)) from exc

    validated = validate_dataset()
    prepared = preprocess(validated)
    trained = train_model(prepared)
    run = evaluate_and_log(trained)
    apply_model_quality_gate(run)


training_pipeline()
