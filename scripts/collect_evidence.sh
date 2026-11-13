#!/usr/bin/env bash
# Collect textual evidence from DVC / MLflow / Airflow / Kubernetes into evidence/ (screenshots go in screenshots/).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p evidence
{ echo '$ dvc status'; dvc status; echo; echo '$ git tag'; git tag; echo; echo '$ cat data/dataset.csv.dvc'; cat data/dataset.csv.dvc; echo; echo '$ dvc diff data-v1 data-v2'; dvc diff data-v1 data-v2; } > evidence/dvc.txt 2>&1
python - > evidence/mlflow.txt 2>/dev/null <<'PY'
import os
from mlflow.tracking import MlflowClient
c = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
for e in c.search_experiments():
    print("EXPERIMENT", e.experiment_id, e.name)
    for r in c.search_runs([e.experiment_id], order_by=["attributes.start_time ASC"]):
        t = r.data.tags
        print(f"  {r.info.run_id[:8]} {t.get('mlflow.runName'):18} f1={r.data.metrics.get('f1'):.4f} data={t.get('dataset.version')} gate={t.get('quality_gate')} params={r.data.params}")
for m in c.search_registered_models():
    print("MODEL", m.name, [(v.version, v.run_id[:8]) for v in c.search_model_versions(f"name='{m.name}'")], getattr(m, "aliases", {}))
PY
{ echo '$ kubectl -n mlops-agent get deploy,pods,svc,cm'; kubectl -n mlops-agent get deploy,pods,svc,cm; echo; echo '$ kubectl -n mlops-agent describe deploy iris-inference'; kubectl -n mlops-agent describe deploy iris-inference; } > evidence/kubernetes.txt 2>&1 || true
{ echo '$ airflow dags list-runs training_pipeline'; airflow dags list-runs training_pipeline; } > evidence/airflow.txt 2>&1 || true
python scripts/demo_conversations.py
echo "Now capture screenshots of the DVC/MLflow/Airflow/Kubernetes UIs into screenshots/ (see screenshots/README.md)."
