# Screenshots to capture (submission evidence)
Capture from your own running stack and save here with these names:
- `dvc_status_tags.png` - `dvc status`, `git tag`, `dvc diff data-v1 data-v2`
- `mlflow_runs.png` - MLflow experiment table (f1, dataset.version, quality_gate columns) + model registry
- `airflow_dag_graph.png`, `airflow_runs.png` - training_pipeline graph; one success and one quality-gate failure
- `k8s_resources.png` - `kubectl -n mlops-agent get deploy,pods,svc,cm` (2/2 replicas, ClusterIP)
- `k8s_failure_*.png` - during failure injection (`kubernetes/failure-injection/`)
- `agent_chat.png` - the five scenarios via `/docs`
Text exports already in `../evidence/` (dvc.txt, mlflow.txt, airflow.txt from a real `airflow dags test` run).
Kubernetes evidence was NOT produced in the build sandbox (no cluster): `evidence/kubernetes.txt` is empty/failed there - run `scripts/collect_evidence.sh` on your cluster.
