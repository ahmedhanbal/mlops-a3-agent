# Submission screenshots

These screenshots were captured from the live local stack:

- `mlflow_runs.png` - MLflow experiment list.
- `airflow_dag_graph.png` - live `training_pipeline` graph.
- `airflow_run_success.png` - successful Airflow run with all five tasks passing.
- `agent_chat.png` - agent Swagger UI.
- `k8s_inference_docs.png` - Kubernetes-served inference Swagger UI.
- `airflow_assets.png` - Airflow assets page; empty is expected because the DAG
  does not define asset outlets.

Textual exports provide the remaining evidence:

- `../evidence/dvc.txt` and `../evidence/airflow-live.txt`
- `../evidence/kubernetes.txt`
- `../evidence/failure_investigations.md`
- `../evidence/live-agent-*.json`

The Kubernetes failure injections were performed live and restored. Their
resource events and diagnosis are preserved in the evidence files above.
