# Live agent conversations

The five live conversations below are stored as exact JSON responses:

1. `live-agent-dataset.json` - DVC/MLflow dataset provenance.
2. `live-agent-mlflow.json` - recent run comparison and candidate selection.
3. `live-agent-airflow.json` - Airflow DAG and task states.
4. `live-agent-kubernetes.json` - live deployment diagnosis.
5. `live-agent-multitool.json` - cross-system reasoning without triggering
   state-changing actions.

The live results report a healthy 2/2 Kubernetes deployment, a successful
five-task Airflow run, a passed MLflow quality gate, and explicit uncertainty
where offline metrics do not prove production readiness.
