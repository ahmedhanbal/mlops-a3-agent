# Architecture

```
User -> FastAPI (agent/main.py: validation, API key, request id)
      -> Agent (agent/orchestrator.py: LLM tool-calling loop, max 6 steps, confirm flag only from caller)
           -> ToolRegistry (agent/tools/registry.py: schema validation, timeout, permissions, structured logs)
                -> dvc_tools | mlflow_tools | airflow_tools | kubernetes_tools
      -> reasoning.py (deterministic evidence correlation) -> facts / uncertainties / recommended actions + evidence list
```
**Why these choices**
- *Registry as single choke point*: every call gets pydantic validation (`invalid_request`), timeout (`timeout`), write-confirmation (`confirmation_required`), policy (`denied`) and a JSON log line (tool, status, duration, request id; secrets redacted).
- *Structured answer is not LLM-authored*: `reasoning.correlate` derives facts/uncertainties/actions from tool results only. If all tools fail the LLM text is discarded. Failures are listed as uncertainties, never filled in.
- *LLM interface with two back-ends*: a provider-neutral OpenAI-compatible tool-calling client (e.g. DeepSeek), and an offline planner (same interface) used without API key, on LLM outage, and in CI.
- *Retraining policy*: recommended only for a failed quality gate, freshness-policy breach, or supplied degradation results; never triggered from an analysis request.

## Tool permissions
| Module | Read-only tools | State-changing (need `confirm=true`) | Restrictions |
|---|---|---|---|
| DVC | list_dvc_datasets, get_dataset_version, compare_dataset_versions, detect_data_changes, explain_dataset_reproduction, get_experiment_dataset | checkout_dataset_version | allow-listed git/dvc sub-commands, no shell, validated revs/paths, never `--force`, refuses when data has unsaved changes |
| MLflow | list_mlflow_experiments, list_mlflow_runs, get_mlflow_run_details, find_best_run, compare_mlflow_runs, compare_mlflow_experiments, get_model_versions | - | `MlflowClient` only; best run carries readiness caveats, never "production-ready" |
| Airflow | list_airflow_dags, get_dag_run_status, find_failed_tasks, get_task_diagnostics | trigger_training_dag | JWT auth; only `AIRFLOW_APPROVED_DAGS`; id validation; log tails redacted |
| Kubernetes | list_k8s_workloads, get_pod_details, list_k8s_services, find_unhealthy_pods, get_deployed_version, diagnose_deployment | restart_deployment | one approved namespace for ALL calls; RBAC Role (`kubernetes/rbac.yaml`) limited to that namespace |

## Training DAG (`dags/training_pipeline.py`)
validate_dataset -> preprocess -> train_model -> evaluate_and_log (MLflow, with dataset provenance tags) -> apply_model_quality_gate.
Retries=2 for transient stages; validation and gate raise `AirflowFailException` (retries=0) so a bad dataset/model stops the run, downstream tasks do not execute and no model is registered/aliased `candidate`.

## Provenance chain
`dvc add` -> `data/dataset.csv.dvc` (md5) -> git commit + tag `data-vN` -> training logs tags `dataset.version/md5/git_commit` on the MLflow run -> the agent reads those tags (`get_experiment_dataset`) and the DAG validates file md5 == recorded md5 before training.
