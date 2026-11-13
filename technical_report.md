# Technical Report - MLOps AI Agent (Assignment 03)

## 1. Complete system architecture
See [architecture.md](architecture.md): FastAPI interface -> orchestrator (LLM tool calling + permissions + tracing) -> tool registry -> DVC / MLflow / Airflow 3 / Kubernetes tools -> evidence-based response (facts, uncertainties, recommended actions). A FastAPI inference service (`api/main.py`, Dockerfile, `kubernetes/`) is the application the agent operates. GitHub Actions runs ruff + pytest on every PR.

## 2. Tool definitions and permissions
26 tools (7 DVC, 7 MLflow, 5 Airflow, 7 Kubernetes); each has a pydantic input schema, a JSON-schema spec given to the LLM, and a permission class. The full table is in [architecture.md](architecture.md#tool-permissions). Three state-changing tools (`checkout_dataset_version`, `trigger_training_dag`, `restart_deployment`) execute only when the API caller sets `confirm=true`; the LLM cannot set it (tested). Kubernetes access, read or write, is limited to `mlops-agent`; Airflow triggering to approved DAGs; DVC/git to an allow-list of sub-commands.

## 3. Evidence from DVC, MLflow, Airflow and Kubernetes
| System | Evidence | Origin |
|---|---|---|
| DVC | `evidence/dvc.txt`: tags data-v1/v2, md5 pointer, `dvc diff`, clean `dvc status` | real run |
| MLflow | `evidence/mlflow.txt`: 10 runs with f1, dataset.version, quality_gate; registry with `candidate` alias | real run |
| Airflow | `evidence/airflow.txt`: real Airflow 3.0.6 `dags test` runs - all 5 tasks success; second run with gate 0.999: `apply_model_quality_gate` failed on try 1 (no retry) | real run |
| Kubernetes | manifests in `kubernetes/`; agent behaviour verified against simulated pod states | **to capture on your cluster** (no cluster in the build sandbox); see `screenshots/README.md` |

Honest scope note: the Airflow *REST tools* were verified against mocked HTTP (`responses`), the DAG itself was executed for real; the Kubernetes tools were verified against in-memory fake clients. Run `scripts/collect_evidence.sh` and the demo on a real cluster before the live demonstration.

## 4. Five tested agent conversations
Reproduced from `evidence/conversations.md` (offline planner; Kubernetes simulated, Airflow REST not served in the sandbox so scenario 3 shows the agent correctly reporting it unavailable):


> DVC + MLflow: real tools against this repo. Kubernetes: in-memory fake pods (no cluster in the build sandbox). Airflow REST: not served here (the DAG itself was really executed, see airflow.txt), so scenario 3 shows the agent reporting it as unavailable. Re-run `scripts/collect_evidence.sh` against your own cluster/Airflow to replace these.

LLM back-end: `rules`


## 1. Dataset provenance

**User:** Which dataset version was used for experiment 1?

**Request id:** `932a88f7dd36` | **Tools:** get_experiment_dataset


**Agent:**

**Known facts (from tool evidence)**
- DVC: run 485ce030 used dataset version data-v2 (md5 a844f94d, git commit b334e03891); commit present in local repo: True.
- DVC: inspected runs span dataset versions ['data-v1', 'data-v2'].


**Tool evidence:**

| tool | status | ms | summary |
|---|---|---|---|
| get_experiment_dataset | ok | 667.59 | ok |

## 2. Model selection

**User:** Compare the last five training runs and recommend the best candidate based on F1 score.

**Request id:** `3279882c0f43` | **Tools:** compare_mlflow_runs


**Agent:**

**Known facts (from tool evidence)**
- MLflow: compared 5 runs on f1; leader 485ce030; params that differ: ['airflow_run_id', 'max_depth', 'n_estimators'].
- MLflow: run 485ce030 (airflow-manual__2026-11-09T08:15:00.031313+00:00) FAILED the quality gate (f1=0.97561) and must not be promoted.
- MLflow: run 27210d8a (injected-weak-model) FAILED the quality gate (f1=0.94737) and must not be promoted.
- MLflow: run be8740ae (rf-v2-strict-gate) FAILED the quality gate (f1=0.94737) and must not be promoted.

**Uncertainties**
- A higher metric on one split does not by itself make a model production-ready.


**Tool evidence:**

| tool | status | ms | summary |
|---|---|---|---|
| compare_mlflow_runs | ok | 10.91 | ok |

## 3. Pipeline execution

**User:** Show the status of the latest training pipeline runs in Airflow.

**Request id:** `32cdc4a448a2` | **Tools:** find_best_run, trigger_training_dag


**Agent:**

**Known facts (from tool evidence)**
- MLflow: best of the 5 most recent finished runs by f1 is run 485ce030 (f1=0.97561, params {'n_estimators': '200', 'max_depth': '8', 'airflow_run_id': 'manual__2026-11-09T08:15:00.031313+00:00'}, started 2026-11-09T08:15:05Z).
- Retraining assessment: justified because the best candidate's quality gate is recorded as FAILED.

**Uncertainties**
- MLflow: 3 runs tie on f1=0.97561; the choice among them rests on recency/other criteria, not on this metric.
- MLflow: quality gate is 'failed'; the candidate was not validated by the pipeline gate
- MLflow: offline metrics alone do not establish production readiness (no online/drift/latency evidence here)

**Recommended actions**
- `trigger_training_dag` changes state and was NOT executed. Re-submit with confirm=true to authorise it.
- Retraining is recommended but NOT started automatically. Ask explicitly for the training DAG trigger and confirm it.


**Tool evidence:**

| tool | status | ms | summary |
|---|---|---|---|
| find_best_run | ok | 9.51 | ok (best run 485ce030) |
| trigger_training_dag | confirmation_required | 0.01 | confirmation_required: 'trigger_training_dag' changes system state. Re-submit the request with confirm=true to |

## 4. Deployment diagnosis

**User:** Why is my prediction API unavailable?

**Request id:** `ea80f4a7b3ec` | **Tools:** diagnose_deployment


**Agent:**

**Known facts (from tool evidence)**
- Kubernetes: deployment iris-inference has 2/2 replicas ready; healthy=True.


**Tool evidence:**

| tool | status | ms | summary |
|---|---|---|---|
| diagnose_deployment | ok | 0.41 | ok (healthy=True) |

## 5. Multi-tool reasoning

**User:** Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.

**Request id:** `8ec4542bf692` | **Tools:** find_best_run, get_experiment_dataset, diagnose_deployment, get_deployed_version, get_dag_run_status


**Agent:**

**Known facts (from tool evidence)**
- MLflow: best of the 5 most recent finished runs by f1 is run 485ce030 (f1=0.97561, params {'n_estimators': '200', 'max_depth': '8', 'airflow_run_id': 'manual__2026-11-09T08:15:00.031313+00:00'}, started 2026-11-09T08:15:05Z).
- DVC: run 485ce030 used dataset version data-v2 (md5 a844f94d, git commit b334e03891); commit present in local repo: True.
- Kubernetes: deployment iris-inference has 2/2 replicas ready; healthy=True.
- Kubernetes: deployed image(s) ['mlops-inference:1.0.0']; app version 1.0.0; model run bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; dataset data-v2.
- Deployed model run (bbbbbbbb) differs from the best candidate (485ce030).
- Retraining assessment: justified because the best candidate's quality gate is recorded as FAILED.

**Uncertainties**
- Airflow evidence unavailable (get_dag_run_status: error - ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8080): Max retries exceeded with url: /auth/token (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8080): Failed to establish a new connection: [Errno 111] Connection refused"))). Nothing is claimed about it.
- MLflow: 3 runs tie on f1=0.97561; the choice among them rests on recency/other criteria, not on this metric.
- MLflow: quality gate is 'failed'; the candidate was not validated by the pipeline gate
- MLflow: offline metrics alone do not establish production readiness (no online/drift/latency evidence here)

**Recommended actions**
- Consider promoting the better candidate through the normal deploy process (a redeploy, not a retrain).
- Retraining is recommended but NOT started automatically. Ask explicitly for the training DAG trigger and confirm it.


**Tool evidence:**

| tool | status | ms | summary |
|---|---|---|---|
| find_best_run | ok | 8.66 | ok (best run 485ce030) |
| get_experiment_dataset | ok | 8.34 | ok |
| diagnose_deployment | ok | 0.52 | ok (healthy=True) |
| get_deployed_version | ok | 0.25 | ok |
| get_dag_run_status | error | 2.29 | error: ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8080): Max retries exceeded with url: /auth/ |


## 5. Two (three) failure investigations


## Failure 1 - data layer: silent dataset modification

**Injected failure:** appended a corrupted row to data/dataset.csv without `dvc add`


**User:** Has the tracked dataset changed?

**Tools:** detect_data_changes (request `80ee016a5ab1`)

**Known facts (from tool evidence)**
- DVC: workspace data differs from the version recorded in git.

**Recommended actions**
- Review the data change; run `dvc add` + commit if intended, or `dvc checkout` to restore the tracked version.

Pipeline stage `validate_dataset` raised: `dataset content (md5 6d479ba7) differs from DVC-recorded version (md5 a844f94d): data changed without `dvc add` + commit`  -> DAG run stops, no downstream task runs.

**Resolution:** `dvc checkout data/dataset.csv.dvc` (or the original bytes restored) -> `dvc status` clean; validation passes again.


**User:** Has the tracked dataset changed?

**Tools:** detect_data_changes (request `ebed0fbc43e0`)

**Known facts (from tool evidence)**
- DVC: workspace data matches the version recorded in git.


## Failure 2 - model layer: model fails the quality gate

**Injected failure:** trained a 1-tree/depth-1 forest against a 0.99 F1 gate

Pipeline stage `apply_model_quality_gate` raised: `f1=0.9473684210526315 is below the gate 0.99; model NOT promoted` -> run tagged quality_gate=failed, nothing registered.


**User:** Compare the last five training runs and recommend the best candidate based on F1 score.

**Tools:** compare_mlflow_runs (request `1ad5dc88b39a`)

**Known facts (from tool evidence)**
- MLflow: compared 5 runs on f1; leader 58e0ef73; params that differ: ['max_depth', 'n_estimators'].
- MLflow: run 27210d8a (injected-weak-model) FAILED the quality gate (f1=0.94737) and must not be promoted.
- MLflow: run be8740ae (rf-v2-strict-gate) FAILED the quality gate (f1=0.94737) and must not be promoted.

**Uncertainties**
- A higher metric on one split does not by itself make a model production-ready.

**Resolution:** the failed run is excluded from promotion; retrain with better hyper-parameters/data and re-run the DAG.


## Failure 3 - infrastructure layer (simulated pod states)

**Injected failure:** bad image tag -> ImagePullBackOff; wrong MODEL_PATH -> pods Running but not Ready


### 3a. bad image tag


**User:** Why is my prediction API unavailable?

**Tools:** diagnose_deployment (request `f7ea285bb740`)

**Known facts (from tool evidence)**
- Kubernetes: deployment iris-inference has 0/2 replicas ready; healthy=False.
- Kubernetes finding: only 0/2 replicas ready - evidence: deployment status
- Kubernetes finding: image cannot be pulled (wrong tag/registry/credentials) [pod iris-inference-new-1] - evidence: ImagePullBackOff: ImagePullBackOff detail
- Kubernetes finding: pod Pending (unschedulable or waiting for resources) [pod iris-inference-new-1] - evidence: phase Pending
- Kubernetes finding: service iris-inference has 0 ready endpoints (API unreachable) - evidence: endpoints empty

**Recommended actions**
- Fix the deployment problem first (infrastructure/config). Retraining would not resolve it.


### 3b. bad MODEL_PATH (readiness failing)


**User:** Why is my prediction API unavailable?

**Tools:** diagnose_deployment (request `2f109c10d937`)

**Known facts (from tool evidence)**
- Kubernetes: deployment iris-inference has 0/2 replicas ready; healthy=False.
- Kubernetes finding: only 0/2 replicas ready - evidence: deployment status
- Kubernetes finding: pod running but failing readiness probe [pod iris-inference-a] - evidence: Ready=False while phase=Running
- Kubernetes finding: pod running but failing readiness probe [pod iris-inference-b] - evidence: Ready=False while phase=Running
- Kubernetes finding: service iris-inference has 0 ready endpoints (API unreachable) - evidence: endpoints empty

**Recommended actions**
- Fix the deployment problem first (infrastructure/config). Retraining would not resolve it.

**Resolution:** `kubectl rollout undo deploy/iris-inference` (3a) / restore ConfigMap + `restart_deployment` with confirmation (3b).



Root causes and resolutions are stated in each section. Layer coverage: data (1), model/workflow (2), infrastructure (3a/3b).

## 6. Known limitations and proposed improvements
- Kubernetes and Airflow REST paths are verified with mocks in CI; a dedicated kind-based integration job is the next step.
- Tool timeouts stop waiting but cannot kill a running thread; use subprocess/async clients with cancellation.
- Airflow DAG passes files between tasks via a local work directory (`MLOPS_WORK_DIR`); a distributed executor needs shared storage or artifact passing through MLflow.
- Rule-based planner covers the demo intents only; with `ANTHROPIC_API_KEY` the LLM selects tools freely (not covered by CI because it needs a secret).
- Retraining triggers use offline metrics + age; add online drift/latency monitoring as another evidence source.
- DVC remote is local; use S3/GCS with restricted credentials in production. Registry/deployment names still carry the `iris-` prefix from the first draft.
