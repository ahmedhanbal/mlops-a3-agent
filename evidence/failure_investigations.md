# Failure investigations


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

