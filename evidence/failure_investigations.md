# Failure investigations

## Failure 1 - data layer: silent dataset modification

Appending a corrupted row to `data/dataset.csv` without `dvc add` caused
`detect_data_changes` to report that the workspace differed from the DVC
pointer. The pipeline validation stage rejected the changed content. The
tracked version was restored with DVC, after which `dvc status` was clean and
validation passed.

## Failure 2 - model layer: quality gate blocks promotion

A weak one-tree/depth-one model was evaluated against a strict F1 threshold.
The quality gate rejected the run because its F1 was below the threshold,
marked the run as failed, and did not promote it. This demonstrates that a
metric failure blocks registry promotion rather than silently deploying a
weak model.

## Failure 3 - live Kubernetes infrastructure failures

These mutations were executed against Docker Desktop Kubernetes in namespace
`mlops-agent` and then reverted.

### 3a. Bad image tag

The deployment image was changed to a nonexistent tag. Kubernetes reported
`ErrImagePull` and `ImagePullBackOff`; the deployment had no healthy ready
replicas. The original image `mlops-inference:1.0.0` was restored.

### 3b. Bad model path

The ConfigMap `MODEL_PATH` was changed to `models/missing.joblib` and the
deployment was restarted. Pods remained running but failed the readiness probe
with HTTP 503, so rollout readiness timed out. The ConfigMap was restored and
the deployment returned to 2/2 ready replicas.

The final live resource state and rollout events are exported in
`evidence/kubernetes.txt`. The live agent diagnosis is in
`evidence/live-agent-kubernetes.json`.
