#!/usr/bin/env bash
# Reproducibly create the demo state: DVC-tracked dataset v1 and v2 (git tags data-v1/data-v2),
# a local DVC remote, and several MLflow runs (incl. one that deliberately fails the quality gate).
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
REMOTE="${DVC_REMOTE_DIR:-$PWD/dvc-remote}"

[ -d .dvc ] || { dvc init -q && git add .dvc .dvcignore && git commit -q -m "chore(dvc): initialise DVC"; }
dvc remote add -d -f -q --local storage "$REMOTE"

train() { python -m training.train --run-name "$1" --n-estimators "$2" --max-depth "$3" "${@:4}" 2>/dev/null | tail -1; }

for v in v1 v2; do
  python -m training.prepare_data --version "$v" >/dev/null
  dvc add -q data/dataset.csv
  git add data/dataset.csv.dvc data/.gitignore
  git commit -q -m "data: dataset $v tracked with DVC" || true
  git tag -f "data-$v" >/dev/null
  dvc push -q
  echo "== dataset $v (md5 $(grep md5 data/dataset.csv.dvc | head -1 | awk '{print $3}' | cut -c1-8)) =="
  train "rf-$v-small" 50 4
  train "rf-$v-medium" 150 8
  train "rf-$v-large" 300 12
done
echo "== deliberately weak model (quality gate must FAIL) =="
train "rf-v2-strict-gate" 1 1 --gate 0.99 || echo "gate failed as designed (f1 < 0.99), model not promoted"
