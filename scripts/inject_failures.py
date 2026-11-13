"""Deliberately inject failures and show how the agent diagnoses them (writes evidence/failure_investigations.md).

A. DATA layer     - dataset modified without `dvc add`        (real DVC + real pipeline validation)
B. MODEL layer    - model misses the quality gate              (real MLflow + real pipeline gate)
C. INFRA layer    - bad image tag / bad MODEL_PATH on K8s      (simulated pod states; real runs: kubernetes/failure-injection/)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLOPS_REPO_ROOT", str(ROOT))
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
os.environ["AGENT_LLM"] = "rules"

from agent.orchestrator import Agent  # noqa: E402
from training import pipeline_steps as ps  # noqa: E402

lines = ["# Failure investigations\n"]


def section(title: str, injected: str) -> None:
    lines.extend([f"\n## {title}\n", f"**Injected failure:** {injected}\n"])


def ask(agent: Agent, q: str) -> None:
    r = agent.run(q)
    lines.extend([f"\n**User:** {q}\n", f"**Tools:** {', '.join(r.tools_used)} (request `{r.request_id}`)\n", r.answer + "\n"])


agent = Agent()
data = ROOT / "data" / "dataset.csv"

# A -------------------------------------------------------------------------------------------
section("Failure 1 - data layer: silent dataset modification", "appended a corrupted row to data/dataset.csv without `dvc add`")
original = data.read_bytes()
try:
    data.write_bytes(original + b"1,2,3\n")
    ask(agent, "Has the tracked dataset changed?")
    try:
        ps.validate_dataset()
    except ps.DataValidationError as exc:
        lines.append(f"Pipeline stage `validate_dataset` raised: `{exc}`  -> DAG run stops, no downstream task runs.\n")
finally:
    data.write_bytes(original)
    subprocess.run(["dvc", "status", "-q"], cwd=ROOT, check=False)
lines.append(
    "**Resolution:** `dvc checkout data/dataset.csv.dvc` (or the original bytes restored) -> `dvc status` clean; validation passes again.\n"
)
ask(agent, "Has the tracked dataset changed?")

# B -------------------------------------------------------------------------------------------
section("Failure 2 - model layer: model fails the quality gate", "trained a 1-tree/depth-1 forest against a 0.99 F1 gate")
uri = os.environ["MLFLOW_TRACKING_URI"]
work = tempfile.mkdtemp()
v = ps.validate_dataset()
s = ps.preprocess("data/dataset.csv", work)
m = ps.train_model(s["train"], work, n_estimators=1, max_depth=1)
ev = ps.evaluate_and_log(m, s["test"], {"n_estimators": 1, "max_depth": 1}, v["provenance"], uri, run_name="injected-weak-model")
try:
    ps.quality_gate(ev["run_id"], 0.99, uri)
except ps.QualityGateError as exc:
    lines.append(f"Pipeline stage `apply_model_quality_gate` raised: `{exc}` -> run tagged quality_gate=failed, nothing registered.\n")
ask(agent, "Compare the last five training runs and recommend the best candidate based on F1 score.")
lines.append("**Resolution:** the failed run is excluded from promotion; retrain with better hyper-parameters/data and re-run the DAG.\n")

# C -------------------------------------------------------------------------------------------
section(
    "Failure 3 - infrastructure layer (simulated pod states)",
    "bad image tag -> ImagePullBackOff; wrong MODEL_PATH -> pods Running but not Ready",
)
from agent.tools import kubernetes_tools as kt  # noqa: E402
from tests.conftest import FakeApps, FakeCore, make_deployment, make_pod, warning_event  # noqa: E402

core = FakeCore(
    [make_pod("iris-inference-new-1", phase="Pending", ready=False, waiting="ImagePullBackOff")],
    endpoints_ready=0,
    events=[warning_event("Failed", "Failed to pull image mlops-inference:does-not-exist: not found")],
)
kt._core, kt._apps = lambda: core, lambda: FakeApps(make_deployment(2, 0, "mlops-inference:does-not-exist"))
lines.append("\n### 3a. bad image tag\n")
ask(agent, "Why is my prediction API unavailable?")
core = FakeCore(
    [make_pod("iris-inference-a", ready=False), make_pod("iris-inference-b", ready=False)],
    endpoints_ready=0,
    events=[warning_event("Unhealthy", "Readiness probe failed: HTTP probe failed with statuscode: 503")],
    logs="ERROR inference model load failed: FileNotFoundError models/missing.joblib",
)
kt._core, kt._apps = lambda: core, lambda: FakeApps(make_deployment(2, 0))
lines.append("\n### 3b. bad MODEL_PATH (readiness failing)\n")
ask(agent, "Why is my prediction API unavailable?")
lines.append(
    "**Resolution:** `kubectl rollout undo deploy/iris-inference` (3a) / restore ConfigMap + `restart_deployment` with confirmation (3b).\n"
)

(ROOT / "evidence").mkdir(exist_ok=True)
(ROOT / "evidence" / "failure_investigations.md").write_text("\n".join(lines) + "\n")
print("wrote evidence/failure_investigations.md")
