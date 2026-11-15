from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

from agent.llm import LLMReply
from agent.tools.airflow_tools import AirflowClient
from agent.tools.registry import ToolRegistry, build_registry


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("MLOPS_REPO_ROOT", str(repo))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://airflow.test")
    monkeypatch.setenv("AIRFLOW_USERNAME", "agent")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "pw-not-a-real-secret")
    monkeypatch.setenv("AIRFLOW_APPROVED_DAGS", "training_pipeline")
    monkeypatch.setenv("K8S_APPROVED_NAMESPACE", "mlops-agent")
    monkeypatch.setenv("AGENT_LLM", "rules")
    monkeypatch.setenv("TOOL_TIMEOUT_S", "5")
    monkeypatch.setenv("HTTP_TIMEOUT_S", "2")
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    AirflowClient._token = None
    return repo


@pytest.fixture
def mlflow_runs(tmp_path):
    """Three FINISHED runs in a real (sqlite) MLflow store, oldest -> newest."""
    import os

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    exp_id = client.create_experiment("exp-a")
    base = int(time.time() * 1000) - 3_600_000
    specs = [
        (
            "run-old",
            0.90,
            {"dataset.version": "data-v1", "dataset.md5": "m1" * 16, "dataset.git_commit": "c1" * 20, "quality_gate": "passed"},
        ),
        (
            "run-best",
            0.96,
            {"dataset.version": "data-v2", "dataset.md5": "m2" * 16, "dataset.git_commit": "c2" * 20, "quality_gate": "passed"},
        ),
        ("run-new-noprov", 0.94, {}),
    ]
    ids = {}
    for i, (name, f1, tags) in enumerate(specs):
        run = client.create_run(exp_id, start_time=base + i * 1000, tags={"mlflow.runName": name, **tags})
        client.log_metric(run.info.run_id, "f1", f1)
        client.log_param(run.info.run_id, "n_estimators", str(50 * (i + 1)))
        client.set_terminated(run.info.run_id, "FINISHED", end_time=base + i * 1000 + 500)
        ids[name] = run.info.run_id
    return ids


class ScriptedLLM:
    name = "scripted"

    def __init__(self, replies: list[LLMReply]) -> None:
        self.replies = list(replies)

    def complete(self, system, messages, tools):
        return self.replies.pop(0) if self.replies else LLMReply(text="done")


class BrokenLLM:
    name = "broken"

    def complete(self, system, messages, tools):
        raise RuntimeError("LLM API down")


B = "b" * 32
CANNED: dict[str, Any] = {
    "find_best_run": {
        "metric": "f1",
        "mode": "max",
        "considered_runs": 3,
        "best": {"run_id": B, "metrics": {"f1": 0.97}, "params": {"n_estimators": "300"}, "started": "2026-11-01T00:00:00Z"},
        "ranking": [{"run_id": B, "f1": 0.97}, {"run_id": "a" * 32, "f1": 0.95}],
        "readiness": {
            "checks": {"quality_gate": "passed", "age_days": 3.0},
            "caveats": ["offline metrics alone do not establish production readiness"],
        },
    },
    "get_experiment_dataset": {
        "runs_inspected": 1,
        "provenance": [
            {
                "run_id": B,
                "dataset_version": "data-v2",
                "dataset_md5": "a844f94d" * 4,
                "dataset_git_commit": "4ec26521" * 5,
                "recorded": True,
                "commit_exists_in_local_repo": True,
            }
        ],
        "distinct_dataset_versions": ["data-v2"],
        "current_workspace_md5": "a844f94d" * 4,
    },
    "diagnose_deployment": {"deployment": "iris-inference", "desired": 2, "ready": 2, "healthy": True, "findings": [], "pods": []},
    "get_deployed_version": {
        "images": ["mlops-inference:1.0.0"],
        "config_app_version": "1.0.0",
        "app_version_label": "1.0.0",
        "model_run_id": B,
        "dataset_version": "data-v2",
    },
    "get_dag_run_status": {
        "dag_id": "training_pipeline",
        "runs": [{"dag_run_id": "manual__1", "state": "success"}],
        "latest_state": "success",
    },
    "list_dvc_datasets": {"datasets": [{"path": "dataset.csv"}], "count": 1, "remotes": []},
    "restart_deployment": {"restarted": "iris-inference", "namespace": "mlops-agent", "restartedAt": "now"},
    "trigger_training_dag": {"triggered": True, "dag_id": "training_pipeline", "dag_run_id": "manual__x", "state": "queued"},
    "find_failed_tasks": {"dag_id": "training_pipeline", "run_id": "r1", "failed_tasks": [], "root_cause_candidates": []},
}


def stub_registry(overrides: dict[str, Any] | None = None, calls: list[str] | None = None) -> ToolRegistry:
    """Real tool names/schemas/permissions, canned or injected behaviour (no external service needed)."""
    reg = build_registry()
    overrides = overrides or {}
    for name, tool in reg._tools.items():
        payload = overrides.get(name, CANNED.get(name, {}))

        def fn(args, _n=name, _p=payload):
            if calls is not None:
                calls.append(_n)
            if isinstance(_p, Exception):
                raise _p
            return _p

        tool.func = fn
    return reg


# --- fake Kubernetes ------------------------------------------------------------------------
def ns(**kw: Any) -> SimpleNamespace:
    return SimpleNamespace(**kw)


def make_pod(
    name: str, *, phase="Running", ready=True, waiting: str | None = None, restarts=0, last_term: str | None = None, exit_code=None
):
    state = ns(waiting=ns(reason=waiting, message=f"{waiting} detail") if waiting else None, terminated=None)
    last = ns(terminated=ns(reason=last_term, exit_code=exit_code)) if last_term else ns(terminated=None)
    cs = ns(name="api", ready=ready, restart_count=restarts, image="mlops-inference:1.0.0", state=state, last_state=last)
    return ns(
        metadata=ns(name=name, labels={"app": "iris-inference"}, creation_timestamp=None),
        spec=ns(node_name="node-1", containers=[ns(name="api", readiness_probe=object(), liveness_probe=object())]),
        status=ns(phase=phase, conditions=[ns(type="Ready", status="True" if ready else "False")], container_statuses=[cs]),
    )


def make_deployment(desired=2, ready=2, image="mlops-inference:1.0.0"):
    return ns(
        metadata=ns(name="iris-inference", generation=3),
        spec=ns(
            replicas=desired,
            template=ns(
                metadata=ns(labels={"app": "iris-inference", "app.kubernetes.io/version": "1.0.0"}),
                spec=ns(containers=[ns(name="api", image=image, env_from=[ns(config_map_ref=ns(name="cfg"))])]),
            ),
        ),
        status=ns(ready_replicas=ready, available_replicas=ready, observed_generation=3),
    )


class FakeCore:
    def __init__(self, pods, events=(), logs="INFO started", endpoints_ready=2):
        self.pods, self.events, self.logs, self.ep = pods, list(events), logs, endpoints_ready

    def list_namespaced_pod(self, ns_, label_selector=None):
        return ns(items=self.pods)

    def read_namespaced_pod(self, name, ns_):
        return next(p for p in self.pods if p.metadata.name == name)

    def read_namespaced_pod_log(self, name, ns_, tail_lines=40):
        return self.logs

    def list_namespaced_event(self, ns_, field_selector=None):
        return ns(items=self.events)

    def list_namespaced_service(self, ns_):
        return ns(
            items=[
                ns(
                    metadata=ns(name="iris-inference"),
                    spec=ns(
                        type="ClusterIP", cluster_ip="10.0.0.1", ports=[ns(port=80, target_port=8000)], selector={"app": "iris-inference"}
                    ),
                )
            ]
        )

    def read_namespaced_endpoints(self, name, ns_):
        return ns(subsets=[ns(addresses=[object()] * self.ep, not_ready_addresses=[])] if self.ep else [])

    def read_namespaced_config_map(self, name, ns_):
        return ns(data={"APP_VERSION": "1.0.0", "MODEL_RUN_ID": B, "DATASET_VERSION": "data-v2"})


class FakeApps:
    def __init__(self, deployment):
        self.deployment, self.patched = deployment, []

    def list_namespaced_deployment(self, ns_):
        return ns(items=[self.deployment])

    def read_namespaced_deployment(self, name, ns_):
        return self.deployment

    def patch_namespaced_deployment(self, name, ns_, body):
        self.patched.append((name, ns_, body))


def warning_event(reason, message, obj="iris-inference-abc"):
    return ns(
        type="Warning",
        reason=reason,
        message=message,
        count=3,
        last_timestamp=None,
        event_time=None,
        metadata=ns(creation_timestamp=None),
        involved_object=ns(name=obj),
    )


@pytest.fixture
def k8s(monkeypatch):
    from agent.tools import kubernetes_tools as kt

    def install(pods, deployment=None, **kw):
        core, apps = FakeCore(pods, **kw), FakeApps(deployment or make_deployment())
        monkeypatch.setattr(kt, "_core", lambda: core)
        monkeypatch.setattr(kt, "_apps", lambda: apps)
        return core, apps

    return install
