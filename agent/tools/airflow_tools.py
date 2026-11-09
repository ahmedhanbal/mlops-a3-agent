"""Task 3 - Apache Airflow 3 workflow tools (REST API v2 with JWT authentication)."""
from __future__ import annotations

import re
from typing import Any

import requests
from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.logging_utils import redact
from agent.tools.registry import PolicyDenied, Tool

_ID_RE = re.compile(r"^[A-Za-z0-9_.:+\-]{1,250}$")


class AirflowClient:
    """Minimal Airflow 3 client: POST /auth/token -> Bearer JWT -> /api/v2/*. Token is cached, never logged."""

    _token: str | None = None

    def __init__(self) -> None:
        s = get_settings()
        self.base = s.airflow_base_url.rstrip("/")
        self.timeout = s.http_timeout_s
        self._user, self._pw = s.airflow_username, s.airflow_password

    def _authenticate(self) -> str:
        r = requests.post(f"{self.base}/auth/token", json={"username": self._user, "password": self._pw}, timeout=self.timeout)
        r.raise_for_status()
        AirflowClient._token = r.json()["access_token"]
        return AirflowClient._token

    def request(self, method: str, path: str, extra_headers: dict[str, str] | None = None, **kw: Any) -> Any:
        token = AirflowClient._token or self._authenticate()
        for attempt in (1, 2):
            resp = requests.request(
                method, f"{self.base}/api/v2{path}",
                headers={"Authorization": f"Bearer {token}", **(extra_headers or {})},
                timeout=self.timeout, **kw,
            )
            if resp.status_code == 401 and attempt == 1:  # expired token -> refresh once
                token = self._authenticate()
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Airflow API {method} {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()


def _check_id(value: str, what: str) -> str:
    if not _ID_RE.match(value):
        raise PolicyDenied(f"invalid {what} '{value}'")
    return value


class NoArgs(BaseModel):
    pass


class DagRunsArgs(BaseModel):
    dag_id: str = Field(default="training_pipeline")
    limit: int = Field(default=5, ge=1, le=25)


class TriggerArgs(BaseModel):
    dag_id: str = Field(default="training_pipeline")
    n_estimators: int | None = Field(default=None, ge=1, le=2000)
    max_depth: int | None = Field(default=None, ge=1, le=64)
    dataset: str | None = Field(default=None, description="Dataset path to train on (default data/dataset.csv)")


class FailedTasksArgs(BaseModel):
    dag_id: str = Field(default="training_pipeline")
    run_id: str | None = Field(default=None, description="DAG run id; default = most recent failed run")


class TaskDiagArgs(BaseModel):
    dag_id: str = Field(default="training_pipeline")
    run_id: str
    task_id: str
    try_number: int = Field(default=1, ge=1, le=20)
    tail_lines: int = Field(default=40, ge=5, le=200)


def list_dags(_: NoArgs) -> dict[str, Any]:
    data = AirflowClient().request("GET", "/dags", params={"limit": 100})
    return {
        "count": data.get("total_entries", len(data.get("dags", []))),
        "approved_for_triggering": list(get_settings().approved_dags),
        "dags": [{"dag_id": d["dag_id"], "paused": d.get("is_paused"), "tags": [t.get("name") for t in d.get("tags", [])]}
                 for d in data.get("dags", [])],
    }


def _run_view(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r.get(k) for k in ("dag_run_id", "state", "run_type", "logical_date", "start_date", "end_date")}


def get_dag_runs(args: DagRunsArgs) -> dict[str, Any]:
    dag = _check_id(args.dag_id, "dag_id")
    data = AirflowClient().request("GET", f"/dags/{dag}/dagRuns", params={"limit": args.limit, "order_by": "-run_after"})
    runs = [_run_view(r) for r in data.get("dag_runs", [])]
    return {"dag_id": dag, "runs": runs, "latest_state": runs[0]["state"] if runs else None}


def trigger_training_dag(args: TriggerArgs) -> dict[str, Any]:
    """STATE-CHANGING. Only DAGs in AIRFLOW_APPROVED_DAGS may be triggered."""
    dag = _check_id(args.dag_id, "dag_id")
    if dag not in get_settings().approved_dags:
        raise PolicyDenied(f"DAG '{dag}' is not approved for triggering (approved: {list(get_settings().approved_dags)})")
    conf = {k: v for k, v in args.model_dump().items() if k != "dag_id" and v is not None}
    run = AirflowClient().request("POST", f"/dags/{dag}/dagRuns", json={"logical_date": None, "conf": conf})
    return {"triggered": True, "dag_id": dag, "conf": conf, **_run_view(run)}


def get_failed_tasks(args: FailedTasksArgs) -> dict[str, Any]:
    client = AirflowClient()
    dag = _check_id(args.dag_id, "dag_id")
    run_id = args.run_id
    if not run_id:
        runs = client.request("GET", f"/dags/{dag}/dagRuns", params={"limit": 10, "order_by": "-run_after"}).get("dag_runs", [])
        failed = [r for r in runs if r.get("state") == "failed"]
        if not failed:
            return {"dag_id": dag, "run_id": None, "failed_tasks": [], "note": "no failed runs among the 10 most recent"}
        run_id = failed[0]["dag_run_id"]
    run_id = _check_id(run_id, "run_id")
    tis = client.request("GET", f"/dags/{dag}/dagRuns/{run_id}/taskInstances").get("task_instances", [])
    states = [{"task_id": t["task_id"], "state": t.get("state"), "try_number": t.get("try_number"),
               "duration_s": t.get("duration")} for t in tis]
    bad = [t for t in states if t["state"] in ("failed", "upstream_failed")]
    return {"dag_id": dag, "run_id": run_id, "task_states": states, "failed_tasks": bad,
            "root_cause_candidates": [t["task_id"] for t in bad if t["state"] == "failed"]}


def get_task_diagnostics(args: TaskDiagArgs) -> dict[str, Any]:
    client = AirflowClient()
    dag, run, task = (_check_id(args.dag_id, "dag_id"), _check_id(args.run_id, "run_id"), _check_id(args.task_id, "task_id"))
    ti = client.request("GET", f"/dags/{dag}/dagRuns/{run}/taskInstances/{task}")
    log_lines: list[str] = []
    log_error = None
    try:  # logs are best-effort diagnostics: JSON only when Accept: application/json is sent
        logs = client.request(
            "GET", f"/dags/{dag}/dagRuns/{run}/taskInstances/{task}/logs/{args.try_number}",
            extra_headers={"Accept": "application/json"}, params={"map_index": -1},
        )
        for item in logs.get("content", []):
            log_lines.append(item if isinstance(item, str) else str(item.get("event", item)))
    except Exception as exc:  # noqa: BLE001
        log_error = f"logs unavailable: {type(exc).__name__}"
    tail = [redact(line) for line in log_lines[-args.tail_lines:]]
    return {
        "task_id": task, "state": ti.get("state"), "try_number": ti.get("try_number"),
        "max_tries": ti.get("max_tries"), "start_date": ti.get("start_date"), "end_date": ti.get("end_date"),
        "duration_s": ti.get("duration"), "log_tail": tail, "log_error": log_error,
    }


TOOLS = [
    Tool("list_airflow_dags", "List Airflow DAGs and which are approved for triggering.", NoArgs, list_dags, "airflow"),
    Tool("get_dag_run_status", "Get recent DAG run states for a DAG (default training_pipeline).", DagRunsArgs, get_dag_runs, "airflow"),
    Tool("trigger_training_dag", "Trigger the approved training DAG with optional hyper-parameters.", TriggerArgs, trigger_training_dag, "airflow", writes=True),
    Tool("find_failed_tasks", "Identify failed / upstream_failed tasks of a DAG run (default: latest failed run).", FailedTasksArgs, get_failed_tasks, "airflow"),
    Tool("get_task_diagnostics", "Task-instance details and log tail (secrets redacted) for failure diagnosis.", TaskDiagArgs, get_task_diagnostics, "airflow"),
]
