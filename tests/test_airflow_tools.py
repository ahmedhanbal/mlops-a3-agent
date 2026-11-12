import json

import pytest
import responses

from agent.tools.registry import build_registry

BASE = "http://airflow.test"
reg = build_registry()


def token():
    responses.add(responses.POST, f"{BASE}/auth/token", json={"access_token": "jwt-abc.def.ghi-secret"}, status=201)


@responses.activate
def test_authenticates_with_jwt_and_lists_dags():
    token()
    responses.add(
        responses.GET,
        f"{BASE}/api/v2/dags",
        json={"total_entries": 1, "dags": [{"dag_id": "training_pipeline", "is_paused": False, "tags": [{"name": "mlops"}]}]},
    )
    res = reg.execute("list_airflow_dags", {}, "r")
    assert res.ok and res.data["dags"][0]["dag_id"] == "training_pipeline"
    assert responses.calls[1].request.headers["Authorization"] == "Bearer jwt-abc.def.ghi-secret"
    assert "jwt-abc" not in res.model_dump_json()


@responses.activate
def test_dag_run_status():
    token()
    responses.add(
        responses.GET,
        f"{BASE}/api/v2/dags/training_pipeline/dagRuns",
        json={"dag_runs": [{"dag_run_id": "manual__1", "state": "success"}, {"dag_run_id": "manual__0", "state": "failed"}]},
    )
    res = reg.execute("get_dag_run_status", {}, "r")
    assert res.data["latest_state"] == "success" and len(res.data["runs"]) == 2


@responses.activate
def test_trigger_requires_confirmation_and_makes_no_http_call_without_it():
    res = reg.execute("trigger_training_dag", {}, "r")
    assert res.status == "confirmation_required" and len(responses.calls) == 0


@responses.activate
def test_trigger_of_unapproved_dag_is_denied_even_when_confirmed():
    res = reg.execute("trigger_training_dag", {"dag_id": "some_other_dag"}, "r", confirmed=True)
    assert res.status == "denied" and len(responses.calls) == 0


@responses.activate
def test_confirmed_trigger_posts_conf():
    token()
    responses.add(
        responses.POST, f"{BASE}/api/v2/dags/training_pipeline/dagRuns", json={"dag_run_id": "manual__9", "state": "queued"}, status=200
    )
    res = reg.execute("trigger_training_dag", {"n_estimators": 120}, "r", confirmed=True)
    body = json.loads(responses.calls[1].request.body)
    assert res.ok and res.data["dag_run_id"] == "manual__9" and body["conf"] == {"n_estimators": 120}


@responses.activate
def test_failed_task_identification_finds_root_cause_and_downstream():
    token()
    responses.add(
        responses.GET, f"{BASE}/api/v2/dags/training_pipeline/dagRuns", json={"dag_runs": [{"dag_run_id": "run1", "state": "failed"}]}
    )
    responses.add(
        responses.GET,
        f"{BASE}/api/v2/dags/training_pipeline/dagRuns/run1/taskInstances",
        json={
            "task_instances": [
                {"task_id": "validate_dataset", "state": "success", "try_number": 1},
                {"task_id": "preprocess", "state": "success", "try_number": 1},
                {"task_id": "train_model", "state": "failed", "try_number": 3},
                {"task_id": "evaluate_and_log", "state": "upstream_failed", "try_number": 0},
            ]
        },
    )
    res = reg.execute("find_failed_tasks", {}, "r")
    assert res.data["root_cause_candidates"] == ["train_model"] and len(res.data["failed_tasks"]) == 2


@responses.activate
def test_task_diagnostics_redacts_secrets_in_logs():
    token()
    responses.add(
        responses.GET,
        f"{BASE}/api/v2/dags/training_pipeline/dagRuns/run1/taskInstances/train_model",
        json={"state": "failed", "try_number": 3, "max_tries": 2},
    )
    responses.add(
        responses.GET,
        f"{BASE}/api/v2/dags/training_pipeline/dagRuns/run1/taskInstances/train_model/logs/1",
        json={"content": ["connecting with password=hunter2", "ValueError: bad shape"]},
    )
    res = reg.execute("get_task_diagnostics", {"run_id": "run1", "task_id": "train_model"}, "r")
    assert res.ok and "hunter2" not in json.dumps(res.data) and "ValueError: bad shape" in res.data["log_tail"]


@responses.activate
def test_expired_token_is_refreshed_once():
    token()
    responses.add(responses.GET, f"{BASE}/api/v2/dags", status=401)
    responses.add(responses.GET, f"{BASE}/api/v2/dags", json={"total_entries": 0, "dags": []})
    assert reg.execute("list_airflow_dags", {}, "r").ok


@responses.activate
def test_unreachable_airflow_is_a_clean_error():
    res = reg.execute("get_dag_run_status", {}, "r")  # nothing registered -> connection error
    assert res.status == "error" and res.data is None


@responses.activate
def test_http_failure_surfaces_status_code():
    token()
    responses.add(responses.GET, f"{BASE}/api/v2/dags", status=500, body="boom")
    res = reg.execute("list_airflow_dags", {}, "r")
    assert res.status == "error" and "HTTP 500" in res.error


@pytest.mark.parametrize("dag", ["../x", "a b", "dag;drop"])
def test_dag_id_injection_rejected(dag):
    assert reg.execute("get_dag_run_status", {"dag_id": dag}, "r").status == "denied"
