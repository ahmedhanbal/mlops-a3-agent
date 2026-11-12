import hashlib
import importlib

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from agent.main import create_app
from agent.orchestrator import Agent
from tests.conftest import BrokenLLM, stub_registry
from training import pipeline_steps as ps


# --- agent HTTP interface -------------------------------------------------------------------
def client(agent=None):
    return TestClient(create_app(agent or Agent(registry=stub_registry())))


def test_chat_endpoint_returns_evidence_based_response():
    r = client().post("/chat", json={"query": "Which dataset version was used for experiment 25?"})
    body = r.json()
    assert r.status_code == 200 and body["tools_used"] == ["get_experiment_dataset"] and body["evidence"][0]["status"] == "ok"


def test_chat_validates_requests():
    assert client().post("/chat", json={"query": "x"}).status_code == 422
    assert client().post("/chat", json={}).status_code == 422


def test_api_key_is_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "s3cret")
    c = client()
    assert c.post("/chat", json={"query": "list datasets"}).status_code == 401
    assert c.post("/chat", json={"query": "list datasets"}, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_internal_errors_do_not_leak_details():
    class Boom:
        registry = stub_registry()

        def run(self, *a, **k):
            raise RuntimeError("password=hunter2 stack detail")

    r = TestClient(create_app(Boom())).post("/chat", json={"query": "anything here"})
    assert r.status_code == 500 and "hunter2" not in r.text and "request_id" in r.text


def test_tool_catalog_lists_at_least_three_tools_per_module():
    tools = client().get("/tools").json()
    assert len(tools) >= 26
    reg = stub_registry()
    assert all(len(v) >= 3 for v in reg.by_module().values()) and set(reg.by_module()) == {"dvc", "mlflow", "airflow", "kubernetes"}


def test_agent_survives_broken_llm_via_api():
    r = TestClient(create_app(Agent(registry=stub_registry(), llm=BrokenLLM()))).post(
        "/chat", json={"query": "Why is my prediction API unavailable?"}
    )
    assert r.status_code == 200 and r.json()["tools_used"] == ["diagnose_deployment"]


# --- inference service (the app deployed to Kubernetes) ----------------------------------------
@pytest.fixture
def model_file(tmp_path):
    x, y = load_breast_cancer(return_X_y=True)
    path = tmp_path / "m.joblib"
    joblib.dump(RandomForestClassifier(n_estimators=5, random_state=0).fit(x, y), path)
    return path, x[0].tolist()


def test_inference_service_probes_predict_and_version(model_file, monkeypatch):
    path, row = model_file
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    api = importlib.import_module("api.main")
    with TestClient(api.app) as c:
        assert c.get("/health").status_code == 200 and c.get("/ready").status_code == 200
        assert c.post("/predict", json={"features": row}).json()["prediction"] in (0, 1)
        assert c.post("/predict", json={"features": [1.0]}).status_code == 422
        assert c.get("/version").json()["app_version"] == "9.9.9"


def test_bad_model_path_keeps_pod_alive_but_not_ready_injected_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    api = importlib.import_module("api.main")
    with TestClient(api.app) as c:
        assert c.get("/health").status_code == 200  # liveness OK -> no restart loop
        r = c.get("/ready")  # readiness fails -> removed from Service endpoints
        assert r.status_code == 503 and "cannot load model" in r.json()["detail"]


# --- training pipeline logic -------------------------------------------------------------------
@pytest.fixture
def dataset(env):
    df = load_breast_cancer(as_frame=True).frame.rename(columns=lambda c: c.replace(" ", "_"))
    (env / "data").mkdir()
    path = env / "data" / "dataset.csv"
    df.to_csv(path, index=False)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324
    (env / "data" / "dataset.csv.dvc").write_text(f"outs:\n- md5: {md5}\n  size: 1\n  path: dataset.csv\n")
    return path


def test_validation_passes_for_matching_dataset(dataset):
    assert ps.validate_dataset()["rows"] == 569


def test_validation_fails_when_data_changed_without_dvc_add(dataset):
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ps.DataValidationError, match="differs from DVC-recorded"):
        ps.validate_dataset()


def test_validation_fails_on_missing_file_and_bad_schema(env, dataset):
    pd.DataFrame({"a": [1] * 200, "target": [0, 1] * 100}).to_csv(dataset, index=False)
    (env / "data" / "dataset.csv.dvc").unlink()
    with pytest.raises(ps.DataValidationError, match="schema mismatch"):
        ps.validate_dataset()
    dataset.unlink()
    with pytest.raises(ps.DataValidationError, match="not found"):
        ps.validate_dataset()


def run_pipeline(tmp_path, gate):
    import os

    uri = os.environ["MLFLOW_TRACKING_URI"]
    v = ps.validate_dataset()
    s = ps.preprocess("data/dataset.csv", str(tmp_path / "w"))
    m = ps.train_model(s["train"], str(tmp_path / "w"), n_estimators=10, max_depth=3)
    ev = ps.evaluate_and_log(m, s["test"], {"n_estimators": 10}, v["provenance"], uri)
    return uri, ev, lambda: ps.quality_gate(ev["run_id"], gate, uri)


def test_quality_gate_failure_blocks_promotion(dataset, tmp_path):
    from mlflow.tracking import MlflowClient

    uri, ev, gate = run_pipeline(tmp_path, gate=1.01)
    with pytest.raises(ps.QualityGateError):
        gate()
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    assert client.get_run(ev["run_id"]).data.tags["quality_gate"] == "failed"
    assert client.search_registered_models() == []  # nothing registered / promoted


def test_quality_gate_pass_marks_and_registers_candidate(dataset, tmp_path):
    from mlflow.tracking import MlflowClient

    uri, ev, gate = run_pipeline(tmp_path, gate=0.5)
    out = gate()
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    assert client.get_run(ev["run_id"]).data.tags["quality_gate"] == "passed" and out["promoted"] is True


# --- DAG structure ---------------------------------------------------------------------------
def test_dag_defines_required_stages_order_retries_and_no_retry_on_deterministic_failures():
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "dags" / "training_pipeline.py").read_text()
    tree = ast.parse(src)
    tasks = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and any("task" in ast.unparse(d) for d in n.decorator_list)]
    assert tasks == ["validate_dataset", "preprocess", "train_model", "evaluate_and_log", "apply_model_quality_gate"]
    assert '"retries": 2' in src and "AirflowFailException" in src
    assert src.count("@task(retries=0)") == 2  # validation + gate never retried
    wiring = src[src.index("validated = validate_dataset()") :]
    for _a, b in (
        ("validated", "preprocess(validated)"),
        ("prepared", "train_model(prepared)"),
        ("trained", "evaluate_and_log(trained)"),
        ("run", "apply_model_quality_gate(run)"),
    ):
        assert b in wiring
