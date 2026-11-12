from agent.tools.registry import build_registry

reg = build_registry()


def test_list_experiments_and_runs(mlflow_runs):
    assert any(e["name"] == "exp-a" for e in reg.execute("list_mlflow_experiments", {}, "r").data["experiments"])
    runs = reg.execute("list_mlflow_runs", {"experiment_name": "exp-a"}, "r").data["runs"]
    assert [r["run_name"] for r in runs] == ["run-new-noprov", "run-best", "run-old"]  # newest first


def test_best_run_by_f1_among_recent_runs(mlflow_runs):
    res = reg.execute("find_best_run", {"experiment_name": "exp-a", "metric": "f1", "recent_n": 3}, "r")
    assert res.ok and res.data["best"]["run_id"] == mlflow_runs["run-best"]
    assert [x["f1"] for x in res.data["ranking"]] == [0.96, 0.94, 0.9]


def test_best_run_respects_recent_window(mlflow_runs):
    res = reg.execute("find_best_run", {"experiment_name": "exp-a", "recent_n": 1}, "r")
    assert res.data["best"]["run_name"] == "run-new-noprov"


def test_highest_score_is_not_declared_production_ready(mlflow_runs):
    res = reg.execute("find_best_run", {"experiment_name": "exp-a", "recent_n": 1}, "r")
    caveats = " ".join(res.data["readiness"]["caveats"])
    assert "production readiness" in caveats
    assert "no dataset provenance" in caveats  # missing provenance is surfaced
    assert "production_ready" not in res.data["readiness"]["checks"]


def test_compare_runs_shows_leader_and_differing_params(mlflow_runs):
    res = reg.execute("compare_mlflow_runs", {"experiment_name": "exp-a", "last_n": 3, "metric": "f1"}, "r")
    assert res.data["leader_by_metric"] == mlflow_runs["run-best"] and "n_estimators" in res.data["params_that_differ"]


def test_compare_experiments(mlflow_runs):
    res = reg.execute("compare_mlflow_experiments", {"experiment_names": ["exp-a", "does-not-exist"]}, "r")
    assert res.data["experiments"][0]["best_f1"] == 0.96 and res.data["experiments"][1]["error"] == "not found"


def test_run_details_include_params_metrics_artifacts(mlflow_runs):
    res = reg.execute("get_mlflow_run_details", {"run_id": mlflow_runs["run-best"]}, "r")
    assert res.ok and res.data["metrics"]["f1"] == 0.96 and "artifacts" in res.data


def test_model_registry_empty_is_reported_honestly(mlflow_runs):
    res = reg.execute("get_model_versions", {}, "r")
    assert res.ok and res.data["count"] == 0 and "no registered models" in res.data["note"]


def test_unknown_experiment_is_an_error_not_a_guess(mlflow_runs):
    assert reg.execute("list_mlflow_runs", {"experiment_name": "nope"}, "r").status == "error"
