import shutil
import subprocess
import time

import pytest
from mlflow.tracking import MlflowClient

from agent.tools import dvc_tools
from agent.tools.registry import PolicyDenied, build_registry

needs_dvc = pytest.mark.skipif(not (shutil.which("dvc") and shutil.which("git")), reason="dvc/git not installed")


def test_only_allow_listed_commands_can_run():
    for bad in (["rm", "-rf", "/"], ["dvc", "push"], ["git", "push"], ["dvc", "remote", "add", "x", "y"]):
        with pytest.raises(PolicyDenied):
            dvc_tools.run_command(bad)


@pytest.mark.parametrize("rev", ["--force", "a b", "x;rm", "../etc", "-h", "a..b", ""])
def test_malicious_revisions_are_denied(rev):
    res = build_registry().execute("explain_dataset_reproduction", {"rev": rev}, "r")
    assert res.status in ("denied", "invalid_request", "error") and res.status != "ok"


@pytest.fixture
def dvc_repo(env, monkeypatch):
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(k, v)

    def sh(*a):
        subprocess.run(a, cwd=env, check=True, capture_output=True)

    sh("git", "init", "-q", "-b", "main")
    sh("dvc", "init", "-q")
    (env / "data").mkdir()
    data = env / "data" / "dataset.csv"
    for ver, content in (("v1", "a,b\n1,2\n"), ("v2", "a,b\n1,2\n3,4\n")):
        data.write_text(content)
        sh("dvc", "add", "-q", "data/dataset.csv")
        sh("git", "add", "-A")
        sh("git", "commit", "-q", "-m", f"data {ver}")
        sh("git", "tag", f"data-{ver}")
    return env


@needs_dvc
@pytest.mark.integration
def test_list_version_and_compare_with_real_dvc(dvc_repo):
    reg = build_registry()
    listed = reg.execute("list_dvc_datasets", {}, "r")
    assert listed.ok and listed.data["datasets"][0]["path"] == "dataset.csv"
    ver = reg.execute("get_dataset_version", {}, "r")
    assert ver.ok and ver.data["version"] == "data-v2" and len(ver.data["head_commit"]) == 40
    cmp = reg.execute("compare_dataset_versions", {"rev_a": "data-v1", "rev_b": "data-v2"}, "r")
    assert cmp.ok and cmp.data["identical"] is False and cmp.data["size_delta_bytes"] > 0


@needs_dvc
@pytest.mark.integration
def test_detect_changes_and_reproduce_older_version(dvc_repo):
    reg = build_registry()
    assert reg.execute("detect_data_changes", {}, "r").data["changed"] is False
    (dvc_repo / "data" / "dataset.csv").write_text("tampered\n")
    assert reg.execute("detect_data_changes", {}, "r").data["changed"] is True

    steps = reg.execute("explain_dataset_reproduction", {"rev": "data-v1"}, "r")
    assert steps.ok and any("dvc checkout" in s for s in steps.data["steps"])

    assert reg.execute("checkout_dataset_version", {"rev": "data-v1"}, "r").status == "confirmation_required"
    blocked = reg.execute("checkout_dataset_version", {"rev": "data-v1"}, "r", confirmed=True)
    assert blocked.status == "error" and "unsaved changes" in blocked.error  # never clobbers modified data
    (dvc_repo / "data" / "dataset.csv").write_text("a,b\n1,2\n3,4\n")  # restore the tracked v2 content
    done = reg.execute("checkout_dataset_version", {"rev": "data-v1"}, "r", confirmed=True)
    assert done.ok and done.data["workspace_matches_version"] is True
    assert (dvc_repo / "data" / "dataset.csv").read_text() == "a,b\n1,2\n"


def test_experiment_dataset_provenance_comes_from_recorded_tags(mlflow_runs, env):
    res = build_registry().execute("get_experiment_dataset", {"run_id": mlflow_runs["run-best"]}, "r")
    assert res.ok
    p = res.data["provenance"][0]
    assert p["dataset_version"] == "data-v2" and p["recorded"] is True


def test_experiment_without_recorded_provenance_is_flagged_not_guessed(mlflow_runs):
    res = build_registry().execute("get_experiment_dataset", {"run_id": mlflow_runs["run-new-noprov"]}, "r")
    assert res.ok and res.data["provenance"][0]["recorded"] is False and res.data["provenance"][0]["dataset_version"] is None


def test_experiment_lookup_by_id_uses_latest_runs(mlflow_runs, env):
    import os

    exp = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"]).get_experiment_by_name("exp-a")
    res = build_registry().execute("get_experiment_dataset", {"experiment_id": exp.experiment_id}, "r")
    assert res.ok and res.data["runs_inspected"] == 3 and set(res.data["distinct_dataset_versions"]) >= {"data-v1", "data-v2"}
    time.sleep(0)
