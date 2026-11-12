import json

from agent.tools.registry import build_registry
from tests.conftest import make_deployment, make_pod, warning_event

reg = build_registry()


def test_list_workloads_and_services(k8s):
    k8s([make_pod("p1"), make_pod("p2")])
    wl = reg.execute("list_k8s_workloads", {}, "r").data
    assert wl["deployments"][0]["desired"] == 2 and len(wl["pods"]) == 2
    svc = reg.execute("list_k8s_services", {}, "r").data["services"][0]
    assert svc["type"] == "ClusterIP" and svc["ready_endpoints"] == 2


def test_reads_outside_approved_namespace_are_denied(k8s):
    k8s([make_pod("p1")])
    res = reg.execute("list_k8s_workloads", {"namespace": "kube-system"}, "r")
    assert res.status == "denied" and "not approved" in res.error


def test_unauthorized_modification_without_confirmation_is_rejected(k8s):
    _, apps = k8s([make_pod("p1")])
    res = reg.execute("restart_deployment", {}, "r")
    assert res.status == "confirmation_required" and apps.patched == []


def test_modification_in_other_namespace_is_denied_even_with_confirmation(k8s):
    _, apps = k8s([make_pod("p1")])
    res = reg.execute("restart_deployment", {"namespace": "production"}, "r", confirmed=True)
    assert res.status == "denied" and apps.patched == []


def test_confirmed_restart_in_approved_namespace_patches_deployment(k8s):
    _, apps = k8s([make_pod("p1")])
    res = reg.execute("restart_deployment", {"deployment": "iris-inference"}, "r", confirmed=True)
    assert res.ok and apps.patched[0][1] == "mlops-agent"
    assert "kubectl.kubernetes.io/restartedAt" in json.dumps(apps.patched[0][2])


def test_diagnose_healthy_deployment(k8s):
    k8s([make_pod("p1"), make_pod("p2")])
    d = reg.execute("diagnose_deployment", {}, "r").data
    assert d["healthy"] is True and not [f for f in d["findings"] if f["severity"] == "high"]


def test_diagnose_image_pull_failure(k8s):
    k8s(
        [make_pod("p1", phase="Pending", ready=False, waiting="ImagePullBackOff")],
        deployment=make_deployment(2, 0, "mlops-inference:does-not-exist"),
        events=[warning_event("Failed", "Failed to pull image mlops-inference:does-not-exist")],
        endpoints_ready=0,
    )
    d = reg.execute("diagnose_deployment", {}, "r").data
    issues = " | ".join(f["issue"] for f in d["findings"])
    assert d["healthy"] is False and "image cannot be pulled" in issues and "0 ready endpoints" in issues


def test_diagnose_readiness_probe_failure_pods_running_but_not_ready(k8s):
    k8s(
        [make_pod("p1", ready=False), make_pod("p2", ready=False)],
        deployment=make_deployment(2, 0),
        events=[warning_event("Unhealthy", "Readiness probe failed: HTTP probe failed with statuscode: 503")],
        logs="ERROR model load failed: FileNotFoundError",
        endpoints_ready=0,
    )
    d = reg.execute("diagnose_deployment", {}, "r").data
    assert any("failing readiness probe" in f["issue"] for f in d["findings"])
    assert any("Readiness probe failed" in f["evidence"] for f in d["findings"])
    assert "model load failed" in " ".join(d["focus_pod_log_tail"])


def test_unhealthy_pods_report_restarts_and_oom(k8s):
    k8s([make_pod("p1", restarts=4, last_term="OOMKilled", exit_code=137), make_pod("p2")])
    bad = reg.execute("find_unhealthy_pods", {}, "r").data["unhealthy_pods"]
    assert len(bad) == 1 and "restarted 4x" in bad[0]["reasons"][0] and "OOMKilled" in bad[0]["reasons"][0]


def test_pod_logs_are_redacted(k8s):
    k8s([make_pod("p1")], logs="starting\nconnecting password=hunter2 token=abc123xyz")
    res = reg.execute("get_pod_details", {"pod_name": "p1"}, "r")
    assert res.ok and "hunter2" not in res.model_dump_json() and "starting" in res.data["log_tail"]


def test_deployed_version_reports_image_tag_and_configmap_values(k8s):
    k8s([make_pod("p1")])
    v = reg.execute("get_deployed_version", {}, "r").data
    assert v["image_tags"] == ["1.0.0"] and v["config_app_version"] == "1.0.0" and v["dataset_version"] == "data-v2"
