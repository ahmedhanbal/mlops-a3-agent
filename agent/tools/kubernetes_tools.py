"""Task 4 - Kubernetes operations tools.

Safety model
  * every tool (read or write) is confined to ONE approved namespace (K8S_APPROVED_NAMESPACE);
  * state-changing tools additionally need explicit confirmation (enforced by the registry);
  * pod logs are redacted before being returned.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.logging_utils import redact
from agent.tools.registry import PolicyDenied, Tool


# --- client factories (patched in unit tests) --------------------------------------------
def _load() -> None:
    from kubernetes import config

    try:
        config.load_incluster_config()
    except Exception:  # noqa: BLE001
        config.load_kube_config()


def _core() -> Any:
    from kubernetes import client

    _load()
    return client.CoreV1Api()


def _apps() -> Any:
    from kubernetes import client

    _load()
    return client.AppsV1Api()


def ensure_namespace(namespace: str | None) -> str:
    approved = get_settings().k8s_namespace
    ns = namespace or approved
    if ns != approved:
        raise PolicyDenied(f"namespace '{ns}' is not approved; this agent may only operate in '{approved}'")
    return ns


# --- inputs ------------------------------------------------------------------------------
class NsArgs(BaseModel):
    namespace: str | None = Field(default=None, description="Must equal the approved namespace (default)")


class PodArgs(NsArgs):
    pod_name: str
    tail_lines: int = Field(default=40, ge=1, le=300)


class DeployArgs(NsArgs):
    deployment: str = Field(default="iris-inference")


class RestartArgs(DeployArgs):
    pass


# --- converters --------------------------------------------------------------------------
def _age(ts: Any) -> str | None:
    if not ts:
        return None
    delta = dt.datetime.now(dt.timezone.utc) - ts
    return f"{int(delta.total_seconds())}s"


def _container_states(pod: Any) -> list[dict[str, Any]]:
    out = []
    for cs in pod.status.container_statuses or []:
        st: dict[str, Any] = {"name": cs.name, "ready": cs.ready, "restart_count": cs.restart_count, "image": cs.image}
        if cs.state and cs.state.waiting:
            st["waiting_reason"] = cs.state.waiting.reason
            st["waiting_message"] = (cs.state.waiting.message or "")[:200]
        if cs.state and cs.state.terminated:
            st["terminated_reason"] = cs.state.terminated.reason
            st["exit_code"] = cs.state.terminated.exit_code
        lt = cs.last_state.terminated if cs.last_state else None
        if lt:
            st["last_terminated_reason"] = lt.reason
            st["last_exit_code"] = lt.exit_code
        out.append(st)
    return out


def _pod_view(pod: Any) -> dict[str, Any]:
    conds = {c.type: c.status for c in (pod.status.conditions or [])}
    containers = _container_states(pod)
    return {
        "name": pod.metadata.name,
        "phase": pod.status.phase,
        "ready": conds.get("Ready") == "True",
        "restarts": sum(c["restart_count"] for c in containers),
        "containers": containers,
        "node": pod.spec.node_name,
        "age": _age(pod.metadata.creation_timestamp),
        "labels": dict(pod.metadata.labels or {}),
    }


def _events(core: Any, ns: str, field_selector: str | None = None, limit: int = 15) -> list[dict[str, Any]]:
    evs = core.list_namespaced_event(ns, field_selector=field_selector).items
    evs = sorted(
        evs,
        key=lambda e: e.last_timestamp or e.event_time or e.metadata.creation_timestamp or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
        reverse=True,
    )
    return [
        {"type": e.type, "reason": e.reason, "object": e.involved_object.name, "message": (e.message or "")[:240], "count": e.count}
        for e in evs[:limit]
    ]


def _app_pods(core: Any, ns: str) -> list[Any]:
    return core.list_namespaced_pod(ns, label_selector=get_settings().k8s_app_label).items


# --- tools -------------------------------------------------------------------------------
def list_workloads(args: NsArgs) -> dict[str, Any]:
    ns = ensure_namespace(args.namespace)
    deployments = _apps().list_namespaced_deployment(ns).items
    pods = _app_pods(_core(), ns)
    return {
        "namespace": ns,
        "deployments": [
            {
                "name": d.metadata.name,
                "desired": d.spec.replicas,
                "ready": d.status.ready_replicas or 0,
                "available": d.status.available_replicas or 0,
                "images": [c.image for c in d.spec.template.spec.containers],
            }
            for d in deployments
        ],
        "pods": [_pod_view(p) for p in pods],
    }


def get_pod_details(args: PodArgs) -> dict[str, Any]:
    ns = ensure_namespace(args.namespace)
    core = _core()
    pod = core.read_namespaced_pod(args.pod_name, ns)
    try:
        raw = core.read_namespaced_pod_log(args.pod_name, ns, tail_lines=args.tail_lines)
        logs = [redact(line) for line in raw.splitlines()[-args.tail_lines :]]
        log_error = None
    except Exception as exc:  # noqa: BLE001 - e.g. container never started
        logs, log_error = [], f"logs unavailable: {type(exc).__name__}"
    probes = {c.name: {"readiness": bool(c.readiness_probe), "liveness": bool(c.liveness_probe)} for c in pod.spec.containers}
    return {
        **_pod_view(pod),
        "probes_configured": probes,
        "log_tail": logs,
        "log_error": log_error,
        "events": _events(core, ns, f"involvedObject.name={args.pod_name}"),
    }


def list_services(args: NsArgs) -> dict[str, Any]:
    ns = ensure_namespace(args.namespace)
    core = _core()
    out = []
    for svc in core.list_namespaced_service(ns).items:
        try:
            ep = core.read_namespaced_endpoints(svc.metadata.name, ns)
            ready = sum(len(s.addresses or []) for s in (ep.subsets or []))
            not_ready = sum(len(s.not_ready_addresses or []) for s in (ep.subsets or []))
        except Exception:  # noqa: BLE001
            ready = not_ready = None
        out.append(
            {
                "name": svc.metadata.name,
                "type": svc.spec.type,
                "cluster_ip": svc.spec.cluster_ip,
                "ports": [{"port": p.port, "target_port": str(p.target_port)} for p in (svc.spec.ports or [])],
                "selector": svc.spec.selector,
                "ready_endpoints": ready,
                "not_ready_endpoints": not_ready,
            }
        )
    return {"namespace": ns, "services": out}


def _unhealthy(core: Any, ns: str) -> list[dict[str, Any]]:
    bad = []
    for pod in _app_pods(core, ns):
        v = _pod_view(pod)
        reasons = []
        for c in v["containers"]:
            if c.get("waiting_reason"):
                reasons.append(f"{c['name']} waiting: {c['waiting_reason']}")
            if c["restart_count"]:
                reasons.append(
                    f"{c['name']} restarted {c['restart_count']}x (last: {c.get('last_terminated_reason', 'n/a')}, exit {c.get('last_exit_code', 'n/a')})"
                )
        if v["phase"] not in ("Running", "Succeeded"):
            reasons.append(f"phase {v['phase']}")
        if v["phase"] == "Running" and not v["ready"]:
            reasons.append("Running but NOT Ready (readiness probe failing?)")
        if reasons:
            bad.append({"pod": v["name"], "reasons": reasons})
    return bad


def find_unhealthy_pods(args: NsArgs) -> dict[str, Any]:
    ns = ensure_namespace(args.namespace)
    core = _core()
    warn = [e for e in _events(core, ns, limit=40) if e["type"] == "Warning"]
    return {"namespace": ns, "unhealthy_pods": _unhealthy(core, ns), "warning_events": warn[:10]}


def get_deployed_version(args: DeployArgs) -> dict[str, Any]:
    ns = ensure_namespace(args.namespace)
    d = _apps().read_namespaced_deployment(args.deployment, ns)
    images = [c.image for c in d.spec.template.spec.containers]
    labels = dict(d.spec.template.metadata.labels or {})
    cm_data: dict[str, str] = {}
    for c in d.spec.template.spec.containers:
        for ref in c.env_from or []:
            if ref.config_map_ref:
                cm_data.update(_core().read_namespaced_config_map(ref.config_map_ref.name, ns).data or {})
    return {
        "deployment": args.deployment,
        "namespace": ns,
        "images": images,
        "image_tags": [i.rsplit(":", 1)[1] if ":" in i.split("/")[-1] else "latest(implicit)" for i in images],
        "app_version_label": labels.get("app.kubernetes.io/version"),
        "config_app_version": cm_data.get("APP_VERSION"),
        "model_run_id": cm_data.get("MODEL_RUN_ID"),
        "dataset_version": cm_data.get("DATASET_VERSION"),
        "generation": d.metadata.generation,
        "observed_generation": d.status.observed_generation,
    }


def diagnose_deployment(args: DeployArgs) -> dict[str, Any]:
    """Evidence-based diagnosis: replicas, pod/container states, probes, events, logs."""
    ns = ensure_namespace(args.namespace)
    d = _apps().read_namespaced_deployment(args.deployment, ns)
    core = _core()
    desired, ready = d.spec.replicas or 0, d.status.ready_replicas or 0
    pods = [_pod_view(p) for p in _app_pods(core, ns)]
    findings: list[dict[str, Any]] = []
    if ready < desired:
        findings.append({"severity": "high", "issue": f"only {ready}/{desired} replicas ready", "evidence": "deployment status"})
    for p in pods:
        for c in p["containers"]:
            wr = c.get("waiting_reason")
            if wr in ("ImagePullBackOff", "ErrImagePull"):
                findings.append(
                    {
                        "severity": "high",
                        "issue": "image cannot be pulled (wrong tag/registry/credentials)",
                        "pod": p["name"],
                        "evidence": f"{wr}: {c.get('waiting_message')}",
                    }
                )
            elif wr == "CrashLoopBackOff":
                findings.append(
                    {
                        "severity": "high",
                        "issue": "container crashes on start",
                        "pod": p["name"],
                        "evidence": f"CrashLoopBackOff, last exit {c.get('last_exit_code')}, {c.get('last_terminated_reason')}",
                    }
                )
            if c.get("last_terminated_reason") == "OOMKilled":
                findings.append(
                    {
                        "severity": "high",
                        "issue": "container OOMKilled - memory limit too low",
                        "pod": p["name"],
                        "evidence": "last terminated reason OOMKilled",
                    }
                )
        if p["phase"] == "Running" and not p["ready"] and not any(c.get("waiting_reason") for c in p["containers"]):
            findings.append(
                {
                    "severity": "high",
                    "issue": "pod running but failing readiness probe",
                    "pod": p["name"],
                    "evidence": "Ready=False while phase=Running",
                }
            )
        if p["phase"] == "Pending":
            findings.append(
                {
                    "severity": "high",
                    "issue": "pod Pending (unschedulable or waiting for resources)",
                    "pod": p["name"],
                    "evidence": "phase Pending",
                }
            )
    warn = [e for e in _events(core, ns, limit=40) if e["type"] == "Warning"]
    for e in warn[:5]:
        findings.append({"severity": "info", "issue": f"warning event {e['reason']}", "evidence": e["message"]})
    # log tail of the least healthy pod
    log_tail: list[str] = []
    focus = next((p for p in pods if not p["ready"]), pods[0] if pods else None)
    if focus:
        try:
            log_tail = [redact(x) for x in core.read_namespaced_pod_log(focus["name"], ns, tail_lines=25).splitlines()[-25:]]
        except Exception:  # noqa: BLE001
            log_tail = []
    svc_ready = None
    try:
        for s in core.list_namespaced_service(ns).items:
            ep = core.read_namespaced_endpoints(s.metadata.name, ns)
            svc_ready = sum(len(x.addresses or []) for x in (ep.subsets or []))
            if svc_ready == 0:
                findings.append(
                    {
                        "severity": "high",
                        "issue": f"service {s.metadata.name} has 0 ready endpoints (API unreachable)",
                        "evidence": "endpoints empty",
                    }
                )
    except Exception:  # noqa: BLE001
        pass
    healthy = ready >= desired > 0 and not any(f["severity"] == "high" for f in findings)
    return {
        "deployment": args.deployment,
        "namespace": ns,
        "desired": desired,
        "ready": ready,
        "healthy": healthy,
        "pods": pods,
        "findings": findings,
        "focus_pod_log_tail": log_tail,
    }


def restart_deployment(args: RestartArgs) -> dict[str, Any]:
    """STATE-CHANGING rollout restart (patches pod-template annotation). Namespace-restricted."""
    ns = ensure_namespace(args.namespace)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": stamp}}}}}
    _apps().patch_namespaced_deployment(args.deployment, ns, body)
    return {"restarted": args.deployment, "namespace": ns, "restartedAt": stamp}


TOOLS = [
    Tool(
        "list_k8s_workloads",
        "List application Deployments and Pods (status, restarts, images) in the approved namespace.",
        NsArgs,
        list_workloads,
        "kubernetes",
    ),
    Tool(
        "get_pod_details",
        "Pod status, container states, probes, recent events and log tail (redacted).",
        PodArgs,
        get_pod_details,
        "kubernetes",
    ),
    Tool("list_k8s_services", "Inspect Services and their ready endpoints.", NsArgs, list_services, "kubernetes"),
    Tool("find_unhealthy_pods", "Identify unhealthy pods, restart events and warning events.", NsArgs, find_unhealthy_pods, "kubernetes"),
    Tool(
        "get_deployed_version",
        "Report the deployed application version (image tag, version label, ConfigMap model/dataset info).",
        DeployArgs,
        get_deployed_version,
        "kubernetes",
    ),
    Tool(
        "diagnose_deployment",
        "Evidence-based diagnosis of why the prediction API is unavailable/unhealthy (pods, probes, events, logs, endpoints).",
        DeployArgs,
        diagnose_deployment,
        "kubernetes",
    ),
    Tool(
        "restart_deployment",
        "Rollout-restart a Deployment in the approved namespace.",
        RestartArgs,
        restart_deployment,
        "kubernetes",
        writes=True,
    ),
]
