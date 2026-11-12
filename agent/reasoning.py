"""Deterministic evidence correlation.

The LLM may write prose, but the *structured* answer (facts / uncertainties / recommendations) is derived here
purely from tool results, so operational claims are always traceable to evidence. Retraining is only
recommended when one of three explicit triggers is present: failed quality gate, freshness-policy breach,
or supplied degradation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.config import Settings, get_settings
from agent.schemas import EvidenceItem, ToolResult

MODULE_OF = {
    "dvc": ("dvc", "dataset", "data_changes"),
    "mlflow": ("mlflow", "run", "model_version"),
    "airflow": ("airflow", "dag", "task"),
    "kubernetes": ("k8s", "pod", "deployment", "deployed"),
}


def _module(tool: str) -> str:
    if tool in {
        "list_dvc_datasets",
        "get_dataset_version",
        "compare_dataset_versions",
        "detect_data_changes",
        "explain_dataset_reproduction",
        "get_experiment_dataset",
        "checkout_dataset_version",
    }:
        return "DVC"
    if "mlflow" in tool or tool in {"find_best_run", "get_model_versions"}:
        return "MLflow"
    if "airflow" in tool or "dag" in tool or tool in {"find_failed_tasks", "get_task_diagnostics"}:
        return "Airflow"
    return "Kubernetes"


@dataclass
class Report:
    facts: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    def add(self, bucket: list[str], line: str) -> None:
        if line not in bucket:
            bucket.append(line)

    def to_text(self) -> str:
        def sec(title: str, items: list[str]) -> str:
            return f"**{title}**\n" + "\n".join(f"- {i}" for i in items) if items else ""

        parts = [
            sec("Known facts (from tool evidence)", self.facts),
            sec("Uncertainties", self.uncertainties),
            sec("Recommended actions", self.actions),
        ]
        return "\n\n".join(p for p in parts if p) or "No operational evidence was collected."


def summarize(res: ToolResult) -> str:
    if not res.ok:
        return f"{res.status}: {res.error}"[:200]
    d = res.data
    if not isinstance(d, dict):
        return "ok"
    for key in ("count", "latest_state", "healthy", "changed", "identical"):
        if key in d:
            return f"ok ({key}={d[key]})"
    if "best" in d:
        return f"ok (best run {d['best']['run_id'][:8]})"
    return "ok"


def evidence_items(results: list[ToolResult]) -> list[EvidenceItem]:
    return [
        EvidenceItem(tool=r.tool, status=r.status, request_id=r.request_id, duration_ms=r.duration_ms, summary=summarize(r))
        for r in results
    ]


def _short(x: Any, n: int = 8) -> str:
    return str(x)[:n] if x else "?"


def correlate(
    results: list[ToolResult], query: str = "", context: dict[str, Any] | None = None, settings: Settings | None = None
) -> Report:
    s = settings or get_settings()
    context = context or {}
    rep = Report()
    ok = {r.tool: r for r in results if r.ok}
    best_id: str | None = None
    retrain_reasons: list[str] = []

    for r in results:  # failures first: never claim anything for an unavailable source
        if r.status == "confirmation_required":
            rep.add(rep.pending, r.tool)
            rep.add(rep.actions, f"`{r.tool}` changes state and was NOT executed. Re-submit with confirm=true to authorise it.")
        elif not r.ok:
            rep.add(
                rep.uncertainties,
                f"{_module(r.tool)} evidence unavailable ({r.tool}: {r.status} - {r.error}). Nothing is claimed about it.",
            )

    if r := ok.get("find_best_run"):
        d = r.data
        b = d["best"]
        best_id = b["run_id"]
        rep.add(
            rep.facts,
            f"MLflow: best of the {d['considered_runs']} most recent finished runs by {d['metric']} is run {_short(best_id)} "
            f"({d['metric']}={b['metrics'].get(d['metric'])}, params {b['params']}, started {b['started']}).",
        )
        top = d["ranking"][0][d["metric"]]
        ties = [x["run_id"] for x in d["ranking"] if x[d["metric"]] == top]
        if len(ties) > 1:
            rep.add(
                rep.uncertainties,
                f"MLflow: {len(ties)} runs tie on {d['metric']}={top}; the choice among them rests on recency/other criteria, not on this metric.",
            )
        for c in d["readiness"]["caveats"]:
            rep.add(rep.uncertainties, f"MLflow: {c}")
        if d["readiness"]["checks"].get("quality_gate") == "failed":
            retrain_reasons.append("the best candidate's quality gate is recorded as FAILED")
        age = d["readiness"]["checks"].get("age_days")
        if age is not None and age > s.freshness_days:
            retrain_reasons.append(f"freshness policy exceeded (best run is {age} days old, policy {s.freshness_days} days)")
    if r := ok.get("compare_mlflow_runs"):
        d = r.data
        rep.add(
            rep.facts,
            f"MLflow: compared {len(d['runs'])} runs on {d['metric']}; leader {_short(d['leader_by_metric'])}; params that differ: {d['params_that_differ'] or 'none'}.",
        )
        rep.add(rep.uncertainties, "A higher metric on one split does not by itself make a model production-ready.")
    if r := ok.get("list_mlflow_runs"):
        rep.add(rep.facts, f"MLflow: {r.data['count']} runs listed.")
    if r := ok.get("get_model_versions"):
        rep.add(
            rep.facts, f"MLflow registry: {r.data['count']} registered model(s)." + (f" {r.data['note']}" if r.data.get("note") else "")
        )

    if r := ok.get("get_experiment_dataset"):
        d = r.data
        prov = d["provenance"]
        pick = next((p for p in prov if p["run_id"] == best_id), prov[0]) if prov else None
        if pick and pick["recorded"]:
            rep.add(
                rep.facts,
                f"DVC: run {_short(pick['run_id'])} used dataset version {pick['dataset_version']} (md5 {_short(pick['dataset_md5'])}, "
                f"git commit {_short(pick['dataset_git_commit'], 10)}); commit present in local repo: {pick['commit_exists_in_local_repo']}.",
            )
            if d.get("current_workspace_md5") and pick["dataset_md5"] != d["current_workspace_md5"]:
                rep.add(rep.facts, "DVC: the workspace dataset pointer has changed since this run (newer data version exists).")
        elif pick:
            rep.add(
                rep.uncertainties,
                f"DVC: run {_short(pick['run_id'])} has no recorded dataset provenance; its data version cannot be established.",
            )
        if len(d["distinct_dataset_versions"]) > 1:
            rep.add(rep.facts, f"DVC: inspected runs span dataset versions {d['distinct_dataset_versions']}.")
    if r := ok.get("get_dataset_version"):
        d = r.data
        rep.add(
            rep.facts,
            f"DVC: current version of {d['dataset']} is {d['version']} (md5 {_short(d['dvc_md5'])}, HEAD {_short(d['head_commit'], 10)}).",
        )
    if r := ok.get("detect_data_changes"):
        rep.add(rep.facts, f"DVC: {r.data['interpretation']}.")
        if r.data["changed"]:
            rep.add(
                rep.actions, "Review the data change; run `dvc add` + commit if intended, or `dvc checkout` to restore the tracked version."
            )
    if r := ok.get("compare_dataset_versions"):
        d = r.data
        rep.add(
            rep.facts,
            f"DVC: {d['a']['rev']} vs {d['b']['rev']}: "
            + ("identical content." if d["identical"] else f"different content (size delta {d['size_delta_bytes']} bytes)."),
        )
    if r := ok.get("explain_dataset_reproduction"):
        rep.add(
            rep.facts,
            f"DVC: target version {r.data['target']['version'] or r.data['target']['rev']} resolved to commit {_short(r.data['target']['git_commit'], 10)}.",
        )
        for step in r.data["steps"]:
            rep.add(rep.actions, step)
    if r := ok.get("list_dvc_datasets"):
        rep.add(rep.facts, f"DVC: {r.data['count']} tracked dataset(s): {[x.get('path') for x in r.data['datasets']]}.")

    if r := ok.get("diagnose_deployment"):
        d = r.data
        rep.add(
            rep.facts, f"Kubernetes: deployment {d['deployment']} has {d['ready']}/{d['desired']} replicas ready; healthy={d['healthy']}."
        )
        highs = [f for f in d["findings"] if f["severity"] == "high"]
        for f in highs:
            rep.add(
                rep.facts,
                f"Kubernetes finding: {f['issue']}" + (f" [pod {f['pod']}]" if f.get("pod") else "") + f" - evidence: {f['evidence']}",
            )
        if highs:
            rep.add(rep.actions, "Fix the deployment problem first (infrastructure/config). Retraining would not resolve it.")
    if r := ok.get("get_deployed_version"):
        d = r.data
        rep.add(
            rep.facts,
            f"Kubernetes: deployed image(s) {d['images']}; app version {d['config_app_version'] or d['app_version_label']}; "
            f"model run {d['model_run_id'] or 'not recorded'}; dataset {d['dataset_version'] or 'not recorded'}.",
        )
        mr = d.get("model_run_id")
        if best_id and mr and not mr.startswith("REPLACE"):
            if mr != best_id:
                rep.add(rep.facts, f"Deployed model run ({_short(mr)}) differs from the best candidate ({_short(best_id)}).")
                rep.add(
                    rep.actions, "Consider promoting the better candidate through the normal deploy process (a redeploy, not a retrain)."
                )
            else:
                rep.add(rep.facts, "The deployed model is the best recent candidate.")
        elif best_id:
            rep.add(
                rep.uncertainties,
                "Kubernetes: ConfigMap does not record which MLflow run is deployed, so deployed-vs-best cannot be compared.",
            )
    if r := ok.get("find_unhealthy_pods"):
        n = len(r.data["unhealthy_pods"])
        rep.add(
            rep.facts,
            f"Kubernetes: {n} unhealthy pod(s)." + "".join(f" {u['pod']}: {'; '.join(u['reasons'])}." for u in r.data["unhealthy_pods"]),
        )
    if r := ok.get("list_k8s_workloads"):
        rep.add(
            rep.facts,
            "Kubernetes: "
            + "; ".join(f"{x['name']} {x['ready']}/{x['desired']} ready" for x in r.data["deployments"])
            + f"; {len(r.data['pods'])} pod(s).",
        )
    if r := ok.get("list_k8s_services"):
        rep.add(
            rep.facts,
            "Kubernetes services: "
            + "; ".join(f"{x['name']} ({x['type']}, ready endpoints {x['ready_endpoints']})" for x in r.data["services"]),
        )
    if r := ok.get("get_pod_details"):
        rep.add(
            rep.facts, f"Kubernetes: pod {r.data['name']} phase {r.data['phase']}, ready={r.data['ready']}, restarts={r.data['restarts']}."
        )
    if r := ok.get("restart_deployment"):
        rep.add(rep.facts, f"Kubernetes: rollout restart issued for {r.data['restarted']} in {r.data['namespace']}.")

    if r := ok.get("get_dag_run_status"):
        d = r.data
        runs = ", ".join(f"{x['dag_run_id'][:28]}={x['state']}" for x in d["runs"][:3]) or "no runs"
        rep.add(rep.facts, f"Airflow: latest {d['dag_id']} state={d['latest_state']}; recent: {runs}.")
        if d["latest_state"] == "failed":
            rep.add(rep.actions, "Inspect the failed Airflow run (find_failed_tasks / get_task_diagnostics) before deciding anything.")
    if r := ok.get("find_failed_tasks"):
        d = r.data
        if d["failed_tasks"]:
            rep.add(
                rep.facts,
                f"Airflow: run {d['run_id']} failed at {d['root_cause_candidates']} (downstream: {[t['task_id'] for t in d['failed_tasks'] if t['state'] == 'upstream_failed']}).",
            )
            if any("quality_gate" in t for t in d["root_cause_candidates"]):
                retrain_reasons.append("the latest Airflow run failed at the model quality gate")
        else:
            rep.add(rep.facts, d.get("note") or "Airflow: no failed tasks in the inspected run.")
    if r := ok.get("get_task_diagnostics"):
        rep.add(rep.facts, f"Airflow: task {r.data['task_id']} state={r.data['state']} after try {r.data['try_number']}.")
    if r := ok.get("trigger_training_dag"):
        rep.add(rep.facts, f"Airflow: DAG run {r.data['dag_run_id']} created (state {r.data['state']}).")
    if r := ok.get("list_airflow_dags"):
        rep.add(rep.facts, f"Airflow: {r.data['count']} DAG(s); approved for triggering: {r.data['approved_for_triggering']}.")

    deg = context.get("degradation")
    if isinstance(deg, dict):
        val, metric = deg.get("value"), deg.get("metric", "f1")
        if deg.get("degraded") is True or (metric == "f1" and isinstance(val, (int, float)) and val < s.f1_quality_gate):
            retrain_reasons.append(f"supplied degradation results ({metric}={val})")

    wants = "retrain" in query.lower() or "find_best_run" in ok
    if wants and (ok.get("find_best_run") or "retrain" in query.lower() or deg):
        if retrain_reasons:
            rep.add(rep.facts, "Retraining assessment: justified because " + "; ".join(retrain_reasons) + ".")
            rep.add(
                rep.actions,
                "Retraining is recommended but NOT started automatically. Ask explicitly for the training DAG trigger and confirm it.",
            )
        elif "find_best_run" in ok:
            missing = [e.tool for e in results if not e.ok and e.status != "confirmation_required"]
            if missing:
                rep.add(
                    rep.facts,
                    "Retraining assessment: no retraining trigger found in the evidence that WAS collected (no failed quality gate, "
                    f"freshness policy of {s.freshness_days} days not exceeded, no degradation results supplied).",
                )
                rep.add(
                    rep.uncertainties,
                    f"The assessment is incomplete because evidence from {sorted(set(missing))} was unavailable; a failed pipeline run or unhealthy deployment cannot be excluded.",
                )
            else:
                rep.add(
                    rep.facts,
                    f"Retraining assessment: not necessary on current evidence - no failed quality gate, freshness policy ({s.freshness_days} days) not exceeded, and no degradation results supplied.",
                )
            rep.add(
                rep.uncertainties, "No online performance/drift monitoring data is available; supply degradation results if you have them."
            )
        else:
            rep.add(
                rep.uncertainties,
                "Retraining need cannot be determined without MLflow evidence (quality gate, age) or supplied degradation results.",
            )
    return rep


def safe_correlate(
    results: list[ToolResult], query: str = "", context: dict[str, Any] | None = None, settings: Settings | None = None
) -> Report:
    """Correlate evidence; if a tool returned an unexpected payload shape, degrade per tool instead of failing the request."""
    try:
        return correlate(results, query, context, settings)
    except Exception:  # noqa: BLE001
        merged = Report()
        for r in results:
            try:
                part = correlate([r], query, context, settings)
            except Exception:  # noqa: BLE001
                merged.add(merged.uncertainties, f"{_module(r.tool)}: result of {r.tool} had an unexpected format and was not interpreted.")
                continue
            for bucket, items in (
                (merged.facts, part.facts),
                (merged.uncertainties, part.uncertainties),
                (merged.actions, part.actions),
                (merged.pending, part.pending),
            ):
                for i in items:
                    merged.add(bucket, i)
        return merged
