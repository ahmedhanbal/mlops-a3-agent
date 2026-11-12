"""Task 2 - MLflow experiment intelligence tools (MlflowClient only, no UI scraping)."""

from __future__ import annotations

import time
from typing import Any

from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.tools.registry import Tool


def _client() -> MlflowClient:
    uri = get_settings().mlflow_tracking_uri
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


class NoArgs(BaseModel):
    pass


class ListRunsArgs(BaseModel):
    experiment_name: str | None = Field(default=None, description="Experiment name (default: all experiments)")
    experiment_id: str | None = None
    max_results: int = Field(default=10, ge=1, le=50)


class RunArgs(BaseModel):
    run_id: str


class BestRunArgs(BaseModel):
    experiment_name: str | None = Field(default=None, description="Experiment name; omit to search all experiments")
    metric: str = Field(default="f1", description="Metric to optimise, e.g. f1")
    mode: str = Field(default="max", pattern="^(max|min)$")
    recent_n: int = Field(default=5, ge=1, le=50, description="Only consider the N most recent finished runs")


class CompareRunsArgs(BaseModel):
    run_ids: list[str] | None = Field(default=None, description="Explicit run ids to compare")
    experiment_name: str | None = None
    last_n: int = Field(default=5, ge=2, le=20, description="If run_ids omitted: compare the last N runs")
    metric: str = "f1"


class CompareExperimentsArgs(BaseModel):
    experiment_names: list[str] = Field(min_length=2, max_length=10)
    metric: str = "f1"


class ModelVersionsArgs(BaseModel):
    model_name: str | None = None


def _iso(ms: int | None) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000)) if ms else None


def _run_summary(r: Any) -> dict[str, Any]:
    return {
        "run_id": r.info.run_id,
        "run_name": r.data.tags.get("mlflow.runName"),
        "experiment_id": r.info.experiment_id,
        "status": r.info.status,
        "started": _iso(r.info.start_time),
        "metrics": {k: round(v, 5) for k, v in r.data.metrics.items()},
        "params": dict(r.data.params),
        "dataset_version": r.data.tags.get("dataset.version"),
        "quality_gate": r.data.tags.get("quality_gate"),
    }


def _exp_ids(client: MlflowClient, name: str | None, exp_id: str | None = None) -> list[str]:
    if exp_id:
        return [exp_id]
    if name:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            raise ValueError(f"experiment '{name}' not found")
        return [exp.experiment_id]
    return [e.experiment_id for e in client.search_experiments()]


def list_experiments(_: NoArgs) -> dict[str, Any]:
    client = _client()
    exps = client.search_experiments()
    return {
        "count": len(exps),
        "experiments": [
            {"experiment_id": e.experiment_id, "name": e.name, "lifecycle_stage": e.lifecycle_stage, "created": _iso(e.creation_time)}
            for e in exps
        ],
    }


def list_runs(args: ListRunsArgs) -> dict[str, Any]:
    client = _client()
    ids = _exp_ids(client, args.experiment_name, args.experiment_id)
    runs = client.search_runs(ids, max_results=args.max_results, order_by=["attributes.start_time DESC"])
    return {"count": len(runs), "runs": [_run_summary(r) for r in runs]}


def get_run_details(args: RunArgs) -> dict[str, Any]:
    client = _client()
    r = client.get_run(args.run_id)
    artifacts = [{"path": a.path, "is_dir": a.is_dir, "size": a.file_size} for a in client.list_artifacts(args.run_id)]
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.") or k == "mlflow.runName"}
    return {**_run_summary(r), "tags": tags, "artifacts": artifacts, "artifact_uri": r.info.artifact_uri}


def _readiness(run: Any, metric: str) -> dict[str, Any]:
    """Evidence checks that a candidate needs beyond a high score. Never claims 'production ready'."""
    s = get_settings()
    tags = run.data.tags
    checks: dict[str, Any] = {
        "run_finished": run.info.status == "FINISHED",
        "dataset_provenance_recorded": bool(tags.get("dataset.md5")),
        "quality_gate": tags.get("quality_gate") or "not recorded",
    }
    caveats = []
    if not checks["run_finished"]:
        caveats.append(f"run status is {run.info.status}, not FINISHED")
    if not checks["dataset_provenance_recorded"]:
        caveats.append("no dataset provenance (dataset.md5 tag) recorded; the run is not reproducible")
    if checks["quality_gate"] != "passed":
        caveats.append(f"quality gate is '{checks['quality_gate']}'; the candidate was not validated by the pipeline gate")
    if metric == "f1":
        value = run.data.metrics.get("f1")
        checks["meets_f1_threshold"] = value is not None and value >= s.f1_quality_gate
        if not checks["meets_f1_threshold"]:
            caveats.append(f"f1 below the configured gate {s.f1_quality_gate}")
    age_days = (time.time() * 1000 - run.info.start_time) / 86_400_000 if run.info.start_time else None
    checks["age_days"] = round(age_days, 1) if age_days is not None else None
    if age_days is not None and age_days > s.freshness_days:
        caveats.append(f"run is {age_days:.0f} days old (freshness policy: {s.freshness_days} days)")
    caveats.append("offline metrics alone do not establish production readiness (no online/drift/latency evidence here)")
    return {"checks": checks, "caveats": caveats}


def best_run(args: BestRunArgs) -> dict[str, Any]:
    client = _client()
    ids = _exp_ids(client, args.experiment_name)
    runs = client.search_runs(
        ids,
        filter_string="attributes.status = 'FINISHED'",
        max_results=args.recent_n,
        order_by=["attributes.start_time DESC"],
    )
    scored = [r for r in runs if args.metric in r.data.metrics]
    if not scored:
        raise ValueError(f"no finished recent runs with metric '{args.metric}'")
    pick = (max if args.mode == "max" else min)(scored, key=lambda r: r.data.metrics[args.metric])
    ranking = sorted(scored, key=lambda r: r.data.metrics[args.metric], reverse=args.mode == "max")
    return {
        "metric": args.metric,
        "mode": args.mode,
        "considered_runs": len(scored),
        "best": _run_summary(pick),
        "ranking": [{"run_id": r.info.run_id, args.metric: round(r.data.metrics[args.metric], 5)} for r in ranking],
        "readiness": _readiness(pick, args.metric),
        "selection_basis": f"{args.mode} {args.metric} among the {len(scored)} most recent finished runs",
    }


def compare_runs(args: CompareRunsArgs) -> dict[str, Any]:
    client = _client()
    if args.run_ids:
        runs = [client.get_run(i) for i in args.run_ids]
    else:
        ids = _exp_ids(client, args.experiment_name)
        runs = client.search_runs(ids, max_results=args.last_n, order_by=["attributes.start_time DESC"])
    rows = [_run_summary(r) for r in runs]
    scored = [r for r in rows if args.metric in r["metrics"]]
    leader = max(scored, key=lambda r: r["metrics"][args.metric])["run_id"] if scored else None
    param_keys = sorted({k for r in rows for k in r["params"]})
    differing = [k for k in param_keys if len({r["params"].get(k) for r in rows}) > 1]
    return {"metric": args.metric, "runs": rows, "leader_by_metric": leader, "params_that_differ": differing}


def compare_experiments(args: CompareExperimentsArgs) -> dict[str, Any]:
    client = _client()
    out = []
    for name in args.experiment_names:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            out.append({"experiment": name, "error": "not found"})
            continue
        runs = client.search_runs([exp.experiment_id], max_results=200)
        vals = [r.data.metrics[args.metric] for r in runs if args.metric in r.data.metrics]
        out.append(
            {
                "experiment": name,
                "runs": len(runs),
                "runs_with_metric": len(vals),
                f"best_{args.metric}": round(max(vals), 5) if vals else None,
                f"mean_{args.metric}": round(sum(vals) / len(vals), 5) if vals else None,
            }
        )
    return {"metric": args.metric, "experiments": out}


def get_model_versions(args: ModelVersionsArgs) -> dict[str, Any]:
    client = _client()
    models = client.search_registered_models()
    if args.model_name:
        models = [m for m in models if m.name == args.model_name]
    result = []
    for m in models:
        versions = client.search_model_versions(f"name = '{m.name.replace(chr(39), '')}'")
        result.append(
            {
                "name": m.name,
                "aliases": dict(getattr(m, "aliases", {}) or {}),
                "versions": [
                    {
                        "version": v.version,
                        "run_id": v.run_id,
                        "status": v.status,
                        "created": _iso(v.creation_timestamp),
                        "aliases": list(getattr(v, "aliases", []) or []),
                    }
                    for v in sorted(versions, key=lambda v: int(v.version), reverse=True)[:10]
                ],
            }
        )
    return {
        "registered_models": result,
        "count": len(result),
        "note": None if result else "registry reachable but no registered models match",
    }


TOOLS = [
    Tool("list_mlflow_experiments", "List MLflow experiments.", NoArgs, list_experiments, "mlflow"),
    Tool(
        "list_mlflow_runs", "List recent MLflow runs with metrics, params and recorded dataset version.", ListRunsArgs, list_runs, "mlflow"
    ),
    Tool("get_mlflow_run_details", "Get full metrics, params, tags and artifacts for one run.", RunArgs, get_run_details, "mlflow"),
    Tool(
        "find_best_run",
        "Identify the best recent run by a metric (default f1) with readiness caveats. Does NOT declare a model production-ready.",
        BestRunArgs,
        best_run,
        "mlflow",
    ),
    Tool(
        "compare_mlflow_runs",
        "Compare several runs (explicit ids, or last N) on a metric; shows which params differ.",
        CompareRunsArgs,
        compare_runs,
        "mlflow",
    ),
    Tool(
        "compare_mlflow_experiments",
        "Compare two or more experiments on a metric (best/mean).",
        CompareExperimentsArgs,
        compare_experiments,
        "mlflow",
    ),
    Tool(
        "get_model_versions",
        "Get MLflow model-registry versions and aliases (if the registry has models).",
        ModelVersionsArgs,
        get_model_versions,
        "mlflow",
    ),
]
