"""Task 1 - DVC data management tools.

Only an explicit allow-list of git/dvc sub-commands can be executed (no shell, no free-form args).
Dataset provenance is *read from recorded MLflow tags*, never inferred by the LLM.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.tools.registry import PolicyDenied, Tool

# --- command allow-list ------------------------------------------------------------------
ALLOWED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("dvc", "status"),
    ("dvc", "diff"),
    ("dvc", "checkout"),
    ("dvc", "pull"),
    ("dvc", "remote", "list"),
    ("git", "rev-parse"),
    ("git", "log"),
    ("git", "tag"),
    ("git", "show"),
    ("git", "cat-file"),
    ("git", "checkout"),
)
_REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,63}$")  # no leading '-', no spaces, no shell chars
_PATH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./\-]{0,127}$")


def _check_rev(rev: str) -> str:
    if not _REV_RE.match(rev) or ".." in rev:
        raise PolicyDenied(f"invalid revision '{rev}'")
    return rev


def _check_path(path: str) -> str:
    if not _PATH_RE.match(path) or ".." in path:
        raise PolicyDenied(f"invalid path '{path}'")
    return path


def run_command(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """Run an approved command without a shell. Raises PolicyDenied for anything else."""
    if not any(tuple(args[: len(a)]) == a for a in ALLOWED_COMMANDS):
        raise PolicyDenied(f"command not on the approved list: {' '.join(args[:2])}")
    return subprocess.run(  # noqa: S603 - allow-listed, shell=False
        args,
        cwd=get_settings().repo_root,
        capture_output=True,
        text=True,
        timeout=timeout or get_settings().http_timeout_s * 3,
        check=False,
    )


def _git_ok(args: list[str]) -> str:
    proc = run_command(["git", *args])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


# --- helpers -----------------------------------------------------------------------------
def _parse_dvc_text(text: str) -> dict[str, Any]:
    doc = yaml.safe_load(text) or {}
    outs = doc.get("outs") or []
    if not outs:
        raise ValueError("no outs in .dvc file")
    out = outs[0]
    return {"path": out.get("path"), "md5": out.get("md5"), "size": out.get("size"), "nfiles": out.get("nfiles")}


def _dvc_file_for(dataset: str) -> str:
    dataset = _check_path(dataset)
    return dataset if dataset.endswith(".dvc") else dataset + ".dvc"


def _dataset_at_rev(dvc_file: str, rev: str) -> dict[str, Any]:
    """md5/size of a dataset as recorded in git at `rev`."""
    _check_rev(rev)
    text = _git_ok(["show", f"{rev}:{dvc_file}"])
    info = _parse_dvc_text(text)
    commit = _git_ok(["rev-parse", rev])
    info.update({"rev": rev, "git_commit": commit})
    info["version"] = _version_label(dvc_file, commit)
    return info


def _version_label(dvc_file: str, commit: str) -> str | None:
    """Human version label = a `data-v*` git tag on the last commit that touched the .dvc file."""
    last = run_command(["git", "log", "-1", "--format=%H", commit, "--", dvc_file]).stdout.strip()
    if not last:
        return None
    tags = run_command(["git", "tag", "--points-at", last]).stdout.split()
    data_tags = sorted(t for t in tags if t.startswith("data-"))
    return data_tags[-1] if data_tags else None


# --- tool inputs -------------------------------------------------------------------------
class NoArgs(BaseModel):
    pass


class DatasetArg(BaseModel):
    dataset: str = Field(default="data/dataset.csv", description="Path of the tracked dataset, e.g. data/dataset.csv")


class CompareArgs(DatasetArg):
    rev_a: str = Field(description="Older git revision or data tag, e.g. data-v1")
    rev_b: str = Field(default="HEAD", description="Newer git revision or data tag, e.g. data-v2 or HEAD")


class ReproduceArgs(DatasetArg):
    rev: str = Field(description="Git revision or data tag whose dataset you want to reproduce, e.g. data-v1")


class ExperimentDatasetArgs(BaseModel):
    experiment_id: str | None = Field(default=None, description="MLflow experiment id (e.g. '25')")
    experiment_name: str | None = Field(default=None, description="MLflow experiment name")
    run_id: str | None = Field(default=None, description="Specific MLflow run id; overrides experiment lookup")


class CheckoutArgs(ReproduceArgs):
    pass


# --- tools -------------------------------------------------------------------------------
def list_datasets(_: NoArgs) -> dict[str, Any]:
    root = get_settings().repo_root
    found = []
    for f in sorted(root.rglob("*.dvc")):
        if not f.is_file() or ".dvc" in f.relative_to(root).parts[:-1] or "node_modules" in f.parts:
            continue
        rel = str(f.relative_to(root))
        try:
            info = _parse_dvc_text(f.read_text())
        except Exception as exc:  # noqa: BLE001
            found.append({"dvc_file": rel, "error": str(exc)})
            continue
        found.append({"dvc_file": rel, **info})
    remotes = run_command(["dvc", "remote", "list"])
    return {
        "datasets": found,
        "count": len(found),
        "remotes": [ln for ln in remotes.stdout.splitlines() if ln.strip()],
    }


def get_dataset_version(args: DatasetArg) -> dict[str, Any]:
    dvc_file = _dvc_file_for(args.dataset)
    commit = _git_ok(["rev-parse", "HEAD"])
    info = _dataset_at_rev(dvc_file, "HEAD")
    last_change = run_command(["git", "log", "-1", "--format=%H %cI %s", "--", dvc_file]).stdout.strip()
    sha, _, rest = last_change.partition(" ")
    return {
        "dataset": args.dataset,
        "version": info["version"] or f"md5:{str(info['md5'])[:8]}",
        "dvc_md5": info["md5"],
        "size_bytes": info["size"],
        "head_commit": commit,
        "dataset_last_changed_commit": sha,
        "dataset_last_changed": rest,
    }


def compare_dataset_versions(args: CompareArgs) -> dict[str, Any]:
    dvc_file = _dvc_file_for(args.dataset)
    a = _dataset_at_rev(dvc_file, args.rev_a)
    b = _dataset_at_rev(dvc_file, args.rev_b)
    diff_proc = run_command(["dvc", "diff", "--json", _check_rev(args.rev_a), _check_rev(args.rev_b)])
    dvc_diff: Any
    try:
        dvc_diff = json.loads(diff_proc.stdout or "{}")
    except json.JSONDecodeError:
        dvc_diff = {"raw": diff_proc.stdout[:500], "stderr": diff_proc.stderr[:300]}
    return {
        "a": a,
        "b": b,
        "identical": a["md5"] == b["md5"],
        "size_delta_bytes": (b["size"] or 0) - (a["size"] or 0),
        "dvc_diff": dvc_diff,
    }


def detect_data_changes(args: DatasetArg) -> dict[str, Any]:
    """Compare workspace data against the md5 recorded in the .dvc file (`dvc status`)."""
    dvc_file = _dvc_file_for(args.dataset)
    proc = run_command(["dvc", "status", "--json", dvc_file])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:400] or "dvc status failed")
    status = json.loads(proc.stdout or "{}")
    changed = bool(status)
    return {
        "dataset": args.dataset,
        "changed": changed,
        "dvc_status": status,
        "interpretation": "workspace data differs from the version recorded in git"
        if changed
        else "workspace data matches the version recorded in git",
    }


def explain_reproduce(args: ReproduceArgs) -> dict[str, Any]:
    dvc_file = _dvc_file_for(args.dataset)
    info = _dataset_at_rev(dvc_file, args.rev)
    steps = [
        f"git checkout {info['git_commit'][:12]} -- {dvc_file}    # restore the pointer file for {info['version'] or args.rev}",
        "dvc checkout                                    # restore data from local cache (or: dvc pull to fetch from remote)",
        f"dvc status {dvc_file}                          # should report 'Data and pipelines are up to date'",
        f"git checkout HEAD -- {dvc_file} && dvc checkout   # return to the latest dataset afterwards",
    ]
    return {
        "dataset": args.dataset,
        "target": info,
        "steps": steps,
        "note": "This tool only explains; use checkout_dataset_version (needs confirmation) to execute.",
    }


def checkout_dataset_version(args: CheckoutArgs) -> dict[str, Any]:
    """STATE-CHANGING: rewrites the workspace dataset to an older version.

    Safe by design: refuses to run if the workspace data has unsaved changes (never uses --force), and rolls the
    pointer file back if `dvc checkout` fails.
    """
    dvc_file = _dvc_file_for(args.dataset)
    info = _dataset_at_rev(dvc_file, args.rev)
    pre = run_command(["dvc", "status", "--json", dvc_file])
    if pre.stdout.strip() not in ("", "{}"):
        raise RuntimeError(
            "workspace dataset has unsaved changes relative to its DVC pointer; refusing to overwrite it (no --force is ever used)"
        )
    g = run_command(["git", "checkout", info["git_commit"], "--", dvc_file])
    if g.returncode != 0:
        raise RuntimeError(g.stderr.strip()[:300])
    d = run_command(["dvc", "checkout", dvc_file])
    if d.returncode != 0:
        run_command(["git", "checkout", "HEAD", "--", dvc_file])  # roll back the pointer
        raise RuntimeError(d.stderr.strip()[:300])
    st = run_command(["dvc", "status", "--json", dvc_file])
    return {"checked_out": info, "workspace_matches_version": (st.stdout.strip() in ("", "{}"))}


def get_experiment_dataset(args: ExperimentDatasetArgs) -> dict[str, Any]:
    """Read the dataset provenance *recorded* on MLflow runs (tags dataset.*)."""
    from agent.tools.mlflow_tools import _client  # local import avoids cycles

    client = _client()
    if args.run_id:
        runs = [client.get_run(args.run_id)]
    else:
        exp_id = args.experiment_id
        if not exp_id and args.experiment_name:
            exp = client.get_experiment_by_name(args.experiment_name)
            if exp is None:
                raise ValueError(f"experiment '{args.experiment_name}' not found")
            exp_id = exp.experiment_id
        if not exp_id:
            raise ValueError("provide run_id, experiment_id or experiment_name")
        runs = client.search_runs([exp_id], max_results=50, order_by=["attributes.start_time DESC"])
        if not runs:
            raise ValueError(f"no runs in experiment {exp_id}")
    root = get_settings().repo_root
    prov = []
    for r in runs:
        t = r.data.tags
        commit = t.get("dataset.git_commit")
        in_repo = False
        if commit and _REV_RE.match(commit):
            in_repo = run_command(["git", "cat-file", "-e", f"{commit}^{{commit}}"]).returncode == 0
        prov.append(
            {
                "run_id": r.info.run_id,
                "run_name": t.get("mlflow.runName"),
                "dataset_version": t.get("dataset.version"),
                "dataset_md5": t.get("dataset.md5"),
                "dataset_git_commit": commit,
                "dataset_path": t.get("dataset.path"),
                "recorded": bool(t.get("dataset.md5")),
                "commit_exists_in_local_repo": in_repo,
            }
        )
    versions = sorted({p["dataset_version"] or f"md5:{(p['dataset_md5'] or '?')[:8]}" for p in prov})
    current = None
    dvc_file = root / "data" / "dataset.csv.dvc"
    if dvc_file.exists():
        current = _parse_dvc_text(dvc_file.read_text())["md5"]
    return {
        "runs_inspected": len(prov),
        "provenance": prov[:10],
        "distinct_dataset_versions": versions,
        "current_workspace_md5": current,
        "matches_current_pointer": [p["run_id"] for p in prov if p["dataset_md5"] and p["dataset_md5"] == current][:10],
        "reproduce_hint": "use explain_dataset_reproduction with rev=<dataset_version or git commit>",
    }


TOOLS = [
    Tool("list_dvc_datasets", "List datasets tracked by DVC (path, md5, size) and configured DVC remotes.", NoArgs, list_datasets, "dvc"),
    Tool(
        "get_dataset_version",
        "Get the current dataset version label, DVC md5 and Git commit for a tracked dataset.",
        DatasetArg,
        get_dataset_version,
        "dvc",
    ),
    Tool(
        "compare_dataset_versions",
        "Compare two dataset versions (git revisions/data tags) using recorded md5/size and `dvc diff`.",
        CompareArgs,
        compare_dataset_versions,
        "dvc",
    ),
    Tool(
        "detect_data_changes",
        "Detect whether tracked data in the workspace changed relative to its recorded DVC version (`dvc status`).",
        DatasetArg,
        detect_data_changes,
        "dvc",
    ),
    Tool(
        "explain_dataset_reproduction",
        "Explain, step by step, how to reproduce a previous dataset version.",
        ReproduceArgs,
        explain_reproduce,
        "dvc",
    ),
    Tool(
        "get_experiment_dataset",
        "Return the dataset version / md5 / git commit RECORDED on an MLflow experiment or run (provenance). Use for 'which dataset version was used for experiment N / run X'.",
        ExperimentDatasetArgs,
        get_experiment_dataset,
        "dvc",
    ),
    Tool(
        "checkout_dataset_version",
        "Check out an older dataset version into the workspace (git + dvc checkout).",
        CheckoutArgs,
        checkout_dataset_version,
        "dvc",
        writes=True,
    ),
]
