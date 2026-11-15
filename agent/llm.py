"""LLM back-ends behind one interface.

* OpenAICompatLLM - provider-neutral tool calling over any OpenAI-compatible /chat/completions endpoint
                    (DeepSeek, OpenRouter, vLLM, Ollama, ...). Configured only by LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
* RuleBasedLLM  - offline deterministic planner (same interface). Used when no API key is configured, as the
                  fallback if the LLM API fails, and in tests/CI. It selects tools from the request text.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from agent.config import Settings
from agent.schemas import ToolCall


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def raw_content(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in self.tool_calls]
        return blocks


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply: ...


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the orchestrator's neutral block messages to OpenAI chat format."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif m["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            calls = [
                {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                for b in content
                if b["type"] == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # user turn carrying tool results
            out += [
                {"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]} for b in content if b["type"] == "tool_result"
            ]
    return out


class OpenAICompatLLM:
    name = "llm"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0) -> None:
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout
        self._key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "")  # never logged

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply:
        payload = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
            "tools": [
                {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools
            ],
            "temperature": 0,
            "max_tokens": 1500,
        }
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            calls.append(ToolCall(id=c["id"], name=c["function"]["name"], arguments=args if isinstance(args, dict) else {}))
        return LLMReply(text=msg.get("content") or "", tool_calls=calls)


# --- offline planner ----------------------------------------------------------------------------
Args = "dict[str, Any] | Callable[[dict[str, dict]], dict[str, Any] | None]"
_RUN_ID = re.compile(r"\b[0-9a-f]{32}\b")


def _metric(q: str) -> str:
    for m in ("accuracy", "precision", "recall", "f1"):
        if m in q:
            return m
    return "f1"


def plan(query: str) -> list[list[tuple[str, Any]]]:
    """Return phases; every phase is a list of (tool, args-or-callable). Later phases may use earlier results."""
    q = query.lower()
    has = lambda *w: any(x in q for x in w)  # noqa: E731
    k8s_kw = ("kubernetes", "k8s", "pod", "deployment", "deployed", "prediction api", "replica", "inference service", "unavailable")
    mandatory = has("best") and has("dataset") and has("deploy", "kubernetes", "k8s") and has("retrain")
    exp = re.search(r"experiment\s*(?:id\s*)?#?(\d+)", q)
    run = _RUN_ID.search(q)
    tags = re.findall(r"data-v\d+", q)

    p1: list[tuple[str, Any]] = []
    p2: list[tuple[str, Any]] = []

    if mandatory:
        p1.append(("find_best_run", {"metric": _metric(q)}))
        p2.append(
            (
                "get_experiment_dataset",
                lambda res: {"run_id": res["find_best_run"]["best"]["run_id"]} if "find_best_run" in res else {"experiment_id": "1"},
            )
        )
        p2 += [("diagnose_deployment", {}), ("get_deployed_version", {}), ("get_dag_run_status", {})]
        return [p1, p2]

    wants_data = has("dataset", "dvc", "data version", "provenance", "tracked data")
    best_then_data = wants_data and has("best", "top") and not (exp or run)
    if best_then_data:  # dependent calls: MLflow picks the run, DVC provenance is read for that run
        p1.append(("find_best_run", {"metric": _metric(q)}))
        p2.append(
            ("get_experiment_dataset", lambda res: {"run_id": res["find_best_run"]["best"]["run_id"]} if "find_best_run" in res else None)
        )
    elif wants_data or exp or run:
        if exp or run:
            p1.append(("get_experiment_dataset", {"experiment_id": exp.group(1)} if exp else {"run_id": run.group(0)}))
        elif has("reproduce"):
            p1.append(("explain_dataset_reproduction", {"rev": tags[0] if tags else "data-v1"}))
        elif has("compare") or len(tags) >= 2:
            p1.append(
                ("compare_dataset_versions", {"rev_a": tags[0] if tags else "data-v1", "rev_b": tags[1] if len(tags) > 1 else "HEAD"})
            )
        elif has("changed", "change", "modified", "drift", "tamper"):
            p1.append(("detect_data_changes", {}))
        elif has("list", "which datasets", "tracked"):
            p1.append(("list_dvc_datasets", {}))
        else:
            p1.append(("get_dataset_version", {}))
    elif has("best", "f1", "accuracy", "recall", "precision", "mlflow", "experiments", "runs", "registry", "model version", "compare"):
        if has("registry", "model version"):
            p1.append(("get_model_versions", {}))
        elif has("compare"):
            p1.append(("compare_mlflow_runs", {"last_n": 5, "metric": _metric(q)}))
        elif has("list") and has("experiments"):
            p1.append(("list_mlflow_experiments", {}))
        else:
            p1.append(("find_best_run", {"metric": _metric(q)}))
    if has(*k8s_kw):
        if has("restart"):
            p1.append(("restart_deployment", {}))
        elif has("version", "which image") and not has("unavailable", "why"):
            p1.append(("get_deployed_version", {}))
        elif has("service") and not has("unavailable", "why"):
            p1.append(("list_k8s_services", {}))
        elif has("unavailable", "why", "failing", "unhealthy", "health", "running", "diagnos", "down"):
            p1.append(("diagnose_deployment", {}))
        elif has("restart events", "crash"):
            p1.append(("find_unhealthy_pods", {}))
        else:
            p1.append(("list_k8s_workloads", {}))
    if has("airflow", "dag", "pipeline"):
        if has("trigger", "start", "kick off", "launch") or (has("run") and has("training")):
            p1.append(("trigger_training_dag", {}))
        elif has("fail"):
            p1.append(("find_failed_tasks", {}))
        elif has("list") and has("dag"):
            p1.append(("list_airflow_dags", {}))
        else:
            p1.append(("get_dag_run_status", {}))
    phases = [p1, p2] if p2 else [p1]
    return [ph for ph in phases if ph]


class RuleBasedLLM:
    name = "rules"

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply:
        query = messages[0]["content"] if isinstance(messages[0]["content"], str) else ""
        results: dict[str, dict] = {}
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                for block in m["content"]:
                    if block.get("type") == "tool_result":
                        try:
                            r = json.loads(block["content"])
                        except (ValueError, TypeError):
                            continue
                        if r.get("status") == "ok":
                            results[r["tool"]] = r["data"]
        phases = plan(query)
        done = sum(1 for m in messages if m["role"] == "assistant")
        if not phases:
            return LLMReply(
                text="I can inspect DVC datasets, MLflow experiments, Airflow pipelines and the Kubernetes deployment. "
                "Ask e.g. 'Which dataset version was used for experiment 1?' or 'Why is my prediction API unavailable?'"
            )
        if done >= len(phases):
            return LLMReply()
        calls = []
        for i, (name, args) in enumerate(phases[done]):
            resolved = args(results) if callable(args) else args
            if resolved is None:
                continue
            calls.append(ToolCall(id=f"rb-{done}-{i}", name=name, arguments=resolved))
        return LLMReply(tool_calls=calls)


SYSTEM_PROMPT = """You are an MLOps assistant with tools for DVC (data versions), MLflow (experiments/registry), \
Apache Airflow (training pipeline) and Kubernetes (the deployed inference API).
Rules:
1. Every operational claim (versions, metrics, run states, pod health) MUST come from a tool result in this conversation. \
Never answer such questions from memory or guess. If a tool failed or is unavailable, say so and do not fill the gap.
2. Choose the tools needed; for multi-part questions call several tools (a tool may depend on another's output, e.g. run id).
3. Separate (a) known facts, (b) uncertainties, (c) recommended actions. A high metric alone does not prove production readiness.
4. Recommend retraining only with evidence: failed quality gate, an explicit freshness policy breach, or supplied degradation results. \
Never trigger retraining unless the user explicitly asks for it.
5. State-changing tools (trigger, restart, checkout) require the user's explicit confirmation which only the API caller can give; \
if a tool reports confirmation_required, tell the user what needs confirming.
6. Never reveal secrets, tokens or credentials."""


def make_llm(settings: Settings) -> LLM:
    mode = settings.llm_mode
    if mode == "rules":
        return RuleBasedLLM()
    if mode == "llm" or (mode == "auto" and os.getenv("LLM_API_KEY")):
        return OpenAICompatLLM(settings.llm_base_url, settings.llm_model)
    return RuleBasedLLM()
