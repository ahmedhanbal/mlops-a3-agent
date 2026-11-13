"""Run the five required demonstration scenarios and write evidence/conversations.md (+ raw JSON).

Uses whatever services are reachable (set MLFLOW_TRACKING_URI, AIRFLOW_*, kube config). Services that are not
reachable are reported as unavailable by the agent itself - nothing is invented.
Set DEMO_SIMULATE_K8S=1 to substitute in-memory Kubernetes fakes (clearly labelled as simulated).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.orchestrator import Agent  # noqa: E402

SCENARIOS = [
    ("1. Dataset provenance", "Which dataset version was used for experiment 1?"),
    ("2. Model selection", "Compare the last five training runs and recommend the best candidate based on F1 score."),
    ("3. Pipeline execution", "Show the status of the latest training pipeline runs in Airflow."),
    ("4. Deployment diagnosis", "Why is my prediction API unavailable?"),
    (
        "5. Multi-tool reasoning",
        "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
    ),
]


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "evidence"
    out_dir.mkdir(exist_ok=True)
    if os.getenv("DEMO_SIMULATE_K8S") == "1":
        from agent.tools import kubernetes_tools as kt
        from tests.conftest import FakeApps, FakeCore, make_deployment, make_pod

        core, apps = FakeCore([make_pod("iris-inference-7d9f-abcde"), make_pod("iris-inference-7d9f-fghij")]), FakeApps(make_deployment())
        kt._core, kt._apps = lambda: core, lambda: apps
    agent = Agent()
    md = ["# Tested agent conversations (actual tool evidence)\n", f"LLM back-end: `{agent.llm.name}`\n"]
    raw = []
    for title, q in SCENARIOS:
        r = agent.run(q)
        raw.append({"scenario": title, "query": q, "response": r.model_dump()})
        md += [
            f"\n## {title}\n",
            f"**User:** {q}\n",
            f"**Request id:** `{r.request_id}` | **Tools:** {', '.join(r.tools_used)}\n",
            "\n**Agent:**\n",
            r.answer + "\n",
            "\n**Tool evidence:**\n",
            "| tool | status | ms | summary |\n|---|---|---|---|",
        ]
        md += [f"| {e.tool} | {e.status} | {e.duration_ms} | {e.summary[:110].replace('|', '/')} |" for e in r.evidence]
    (out_dir / "conversations.md").write_text("\n".join(md) + "\n")
    (out_dir / "conversations.json").write_text(json.dumps(raw, indent=2))
    print(f"wrote {out_dir}/conversations.md")


if __name__ == "__main__":
    main()
