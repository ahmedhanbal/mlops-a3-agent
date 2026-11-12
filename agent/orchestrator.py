"""Agent orchestrator: LLM tool-calling loop + permission enforcement + evidence-grounded response."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from agent.config import Settings, get_settings
from agent.llm import LLM, SYSTEM_PROMPT, LLMReply, RuleBasedLLM, make_llm
from agent.logging_utils import log_event, new_request_id
from agent.reasoning import correlate, evidence_items
from agent.schemas import ChatResponse, ToolResult
from agent.tools.registry import ToolRegistry, build_registry

MAX_TOOL_PAYLOAD = 6000


class Agent:
    def __init__(self, registry: ToolRegistry | None = None, llm: LLM | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or build_registry()
        self.llm = llm or make_llm(self.settings)

    # -- LLM call with graceful degradation -------------------------------------------------------
    def _complete(self, messages: list[dict[str, Any]], rid: str) -> LLMReply:
        try:
            return self.llm.complete(SYSTEM_PROMPT, messages, self.registry.specs())
        except Exception as exc:  # noqa: BLE001 - LLM outage must not take the agent down
            log_event("llm_failed_fallback_to_rules", level=logging.WARNING, request_id=rid, error=type(exc).__name__)
            self.llm = RuleBasedLLM()
            return self.llm.complete(SYSTEM_PROMPT, messages, self.registry.specs())

    @staticmethod
    def _payload(res: ToolResult) -> str:
        text = res.model_dump_json()
        return text if len(text) <= MAX_TOOL_PAYLOAD else text[:MAX_TOOL_PAYLOAD] + '..."truncated"}'

    def run(self, query: str, confirm: bool = False, context: dict[str, Any] | None = None, request_id: str | None = None) -> ChatResponse:
        rid = request_id or new_request_id()
        t0 = time.perf_counter()
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        evidence: list[ToolResult] = []
        final_text = ""
        for _ in range(self.settings.max_agent_steps):
            reply = self._complete(messages, rid)
            if not reply.tool_calls:
                final_text = reply.text
                break
            messages.append({"role": "assistant", "content": reply.raw_content})
            blocks = []
            for call in reply.tool_calls:
                # `confirm` comes ONLY from the API caller; any 'confirm' the model puts in arguments is ignored.
                res = self.registry.execute(call.name, call.arguments, rid, confirmed=confirm)
                evidence.append(res)
                blocks.append({"type": "tool_result", "tool_use_id": call.id, "content": self._payload(res)})
            messages.append({"role": "user", "content": blocks})
        else:
            final_text = "Stopped: reached the maximum number of tool-use steps."

        report = correlate(evidence, query, context, self.settings)
        failed = [e for e in evidence if not e.ok and e.status != "confirmation_required"]
        if evidence and not any(e.ok for e in evidence):
            # nothing verified: discard whatever the model said and report only what we know
            answer = ("I could not verify anything: " + "; ".join(f"{e.tool} -> {e.status}" for e in evidence)
                      + ". I won't guess operational details without tool evidence.")
        else:
            answer = final_text.strip() or report.to_text()
            if failed and final_text.strip():
                answer += "\n\nNote: unavailable evidence - " + ", ".join(f"{e.tool} ({e.status})" for e in failed) + "."
            if final_text.strip() and evidence:
                answer += "\n\n" + report.to_text()
        log_event("agent_request", request_id=rid, tools=[e.tool for e in evidence], statuses=[e.status for e in evidence],
                  duration_ms=round((time.perf_counter() - t0) * 1000, 1), llm=self.llm.name)
        return ChatResponse(
            request_id=rid, answer=answer, known_facts=report.facts, uncertainties=report.uncertainties,
            recommended_actions=report.actions, evidence=evidence_items(evidence),
            tools_used=[e.tool for e in evidence], pending_confirmation=report.pending,
        )


def tool_catalog(registry: ToolRegistry) -> list[dict[str, Any]]:
    return [{"name": t["name"], "description": t["description"], "schema": json.loads(json.dumps(t["input_schema"]))} for t in registry.specs()]
