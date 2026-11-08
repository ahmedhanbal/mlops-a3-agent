"""Structured inputs/outputs shared by the tools, the orchestrator and the HTTP interface."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolStatus = Literal["ok", "error", "timeout", "invalid_request", "confirmation_required", "denied"]


class ToolResult(BaseModel):
    tool: str
    status: ToolStatus
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    request_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    confirm: bool = Field(
        default=False,
        description="Explicit human confirmation for state-changing tools. Only the API caller can set it; the LLM cannot.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional supplied evidence, e.g. {'degradation': {'metric': 'f1', 'value': 0.81}}",
    )


class EvidenceItem(BaseModel):
    tool: str
    status: str
    request_id: str
    duration_ms: float
    summary: str


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    known_facts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    pending_confirmation: list[str] = Field(default_factory=list)
