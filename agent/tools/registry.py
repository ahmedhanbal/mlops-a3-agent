"""Tool registry: typed inputs, permission model, timeouts, structured logging.

Every tool is a function taking a pydantic model and returning JSON-serialisable data.
The registry is the single choke point where we enforce:
  * argument validation            -> status "invalid_request"
  * wall-clock timeout             -> status "timeout"
  * confirmation for write tools   -> status "confirmation_required"
  * policy denials (namespace, ...) -> status "denied"
  * structured logs (tool, status, duration, request id), secrets redacted
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from agent.config import get_settings
from agent.logging_utils import log_event, redact
from agent.schemas import ToolResult


class PolicyDenied(Exception):
    """Raised by tools when an operation violates a safety policy (e.g. wrong namespace)."""


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    func: Callable[[Any], Any]
    module: str
    writes: bool = False  # True => changes external state => needs explicit confirmation

    def spec(self) -> dict[str, Any]:
        """Anthropic-style tool specification (name, description, JSON schema)."""
        suffix = " [STATE-CHANGING: requires explicit user confirmation]" if self.writes else " [read-only]"
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description + suffix, "input_schema": schema}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def by_module(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for t in self._tools.values():
            out.setdefault(t.module, []).append(t.name)
        return out

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        request_id: str,
        confirmed: bool = False,
        timeout_s: float | None = None,
    ) -> ToolResult:
        start = time.perf_counter()
        tool = self._tools.get(name)

        def finish(status: str, data: Any = None, error: str | None = None) -> ToolResult:
            dur = round((time.perf_counter() - start) * 1000, 2)
            res = ToolResult(tool=name, status=status, data=data, error=error, duration_ms=dur, request_id=request_id)  # type: ignore[arg-type]
            log_event(
                "tool_call",
                level=logging.INFO if status == "ok" else logging.WARNING,
                tool=name,
                status=status,
                duration_ms=dur,
                request_id=request_id,
                writes=bool(tool and tool.writes),
                arguments=redact(arguments or {}),
                error=error,
            )
            return res

        if tool is None:
            return finish("invalid_request", error=f"unknown tool '{name}'")
        try:
            parsed = tool.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            return finish("invalid_request", error=exc.errors(include_url=False).__repr__()[:500])
        if tool.writes and not confirmed:
            return finish(
                "confirmation_required",
                error=f"'{name}' changes system state. Re-submit the request with confirm=true to authorise it.",
            )

        timeout = timeout_s if timeout_s is not None else get_settings().tool_timeout_s
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.func, parsed)
        try:
            data = future.result(timeout=timeout)
            return finish("ok", data=data)
        except FutureTimeout:
            return finish("timeout", error=f"tool exceeded {timeout}s timeout")
        except PolicyDenied as exc:
            return finish("denied", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            return finish("error", error=f"{type(exc).__name__}: {exc}"[:600])
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


def build_registry() -> ToolRegistry:
    """Assemble the registry from the four tool modules."""
    from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools

    reg = ToolRegistry()
    for module in (dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools):
        for tool in module.TOOLS:
            reg.register(replace(tool))  # fresh copy: registries never share mutable Tool objects
    return reg
