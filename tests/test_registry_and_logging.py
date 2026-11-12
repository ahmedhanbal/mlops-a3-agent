import json
import logging
import time

from pydantic import BaseModel, Field

from agent.logging_utils import JsonFormatter, redact
from agent.tools.registry import PolicyDenied, Tool, ToolRegistry


class Args(BaseModel):
    n: int = Field(default=1, ge=1, le=5)
    password: str | None = None


def reg_with(func, writes=False):
    r = ToolRegistry()
    r.register(Tool("t", "test tool", Args, func, "test", writes=writes))
    return r


def test_unknown_tool_is_invalid_request():
    res = ToolRegistry().execute("nope", {}, "rid1")
    assert res.status == "invalid_request" and "unknown tool" in res.error


def test_invalid_arguments_are_rejected_before_execution():
    called = []
    res = reg_with(lambda a: called.append(1)).execute("t", {"n": 99}, "rid2")
    assert res.status == "invalid_request" and not called


def test_tool_timeout_is_reported():
    res = reg_with(lambda a: time.sleep(1)).execute("t", {}, "rid3", timeout_s=0.1)
    assert res.status == "timeout" and "timeout" in res.error


def test_tool_exception_is_contained_as_error():
    def boom(a):
        raise ConnectionError("service down")

    res = reg_with(boom).execute("t", {}, "rid4")
    assert res.status == "error" and "ConnectionError" in res.error


def test_policy_denial_has_its_own_status():
    def deny(a):
        raise PolicyDenied("not allowed")

    assert reg_with(deny).execute("t", {}, "rid5").status == "denied"


def test_write_tool_needs_explicit_confirmation():
    called = []
    reg = reg_with(lambda a: called.append(1) or {"ok": 1}, writes=True)
    assert reg.execute("t", {}, "rid6").status == "confirmation_required" and not called
    assert reg.execute("t", {}, "rid6", confirmed=True).status == "ok" and called


def test_structured_logs_have_tool_status_duration_request_id_and_no_secrets(caplog):
    reg = reg_with(lambda a: {"ok": True})
    with caplog.at_level(logging.INFO, logger="mlops_agent"):
        reg.execute("t", {"n": 2, "password": "hunter2-super-secret"}, "req-abc")
    rec = next(r for r in caplog.records if r.getMessage() == "tool_call")
    line = JsonFormatter().format(rec)
    payload = json.loads(line)
    assert payload["tool"] == "t" and payload["status"] == "ok" and payload["request_id"] == "req-abc"
    assert isinstance(payload["duration_ms"], float)
    assert "hunter2" not in line


def test_redact_masks_tokens_keys_and_bearer_values():
    out = redact({"Authorization": "Bearer abc.def.ghi", "msg": "password=abc123 ok", "nested": [{"api_key": "sk-1234567890abcdef"}]})
    text = json.dumps(out)
    assert "abc123" not in text and "sk-1234567890" not in text and "Bearer abc" not in text
