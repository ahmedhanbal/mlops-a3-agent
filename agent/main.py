"""FastAPI interface of the agent: request validation -> Agent -> evidence-based response."""

from __future__ import annotations

import hmac
import logging
import os

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException

from agent.logging_utils import log_event, new_request_id
from agent.orchestrator import Agent, tool_catalog
from agent.schemas import ChatRequest, ChatResponse


def create_app(agent: Agent | None = None) -> FastAPI:
    app = FastAPI(title="MLOps AI Agent", version="1.0.0")
    holder: dict[str, Agent] = {}

    def get_agent() -> Agent:
        if agent is not None:
            return agent
        if "a" not in holder:
            holder["a"] = Agent()
        return holder["a"]

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = os.getenv("AGENT_API_KEY", "")
        if expected and not hmac.compare_digest(x_api_key or "", expected):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tools", dependencies=[Depends(check_key)])
    def tools() -> list[dict]:
        return tool_catalog(get_agent().registry)

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_key)])
    def chat(req: ChatRequest) -> ChatResponse:
        rid = new_request_id()
        try:
            return get_agent().run(req.query, confirm=req.confirm, context=req.context, request_id=rid)
        except Exception as exc:  # noqa: BLE001 - never leak internals/secrets to the client
            log_event("chat_failed", level=logging.ERROR, request_id=rid, error=type(exc).__name__)
            raise HTTPException(status_code=500, detail=f"internal error (request_id={rid})") from exc

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("agent.main:app", host=os.getenv("AGENT_HOST", "127.0.0.1"), port=int(os.getenv("AGENT_PORT", "9000")))
