"""FastAPI inference service deployed on Kubernetes (the application the agent operates).

Configuration comes from the ConfigMap (env vars):
  APP_VERSION, MODEL_PATH, MODEL_RUN_ID, DATASET_VERSION, LOG_LEVEL
Probes: /health (liveness: process is up)  /ready (readiness: model loaded and usable)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

N_FEATURES = 30
log = logging.getLogger("inference")


class PredictRequest(BaseModel):
    features: list[float] = Field(description=f"{N_FEATURES} numeric features")

    @field_validator("features")
    @classmethod
    def _len(cls, v: list[float]) -> list[float]:
        if len(v) != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {len(v)}")
        return v


class State:
    model: Any = None
    error: str | None = None


state = State()


def load_model() -> None:
    path = os.getenv("MODEL_PATH", "models/model.joblib")
    try:
        state.model = joblib.load(path)
        state.error = None
        log.info("model loaded from %s", path)
    except Exception as exc:  # noqa: BLE001 - readiness must report, not crash the process
        state.model, state.error = None, f"{type(exc).__name__}: cannot load model from {path}"
        log.error("model load failed: %s", state.error)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_model()
    yield


app = FastAPI(title="Inference Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if state.model is None:
        raise HTTPException(status_code=503, detail=state.error or "model not loaded")
    return {"status": "ready"}


@app.get("/version")
def version() -> dict[str, str | None]:
    return {
        "app_version": os.getenv("APP_VERSION", "unknown"),
        "model_run_id": os.getenv("MODEL_RUN_ID"),
        "dataset_version": os.getenv("DATASET_VERSION"),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    pred = int(state.model.predict([req.features])[0])
    return {"prediction": pred, "label": "benign" if pred == 1 else "malignant"}
