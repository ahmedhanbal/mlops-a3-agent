"""Runtime configuration. Everything comes from environment variables (no secrets in code)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in os.getenv(name, default).split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    repo_root: Path = field(default_factory=lambda: Path(os.getenv("MLOPS_REPO_ROOT", ".")).resolve())
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    airflow_base_url: str = field(default_factory=lambda: os.getenv("AIRFLOW_BASE_URL", "http://127.0.0.1:8080"))
    airflow_username: str = field(default_factory=lambda: os.getenv("AIRFLOW_USERNAME", "admin"))
    airflow_password: str = field(default_factory=lambda: os.getenv("AIRFLOW_PASSWORD", ""), repr=False)
    approved_dags: tuple[str, ...] = field(default_factory=lambda: _csv("AIRFLOW_APPROVED_DAGS", "training_pipeline"))
    k8s_namespace: str = field(default_factory=lambda: os.getenv("K8S_APPROVED_NAMESPACE", "mlops-agent"))
    k8s_app_label: str = field(default_factory=lambda: os.getenv("K8S_APP_LABEL", "app=iris-inference"))
    tool_timeout_s: float = field(default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_S", "30")))
    http_timeout_s: float = field(default_factory=lambda: float(os.getenv("HTTP_TIMEOUT_S", "10")))
    llm_mode: str = field(default_factory=lambda: os.getenv("AGENT_LLM", "auto"))
    llm_model: str = field(default_factory=lambda: os.getenv("AGENT_LLM_MODEL", "claude-sonnet-5-5"))
    f1_quality_gate: float = field(default_factory=lambda: float(os.getenv("F1_QUALITY_GATE", "0.93")))
    freshness_days: int = field(default_factory=lambda: int(os.getenv("FRESHNESS_POLICY_DAYS", "30")))
    max_agent_steps: int = field(default_factory=lambda: int(os.getenv("MAX_AGENT_STEPS", "6")))


def get_settings() -> Settings:
    """Read env on every call so tests (and operators) can change configuration at runtime."""
    return Settings()
