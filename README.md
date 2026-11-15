# Assignment 03 - Production-Ready MLOps AI Agent

An AI agent that answers natural-language MLOps questions by calling **real tools** for **DVC**, **MLflow**, **Apache Airflow 3** and **Kubernetes**,
and answers only from tool evidence (facts / uncertainties / recommended actions).
Design details: [architecture.md](architecture.md) · Technical report: [technical_report.md](technical_report.md) · Evidence: [evidence/](evidence/), [screenshots/](screenshots/).

## 1. Installation (reproducible environment)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # or: pip install -r requirements-lock.txt && pip install -e .
cp .env.example .env               # fill in values; NEVER commit .env
set -a; source .env; set +a
```
Airflow 3 is installed separately (heavy, pinned by constraints):
```bash
python3.11 -m venv .af-venv && source .af-venv/bin/activate
pip install "apache-airflow==3.0.6" --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.11.txt
pip install mlflow-skinny scikit-learn pandas joblib pyyaml
```

## 2. Environment configuration (all via env vars - see `.env.example`)
| Variable | Meaning |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server (`http://127.0.0.1:5000`) or `sqlite:///mlflow.db` |
| `AIRFLOW_BASE_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | Airflow 3 API (JWT via `POST /auth/token`) |
| `AIRFLOW_APPROVED_DAGS` | only these DAGs can be triggered (default `training_pipeline`) |
| `K8S_APPROVED_NAMESPACE` | the only namespace the agent may read/modify (`mlops-agent`) |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `AGENT_LLM` (`auto`/`rules`/`llm`) | any OpenAI-compatible LLM (default DeepSeek `deepseek-chat`); with no key an offline rule-based planner is used |
| `AGENT_API_KEY` | if set, `/chat` requires header `X-API-Key` |
| `F1_QUALITY_GATE`, `FRESHNESS_POLICY_DAYS` | quality gate (0.93) and freshness policy (30 d) |

## 3. Launching each service
1. **Demo data + DVC + MLflow runs**: `./scripts/seed_demo.sh` (creates `data-v1`/`data-v2` tags, local DVC remote, 8 MLflow runs incl. a failed gate).
2. **MLflow UI**: `mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000`
3. **Airflow 3**: `export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags PYTHONPATH=$PWD; airflow standalone` (admin password printed / in `simple_auth_manager_passwords.json.generated`). Quick run without a scheduler: `airflow dags test training_pipeline`.
4. **Kubernetes** (kind/minikube): 
   ```bash
   python -m training.train                       # exports models/model.joblib (only if the quality gate passes)
   docker build -t mlops-inference:1.0.0 . && kind load docker-image mlops-inference:1.0.0
   kubectl apply -f kubernetes/namespace.yaml -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml -f kubernetes/rbac.yaml
   ```
   Set `MODEL_RUN_ID` in `kubernetes/configmap.yaml` to the MLflow run id of the deployed model.
5. **Agent API**: `python -m agent.main` (http://127.0.0.1:9000/docs).

## 4. Agent interaction
```bash
curl -s localhost:9000/chat -H 'content-type: application/json' \
  -d '{"query":"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."}'
```
State-changing requests (trigger DAG, restart deployment, dataset checkout) return `pending_confirmation`; repeat with `"confirm": true`. Supplied degradation evidence: `"context": {"degradation": {"metric":"f1","value":0.81}}`.
`GET /tools` lists the 26 tools (7 DVC, 7 MLflow, 5 Airflow, 7 Kubernetes).

## 5. Tests and CI
`pytest -q` (89 tests; Kubernetes/Airflow/LLM mocked; real DVC + MLflow in temp dirs) · `ruff check . && ruff format --check .` · `.github/workflows/ci.yml` runs both on every PR and push.
