FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "scikit-learn>=1.4" "joblib>=1.3" "pydantic>=2.6" numpy pandas
COPY api/ api/
# models/model.joblib is produced by `python -m training.train` (only exported when the quality gate passes)
COPY models/ models/
RUN useradd -r -u 10001 appuser && chown -R appuser /app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
