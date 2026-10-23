FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/fraud_model.joblib

WORKDIR /app

# Install dependencies first so this layer is cached when only source code changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# The trained model is mounted at runtime (-v ./models:/app/models), not baked into the image.
RUN useradd --create-home appuser && mkdir -p /app/models && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "fraud_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]
