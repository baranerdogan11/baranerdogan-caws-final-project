# Forging Line — Streamlit App

FROM ghcr.io/astral-sh/uv:latest AS uv
FROM python:3.13-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim AS runtime

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

COPY pyproject.toml uv.lock ./
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY data/gold/ ./data/gold/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV SAGEMAKER_ENDPOINT_NAME="vaultech-bath-endpoint"
ENV AWS_DEFAULT_REGION="eu-west-1"

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
