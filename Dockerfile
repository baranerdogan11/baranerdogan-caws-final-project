# Forging Line — Streamlit App
# Packages only what's needed to run the app.

# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:latest AS uv
FROM python:3.13-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy only what the app needs to run
COPY pyproject.toml uv.lock ./
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY data/gold/ ./data/gold/

# Make the venv the active Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

EXPOSE 8501

# Run Streamlit on all interfaces
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
