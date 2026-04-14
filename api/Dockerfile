# Forging Line Diagnostics API
# Single-stage build for simplicity and reliability

FROM python:3.13-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic pytest

# Bundle application source and data files
COPY src/ ./src/
COPY reference_times.json ./reference_times.json

ENV PYTHONPATH="/app/src"

EXPOSE 80

# PYTHONPATH=/app/src so uvicorn finds app.py directly as 'app'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
