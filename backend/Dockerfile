# Single, cleaned multi-stage Dockerfile for backend
### Stage 1: builder (install dependencies)
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build deps commonly required by Python packages (adjust as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /backend

# Copy and install python deps (this should be backend/requirements.txt)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

### Stage 2: runtime (smaller)
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime helpers (curl for healthchecks) and certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user for running the app
RUN useradd --create-home appuser || true
WORKDIR /backend

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app source into image
# We expect this Dockerfile to live in the `backend/` directory so COPY . will include src/ and other backend files
COPY . /backend

# Ensure proper ownership
RUN chown -R appuser:appuser /backend
# Give appuser ownership of the pycox package directory,
# as it tries to write data inside itself.
RUN chown -R appuser:appuser /usr/local/lib/python3.11/site-packages/pycox
USER appuser

# Expose default port
ARG PORT=8000
EXPOSE ${PORT}

# Default environment vars (can be overridden by compose/.env)
ENV PORT=${PORT}
ENV HOST=0.0.0.0
ENV UVICORN_WORKERS=1

# Default module to run. The project layout contains `src/api/main.py`, so default to that.
ENV BACKEND_MODULE=src.api.main:app

# Production CMD (use uvicorn directly; supports --workers)
CMD ["sh", "-c", "uvicorn ${BACKEND_MODULE} --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]

