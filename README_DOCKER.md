# Docker setup for DataMimicAI

This file contains the Docker and docker-compose usage instructions for development and production.

## Files added

- `backend/Dockerfile` — multi-stage image, installs `backend/requirements.txt`, runs Uvicorn in production. Dev command uses `--reload`.
- `frontend/Dockerfile` — lightweight Streamlit image, installs `frontend/requirements.txt`, starts Streamlit on 0.0.0.0.
- `docker-compose.yml` — defines `backend` and `frontend` services, healthchecks, ports, and env support via `.env`.
- `docker-compose.override.yml` — mounts local code and enables hot reload for development (`docker-compose up` will pick this up).
- `.env.example` — example environment variables to copy to `.env`.

## Quick PowerShell copy-paste commands

1) Copy example env and edit if needed:

```powershell
Copy-Item .env.example .env
# Edit .env with your values (optional). Use Notepad or your editor.
notepad .env
```

2) Build images (production images):

```powershell
docker-compose build
```

3) Run in production (no host mounts):

```powershell
docker-compose up -d
docker-compose logs -f
```

4) Run in development (hot reload, mounts):

```powershell
# docker-compose.override.yml is picked up automatically by docker-compose
docker-compose -f docker-compose.yml -f docker-compose.override.yml up --build
```

5) Stop and cleanup:

```powershell
docker-compose down --volumes --remove-orphans
```

## Verification checklist (PowerShell)

```powershell
# Check backend /health (expected HTTP 200 and JSON {"status":"ok"} if you add the endpoint)
Invoke-RestMethod -Uri "http://localhost:${env:BACKEND_PORT -or 8000}/health" -UseBasicParsing

# Check Streamlit root (returns HTML/JS; status implies reachable)
Invoke-RestMethod -Uri "http://localhost:${env:STREAMLIT_SERVER_PORT -or 8501}" -UseBasicParsing

# Inspect services and health
docker-compose ps
```

## Notes & assumptions

- Backend FastAPI entrypoint assumed: `app.main:app`. If different, update `backend/Dockerfile` CMD and override command.
- Backend requirements: `backend/requirements.txt`.
- Streamlit entrypoint assumed: `frontend/app.py`.
- No reverse proxy included; ports are published directly.
- For container-to-host access on Docker Desktop for Windows, `host.docker.internal` can be used. When both services run in compose, use the service name `backend` to reach it from `frontend`.

## Health endpoint snippet (add to your FastAPI app if missing)

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/health')
async def health():
    return {"status": "ok"}
```

## Optional: Gunicorn + Uvicorn for production

Install `gunicorn` and change the backend CMD to:

```text
gunicorn -k uvicorn.workers.UvicornWorker -w ${UVICORN_WORKERS:-2} -b 0.0.0.0:${PORT:-8000} app.main:app
```

---

If you want, I can now run a local `docker-compose build` and attempt verification (if Docker is available). Tell me to proceed.

```bash
```# 1. Copy env file
Copy-Item .env.example .env
notepad .env  # Review and update values

# 2. Create uploads directory if it doesn't exist
New-Item -ItemType Directory -Force -Path uploads

# 3. Build images
docker-compose build

# 4. Run in development mode
docker-compose -f docker-compose.yml -f docker-compose.override.yml up
```

```bash
Both containers are running:

Frontend container:
Port: 8501
You can access it at: http://localhost:8501


Backend container:
Port: 8000
API should be available at: http://localhost:8000
```
