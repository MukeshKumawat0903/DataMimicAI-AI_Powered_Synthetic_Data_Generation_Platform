# DataMimicAI Backend

FastAPI backend for **DataMimicAI – AI‑Powered Synthetic Data Generation Platform**.

This service exposes APIs for:

- File upload & dataset management
- EDA & profiling
- Feature engineering suggestions
- Outlier & drift analysis
- Privacy checks
- Synthetic data generation (SDV, SynthCity, basic LLM modes)
- Quality & validation reports

The backend is designed to be consumed primarily by the Streamlit frontend (`frontend/app.py`), but you can call the APIs directly for automation.

---

## 1. Tech Stack

- **Python** (3.10+ recommended)
- **FastAPI** + **Uvicorn**
- **Pandas**, **NumPy**
- **SDV / SynthCity** (depending on enabled generators)
- **scikit‑learn** (metrics, preprocessing)
- Optional: **Docker** / **docker‑compose** for containerized runs

---

## 2. Project Structure (Backend Only)

```text
backend/
├─ src/
│  ├─ api/
│  │  ├─ main.py              # FastAPI app, router registration
│  │  ├─ files_api.py         # Upload, list, load, delete datasets
│  │  ├─ eda_api.py           # Profiling, summary stats, correlations
│  │  ├─ feature_api.py       # Feature suggestions, transformations
│  │  ├─ synth_api.py         # Synthetic data generation endpoints
│  │  ├─ validation_api.py    # Quality / utility / drift evaluation
│  │  └─ health_api.py        # Health/liveness checks
│  │
│  ├─ core/
│  │  ├─ eda/                 # EDA & profiling utilities
│  │  ├─ feature_engineering/ # Feature suggester & transforms
│  │  ├─ synth/               # Model wrappers for SDV / SynthCity / LLM
│  │  ├─ privacy/             # PII / privacy checks (where implemented)
│  │  └─ utils/               # Common helpers, caching, I/O
│  │
│  ├─ models/                 # Pydantic schemas
│  └─ services/               # Service layer (orchestration)
│
├─ tests/                     # Pytest‑based tests (unit & API)
├─ Dockerfile
├─ requirements.txt / pyproject.toml
└─ README.md                  # (this file)
```

> Note: Exact filenames may differ slightly; refer to `src/api` and `src/core` for the latest layout.

---

## 3. Environment Setup

From the **`backend/`** directory:

```bash
# 1) Create virtualenv (example using venv)
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt
# or (if using poetry)
# poetry install
```

Environment variables (typical):

```bash
# API base URL used by frontend (must match docker-compose if you use it)
export API_URL="http://localhost:8000"

# Optional: logging / model options
export LOG_LEVEL="info"
# export SYNTHCITY_ENABLED=true
# export SDV_SEED=42
```

On Windows PowerShell:

```powershell
$env:API_URL="http://localhost:8000"
$env:LOG_LEVEL="info"
```

---

## 4. Running the Backend

### Local (dev, hot‑reload)

From `backend/`:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- API docs (Swagger): http://localhost:8000/docs
- ReDoc docs: http://localhost:8000/redoc

The Streamlit frontend (in `frontend/app.py`) should point to the same `API_URL`.

### With Docker

From the repo root (where `docker-compose.yml` lives):

```bash
docker compose up --build backend
# or full stack:
docker compose up --build
```

This typically exposes:

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

---

## 5. Key API Endpoints (Overview)

> See live docs at `/docs` for the exact latest schemas.

### 5.1 Health

- `GET /health` – basic health check

### 5.2 File & Dataset Management

- `POST /files/upload` – upload CSV; returns `file_id`
- `GET  /files/{file_id}` – basic metadata
- `GET  /files/{file_id}/data` – load dataset (sample or full)
- `DELETE /files/{file_id}` – remove dataset

### 5.3 EDA & Profiling

- `POST /eda/summary` – high‑level stats (rows, columns, types, missing)
- `POST /eda/describe` – per‑column summary
- `POST /eda/correlation` – correlation matrix / top‑N correlations
- `POST /eda/distributions` – distribution info for selected columns

### 5.4 Feature Engineering & Suggestions

- `POST /features/suggestions`Inputs: dataset metadata, optional target columnOutputs: list of suggested transforms (fillna, encoding, binning, derived features, etc.)
- `POST /features/apply`
  Applies selected suggestions server‑side and returns a new `file_id` or transformed sample.
  The Streamlit app currently applies most transformations client‑side, but this endpoint is available for future server‑side use.

### 5.5 Synthetic Data Generation

- `POST /synth/generate`

  - Body includes:
    - `file_id` (source dataset)
    - generator type (`sdv_ctgan`, `sdv_copula`, `synthcity_*`, etc.)
    - row count / ratio
    - random seed / model params
  - Returns:
    - `generated_file_id`
    - basic metadata about synthetic dataset
- `GET /synth/{generated_file_id}/download` – download synthetic data as CSV

### 5.6 Validation & Quality

- `POST /validation/quality`

  - Utility metrics (distribution similarity, column‑wise distances)
- `POST /validation/drift`

  - Drift between original and synthetic datasets
- `POST /validation/privacy`

  - Where implemented: nearest‑neighbor / memorization‑style checks

---

## 6. How It Integrates with the Frontend

The Streamlit frontend (`frontend/app.py`) uses `API_BASE` (resolved from `CONFIG_API_BASE` + `API_URL` env var) to call this backend.

Typical flow:

1. User uploads CSV in frontend → frontend calls `POST /files/upload`
2. Frontend fetches EDA summaries → `POST /eda/*`
3. Feature suggestions shown → user applies them → data history maintained in Streamlit
4. User configures generator → frontend calls `POST /synth/generate`
5. Validation / comparison → `POST /validation/*`
6. Both original and synthetic can be downloaded from the frontend.

The **Quick Preview / Smart Preview** UI in the frontend now works purely from the current session DataFrame; the backend is used for heavier EDA and generation.

---

## 7. Testing

From `backend/`:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

You may need to start a local test DB or configure test fixtures if you add DB‑backed features later.

---

## 8. Development Notes

- Keep long‑running model training / generation off the main thread if you expect large datasets or many users (consider background tasks, Celery, or async jobs).
- Prefer **pydantic models** for request/response validation.
- When adding new endpoints:
  1. Add a pydantic schema in `src/models/`
  2. Implement logic in `src/core/*` or `src/services/*`
  3. Expose it from `src/api/*.py` with clear tags & descriptions
  4. Add minimal tests under `tests/`

---

## 9. License

**Copyright Notice:**
All code and documentation in this repository is Copyright (c) 2025 [Mukesh Kumawat]. All rights are reserved.

This project is published publicly for showcasing and educational purposes only. No legal license for reuse, copying, modification, or distribution is granted. Unauthorized use is prohibited.
