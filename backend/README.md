# DataMimicAI Backend

**Powering the synthetic data platform with AI-driven EDA, data profiling, and advanced feature engineering.**

---

## 🚀 Features

* **FastAPI-based REST API**
* **EDA Endpoints:**

  * Automated profiling, missing value fixes, data health report
  * Correlation, association, and data leakage checks
  * Outlier and drift detection
  * AI-driven feature suggestions
* **Synthetic Data Generation:**

  * Tabular synthesis (CTGAN, TVAE, Copula, etc.)
  * Supports custom, industry, or demo datasets
* **Pluggable, modular backend**

  * Clean separation of API (routing) and core data logic
  * Easy to extend for new ML/EDA features

---

## 📦 Directory Structure

```
backend/
│
├── main.py                # FastAPI entry point
├── requirements.txt       # Python dependencies
├── README.md              # This file!
│
├── src/
│   ├── api/               # API routers (all endpoints)
│   ├── core/              # EDA, feature eng, synthesis logic
│   ├── config.py          # Backend-wide config
│   └── ...                # (See tree above)
│
├── uploads/               # Uploaded datasets (dev only)
└── tests/                 # (Optional) Tests and utilities
```

---

## ⚙️ Getting Started

### 1. **Install Python & Dependencies**

* Python 3.9–3.11 recommended
* Install all backend deps:

```bash
cd backend
python -m venv venv
source venv/bin/activate      # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

---

### 2. **Run the API Server**

```bash
uvicorn main:app --reload
```

* Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger UI
* The API is ready for Streamlit frontend or manual API calls

---

### 3. **Environment Variables**

Set via `.env` or directly in shell.

| Variable          | Purpose                            | Example                 |
| ----------------- | ---------------------------------- | ----------------------- |
| `UPLOAD_DIR`      | Where uploads are stored           | `uploads`               |
| `API_URL`         | Public API base (used by frontend) | `http://localhost:8000` |
| `SYNTH_MODEL_DIR` | Pretrained model cache (optional)  | `models`                |

---

## 🔗 API Overview

Key endpoints (see `/docs` for details):

* `POST /upload` — Upload new dataset (CSV)
* `POST /eda/profile` — Automated data profiling/report
* `POST /eda/fix-missing` — Impute/fix missing values
* `POST /eda/correlation` — Correlation heatmaps, patterns
* `POST /eda/feature-suggestions` — AI feature engineering
* `POST /eda/detect-outliers` — Outlier detection/removal
* `POST /eda/detect-drift` — Drift detection between real/synthetic
* `POST /generate` — Run synthetic data generation

---

## 🛠️ Developer Notes

* **API routers**: `src/api/eda_feature_api.py` (add new endpoints here)
* **Business/data logic**:

  * Profiling/correlation/outliers: `src/core/eda/`
  * Synthetic generation: `src/core/synth/`
  * Utils/helpers: `src/core/utils/`
* **Uploads**: `uploads/` is gitignored (local/dev only, not for prod)

---

## 🧪 Testing

Add unit tests in `/tests/` (pytest recommended):

```bash
pytest
```

---

## ✨ Contributing

* Style: PEP8 for Python, docstrings for public functions/classes
* Write/extend endpoints in `src/api/`
* Modularize new EDA/synthesis features in `src/core/`
* Issues/PRs welcome!

---

## 📄 License

MIT (or your org’s preferred)

---

> For frontend/UI instructions, see `frontend/README.md`.

---