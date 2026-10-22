# DataMimicAI 🚀

**The Next-Gen Synthetic Data Platform for AI, Analytics, and Data Science**

> _Create, analyze, explain, and validate synthetic data with cutting-edge generative models, an AI assistant that proposes (never executes) fixes, and a guided four-step workflow._

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-legacy%20UI-FF4B4B?logo=streamlit&logoColor=white)
![SDV](https://img.shields.io/badge/SDV-1.17-4B8BBE)
![SynthCity](https://img.shields.io/badge/SynthCity-0.2.12-6A5ACD)

---

## 🌟 Overview

**DataMimicAI** is a no-code platform for **privacy-preserving, high-fidelity synthetic data generation** across **tabular and time-series datasets**.

It's built for **data scientists, ML engineers, and analytics teams**, and lets them:

- Turn raw CSVs into **synthetic datasets** in a few clicks
- Profile, clean, and engineer features before generation
- Audit datasets for **PII, k-anonymity, and re-identification risk**
- Get **LLM-powered explanations** of diagnostics and AI-proposed transformation plans, gated by human approval
- Validate synthetic data quality and iterate until it's fit for purpose

> 🔍 **Primary Focus:**
> Seamless, trustworthy, explainable synthetic data, **powered by SDV, SynthCity, and a human-in-the-loop AI assistant.**

---

## 🧪 Why Synthetic Data?

| Benefit | What it enables |
| --- | --- |
| 🔐 **Privacy-first** | Share realistic datasets with no PII leakage; support GDPR/HIPAA-friendly dev and test workflows |
| 🧠 **Smarter, fairer ML** | Upsample rare events, fill gaps in under-represented groups, stress-test models with edge cases and drift |
| 📊 **Safe testing** | Populate dev/QA/sandbox environments and cross-team migrations without exposing real data |
| ⚡ **Faster iteration** | Generate data for prototypes, hackathons, and demos without waiting for data approvals |

**Industry use cases:** 🏥 Healthcare · 💰 Finance (fraud, credit, risk) · 🏭 Manufacturing/IoT (rare faults, predictive maintenance) · 🚗 Autonomous systems · 💼 HR & Marketing

---

## 🖼️ Dashboard Preview

> Screenshots below are from the Streamlit UI.

![Data Load](image/README/deshboard_data_load.png)

![Smart Preview](image/README/Smart_Preview.png)

![EDA & Feature Engineering](image/README/eda_&_feature_engineering.png)

![Synthetic Data Generation](image/README/synthetic_data_generation.png)

![Visualization](image/README/Visualization.png)

---

## 🏗️ Architecture

```
┌──────────────────────────┐      ┌──────────────────────────┐
│  Next.js frontend (new)  │      │  Streamlit frontend      │
│  frontend-next/  :3000   │      │  frontend/  :8501        │
└────────────┬─────────────┘      └────────────┬─────────────┘
             │          REST (JSON / CSV)       │
             └───────────────┬──────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  FastAPI backend  :8000      │
              │  backend/src/api             │
              ├──────────────────────────────┤
              │  core/eda        profiling, outliers, drift, PII, time-series
              │  core/feature_engineering   utility & privacy suggestions
              │  core/synth      SDV + SynthCity generators, model selection
              │  core/ai_assistance         LLM explain, RAG, agents, approval gate
              │  core/execution  deterministic transformation engine
              │  core/validation quality scoring & feedback loop
              └──────────────┬───────────────┘
                             ▼
           PostgreSQL (optional) · uploads/ · Groq LLM API
```

**Two frontends, one backend.** The project is migrating from Streamlit to a **Next.js 14 + TypeScript** frontend. Both UIs talk to the same FastAPI backend. Streamlit is still the most feature-complete UI (it includes Agent Mode); the Next.js app covers the core Upload → Explore → Generate → Validate workflow.

---

## 🧭 Key Features

### ✅ Guided Four-Step Workflow

**Upload → Explore → Generate → Validate**

- Steps unlock progressively: Explore/Generate need a loaded dataset, and Validate needs a generated one
- Next.js: wizard step indicator, sidebar navigation, **Ctrl+1–4** keyboard shortcuts, and a guard that warns before you leave mid-task
- Workspace state persists across steps (Zustand in Next.js, session state in Streamlit)

### 📁 Step 1: Upload & Smart Preview

- Drag-and-drop CSV upload, or **Demo Mode** with built-in SDV sample datasets
- Auto-inferred schema: types, missing values, and basic stats
- KPI banner and an interactive data table for a quick readiness check

### 🔍 Step 2: Explore & Configure (EDA & Feature Engineering)

| Module | Highlights |
| --- | --- |
| 📄 **Data Profiling** | Structure, distributions, distribution fitting, quality flags, missing-value fixes |
| 💡 **Feature Suggestions** | Utility- and privacy-aware suggestions, conflict resolution, transform configs you can save, roll back, and export as code |
| 📊 **Correlation** | Pairwise correlation, mutual-information matrix, functional-dependency detection |
| ⚠️ **Outliers & Drift** | IQR / Isolation Forest / LOF / Cook's distance, time-series outliers, winsorize/cap/remove remediation with history, drift analysis |
| 🔒 **Privacy Audit** | Fast and deep PII scans (Presidio), quasi-identifier suggestions, k-anonymity scoring |
| ⏰ **Time-Series Analysis** | Auto-detection, ACF/PACF, seasonal decomposition |

### 🤖 AI Assistance (LLM Explain + Agent Mode)

A **read-only intelligence layer**: AI proposes, humans approve, and a deterministic engine executes.

- **🔍 Diagnostics:** Builds structured diagnostics from EDA signals and explains them in plain language using **Groq-hosted LLaMA 3.3 70B**, grounded by a **RAG knowledge base** (EDA, transformation, generator, and privacy rules in YAML)
- **🤖 Action Planner:** A transformation-planner agent drafts a step-by-step remediation plan with a rationale for each step
- **✅ Approval Gate:** Every plan is reviewed and explicitly approved or rejected by a human; nothing runs automatically
- **⚙️ Deterministic Execution:** Approved plans run through a non-AI execution engine with status tracking
- **📄 Decision Report:** An auditable record of what was proposed, approved, executed, and why
- Guardrails: baseline guard, signal selection, and output validation keep LLM responses grounded and well-formed

> More detail: [backend/src/core/ai_assistance/docs/AI_Assistance_Module_README.md](backend/src/core/ai_assistance/docs/AI_Assistance_Module_README.md)

### ⚙️ Step 3: Synthetic Data Generation

- **Standard (SDV):** CTGAN, TVAE, GaussianCopula, PAR (sequential/time-series)
- **Advanced (SynthCity):** TabDDPM (diffusion), CTGAN, TVAE, PrivBayes, DP-GAN, PATE-GAN, ARF
- **AutoML model selection:** Benchmarks candidate models and picks the best fit for your dataset
- **Presets:** Quick (200 rows / 100 epochs), Balanced (1,000 / 300), High Fidelity (10,000 / 500), plus saved custom presets
- Per-algorithm guidance on when to use each model

### ✅ Step 4: Validate & Refine

- **Quality Report:** Fidelity scoring and original-vs-synthetic distribution comparisons
- **Visualizations:** Side-by-side charts for distributions and drift
- **Iterate:** Version timeline of generated datasets, refinement recommendations, and a feedback loop that feeds back into generation

---

## 🧰 Tech Stack

| Layer | Technologies |
| --- | --- |
| **Backend** | FastAPI, Uvicorn, Pandas, NumPy, SciPy, scikit-learn, statsmodels, distfit, ennemi |
| **Synthesis** | SDV 1.17, SynthCity 0.2.12, Opacus (differential privacy) |
| **Privacy** | Microsoft Presidio, k-anonymity, quasi-identifier detection |
| **AI / LLM** | LangChain + Groq (LLaMA 3.3 70B), YAML RAG knowledge base |
| **Storage** | PostgreSQL via SQLAlchemy + asyncpg (optional), local `uploads/` |
| **Frontend (new)** | Next.js 14, React 18, TypeScript, Tailwind CSS, shadcn/ui, TanStack Query & Table, Zustand, React Hook Form + Zod, Recharts |
| **Frontend (legacy)** | Streamlit, Plotly |
| **DevOps** | Docker, docker-compose, Render (`render.yaml`) |

---

## 🏁 Quickstart

### Prerequisites

- **Python 3.9+** (3.11 recommended for local development)
- **Node.js 18+** and npm (for the Next.js frontend)
- A **Groq API key**, required only for the AI Assistance features ([get one here](https://console.groq.com/))
- PostgreSQL, optional; the app runs without it and logs a warning

### 1. Clone and configure

```bash
git clone https://github.com/MukeshKumawat0903/DataMimicAI-AI_Powered_Synthetic_Data_Generation_Platform.git
cd DataMimicAI-AI_Powered_Synthetic_Data_Generation_Platform
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

Add your LLM key (and optionally a database URL) to `.env`:

```env
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

### 2. Start the backend

```bash
cd backend
python -m venv .venv
# macOS/Linux: source .venv/bin/activate   |   Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs are then available at **http://localhost:8000/docs**.

### 3a. Start the Next.js frontend (recommended)

```bash
cd frontend-next
npm install
npm run dev
```

Open **http://localhost:3000**. The frontend uses `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`) to reach the backend.

> 🪟 **Windows one-click:** `run_frontend_next.bat` starts the backend and the Next.js frontend in separate windows. It finds `backend\.venv` automatically and runs `npm install` on first launch.

### 3b. Or start the Streamlit frontend (legacy, includes Agent Mode)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**. On macOS/Linux, `bash ./start.sh` launches the backend and Streamlit together.

### 4. Use it

Upload a CSV (or pick a demo dataset), then follow the steps: **Explore → Generate → Validate**.

---

## 🐳 Docker

The compose setup runs the **backend + Streamlit frontend**:

```bash
cp .env.example .env
docker-compose up --build          # dev: hot reload via docker-compose.override.yml
docker-compose up -d               # detached
docker-compose down
```

| Service | URL |
| --- | --- |
| Backend | http://localhost:8000 (health check: `/health`) |
| Streamlit | http://localhost:8501 |

See [README_DOCKER.md](README_DOCKER.md) for the full guide.

---

## ⚙️ Environment Variables

| Variable | Used by | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Backend | Enables LLM explanations and the Action Planner |
| `DATABASE_URL` | Backend | PostgreSQL connection string (optional) |
| `UPLOAD_DIR` | Backend | Where uploaded and generated files are stored |
| `LOG_LEVEL` | Backend | Logging verbosity (default `INFO`) |
| `CORS_ORIGINS` | Backend | Extra browser origins allowed to call the API, comma-separated (any `localhost` port is always allowed) |
| `BACKEND_PORT` / `UVICORN_WORKERS` | Docker | Backend port and worker count |
| `NEXT_PUBLIC_API_URL` | Next.js | Backend base URL (default `http://localhost:8000`) |
| `BACKEND_URL` / `API_URL` | Streamlit | Backend base URL |
| `STREAMLIT_SERVER_PORT` | Streamlit | Streamlit port (default `8501`) |

---

## 🔌 API Overview

The full interactive reference is at `/docs` (Swagger UI). Main route groups:

| Prefix | Purpose |
| --- | --- |
| `/upload`, `/load-demo`, `/generate`, `/evaluate_models`, `/visualize`, `/metrics`, `/health` | Core data and generation |
| `/eda/*` | Profiling, correlation, outliers, drift, feature suggestions, transforms, PII, k-anonymity, time-series |
| `/feedback-generation/*` | Feedback-driven regeneration and download |
| `/llm/*` | LLM explanations and explanation scopes |
| `/api/diagnostics/*` | Build and interpret diagnostics |
| `/api/planner/*` | AI transformation planning |
| `/api/approval/*` | Human review gate for plans |
| `/api/execution/*` | Deterministic plan execution and status |
| `/api/validation/*` | Decision reports and plan history |

---

## 🛠️ Project Structure

```
.
├── backend/                     # FastAPI service
│   ├── src/api/                 # Routers: main, eda_feature, llm, diagnostics, planner,
│   │                            #          approval, execution, validation, feedback
│   └── src/core/
│       ├── eda/                 # Profiling, correlation, outliers, drift, PII, privacy, time-series
│       ├── feature_engineering/ # Utility/privacy suggesters, conflict resolver
│       ├── synth/               # SDV & SynthCity generators, AutoML model selection
│       ├── ai_assistance/       # LLM explainability engine, RAG, agents, approval gate
│       ├── execution/           # Deterministic execution engine
│       └── validation/          # Validation feedback loop & storage
├── frontend-next/               # Next.js 14 + TypeScript frontend (new)
│   └── src/
│       ├── app/                 # Routes: /upload, /explore, /generate, /validate
│       ├── components/          # data, eda, generation, validation, wizard, layout, ui (shadcn)
│       ├── hooks/               # TanStack Query API hooks, shortcuts, navigation guard
│       └── lib/                 # API client, types, constants, Zustand workspace store
├── frontend/                    # Streamlit frontend (legacy, includes Agent Mode)
│   ├── app.py
│   ├── helpers/                 # Upload, EDA, generation, validation, visualization modules
│   └── ui/                      # Action Planner & Decision Report UIs
├── docker-compose.yml           # Backend + Streamlit services
├── render.yaml                  # Render deployment config
├── run_frontend_next.bat        # Windows launcher: backend + Next.js
└── start.sh                     # Bash launcher: backend + Streamlit
```

---

## 🗺️ Roadmap

- **🖥️ Next.js migration:** Bring Agent Mode (Diagnostics, Action Planner, Decision Report) to the new frontend and add it to Docker
- **✍️ LLM-driven generator:** Prompt/schema-to-data generation
- **🔒 Privacy & Risk Audit tab:** Automated re-identification risk scoring in AI Assistance
- **🏭 Industry templates:** Healthcare, Finance, Retail, and Education data simulation
- **📊 Advanced visualization hub** and SDMetrics-based evaluation reports

---

## 🔐 License

**Copyright Notice:**
All code and documentation in this repository is Copyright (c) 2025 [Mukesh Kumawat]. All rights are reserved.

This project is published publicly for showcasing and educational purposes only. No legal license for reuse, copying, modification, or distribution is granted. Unauthorized use is prohibited.
