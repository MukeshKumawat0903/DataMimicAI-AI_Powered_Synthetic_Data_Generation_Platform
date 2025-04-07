# 🧠 DataMimicAI

**AI Powered Synthetic Data Generation Platform for Tabular and Time-Series Data**

DataMimicAI is an intelligent, end-to-end platform designed to generate high-quality synthetic data using advanced machine learning models like CTGAN, TVAE, and Gaussian Copula. It supports both tabular and time-series data, making it ideal for ML model training, privacy preservation, simulation, and educational purposes.

---

## 🚀 Features

- 🧠 AI-driven synthetic data generation (CTGAN, TVAE, etc.)
- 📊 Support for tabular and time-series data
- ⚙️ FastAPI backend for scalable APIs
- 🎛️ Streamlit frontend for user-friendly interaction
- 📦 Dockerized for local
- 📈 Data visualization for comparing real vs synthetic distributions
- 🔐 Pluggable with PostgreSQL / MongoDB (future-ready)

---

## 🛠️ Tech Stack

| Layer       | Tools / Tech |
|-------------|--------------|
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit (optional React) |
| **Data Synthesis** | SDV (CTGAN, TVAE, GaussianCopula), Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Database** | PostgreSQL (local), MongoDB (optional) |
| **Infrastructure** | Docker, Docker Compose, AWS EC2 (planned), Terraform, Kubernetes (planned) |
| **CI/CD** | GitHub Actions |
| **Testing** | Pytest |

---

## 📁 Project Directory Structure

```bash
Tabular_Synthetic_Data_Generation_App/
├── backend/
│   ├── src/
│   │   ├── api/
│   │   ├── core/
│   │   └── __init__.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── uploads/ 
├── docker-compose.yml
├── .render.yaml
└── README.md
```

---

## ⚡ Quick Start (Local Dev)

### ✅ Prerequisites:
- Python 3.9+
- Docker + Docker Compose
- Git

### 🚀 1. Clone the repo
```bash
git clone https://github.com/yourusername/DataMimicAI.git
cd DataMimicAI
```

### 🚀 2. Launch backend (FastAPI)
```bash
cd backend/src
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 🚀 3. Launch frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
```

---

## 🧪 API Endpoints

| Endpoint         | Method | Description |
|------------------|--------|-------------|
| `/`              | GET    | Welcome message |
| `/upload`        | POST   | Upload a CSV dataset |
| `/generate`      | POST   | Generate synthetic data |
| `/visualize`     | POST   | Compare real vs synthetic data |

---

## 🐳 Docker (Local Dev)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 🌐 Deployment (Planned)

- ✅ Local Docker + Streamlit
- 🚀 AWS EC2 instance (via SSH or GitHub Actions)
- ⚙️ Future: Terraform for infrastructure as code
- 🔁 Kubernetes for scalability

---

## 📌 Use Cases

- Model training & testing
- Bias reduction
- Data privacy & anonymization
- Educational demos
- Simulation & forecasting (e.g., energy, finance)

---

## 👤 Author

**Mukesh Kumawat**

> *Data Analyst | AI & ML Enthusiast | Builder of things that mimic data behavior intelligently.*

---

## 📃 License

MIT License — feel free to use, modify, and contribute!

---

