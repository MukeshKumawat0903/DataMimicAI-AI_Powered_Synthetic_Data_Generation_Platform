# DataMimicAI Frontend

**Streamlit UI for Synthetic Data Generation, EDA, and Feature Engineering.**

---

## 🎯 Features

* **Modern Streamlit UI**: Multi-tab workflow (upload, generation, EDA, visualization, roadmap)
* **Automated EDA**: Data profiling, correlation, feature suggestions, outlier & drift detection
* **Seamless Backend Integration**: Works with FastAPI backend (see `/backend`)
* **Modular Helpers**: All logic split for easy extension and clarity

---

## 📁 Directory Structure

```
frontend/
│
├── app.py                          # Main Streamlit entry point
├── frontend_config.py              # UI config/constants
│
├── helpers/
│   ├── __init__.py
│   ├── file_upload.py              # Upload & demo data
│   ├── generation.py               # Synthetic data UI/logic
│   ├── visualization.py            # Visualization controls
│   ├── roadmap.py                  # Roadmap/features coming soon
│   │
│   └── eda_feature_eng/            # EDA/feature eng expanders
│       ├── __init__.py
│       ├── expander_data_profiling.py
│       ├── expander_correlation.py
│       ├── expander_feature_suggestions.py
│       └── expander_outlier_and_drift.py
│
├── assets/                         # (Optional) Images, logos, sample CSVs
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### 1. **Install Python & Dependencies**

* Python 3.9–3.11 recommended
* Install frontend dependencies:

```bash
cd frontend
python -m venv venv
source venv/bin/activate      # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

---

### 2. **Configure API Endpoint**

By default, the frontend tries to reach the backend at `http://localhost:8000`.

* To use a different API base (e.g., cloud deployment), set the environment variable:

```bash
export API_URL="http://your-api-host:8000"
```

* Or edit `frontend_config.py` as needed.

---

### 3. **Run the Streamlit App**

```bash
streamlit run app.py
```

* Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ How It Works

* **app.py**: Orchestrates all tabs (Upload, Generation, EDA, Visualization, Roadmap)
* **Modular helpers/**:

  * Each major section (`file_upload`, `generation`, etc.) has its own Python module
  * EDA/feature expanders split under `eda_feature_eng/` for maintainability
* **Config**: Texts, constants, and API base in `frontend_config.py`

---

## ✨ Developer Notes

* **Extend EDA**: Add new expanders under `helpers/eda_feature_eng/`
* **Add UI logic**: All business/UI logic is in helpers (no clutter in `app.py`)
* **Assets**: Store custom images, sample data, or logos in `/assets`
* **Requirements**: Check for Streamlit and requests in `requirements.txt`

---

## 🚀 Roadmap & Customization

* Feature roadmap is built into the app (see last tab)
* To add new ML or visualization features, simply add a helper module and import it in `app.py`

---

## 9. License

**Copyright Notice:**
All code and documentation in this repository is Copyright (c) 2025 [Mukesh Kumawat]. All rights are reserved.

This project is published publicly for showcasing and educational purposes only. No legal license for reuse, copying, modification, or distribution is granted. Unauthorized use is prohibited.

---

For backend/API instructions, see `backend/README.md`.

---
