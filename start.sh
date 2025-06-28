#!/bin/bash

# Start FastAPI backend in the background
(cd backend && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000) &

# Wait for backend to be ready (optional, e.g., sleep 5)
sleep 3

# Start Streamlit frontend (in current terminal)
(cd frontend && streamlit run app.py)