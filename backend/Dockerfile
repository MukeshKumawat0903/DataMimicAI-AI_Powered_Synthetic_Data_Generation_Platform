FROM python:3.9-slim

# WORKDIR /app
WORKDIR /backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY /requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
# Copy entire backend folder contents (including api.py, database.py, models.py, src/)
# COPY . .
COPY . /backend


CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
