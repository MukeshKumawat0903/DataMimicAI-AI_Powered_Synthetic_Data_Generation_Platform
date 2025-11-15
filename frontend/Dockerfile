FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime helpers and certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential gfortran libopenblas-dev liblapack-dev \
	curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies (cache this layer when requirements.txt unchanged)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app

# Use a non-root user for running the app
RUN useradd --create-home streamuser || true
RUN chown -R streamuser:streamuser /app
USER streamuser

ARG STREAMLIT_SERVER_PORT=8501
EXPOSE ${STREAMLIT_SERVER_PORT}

# Start Streamlit, binding to 0.0.0.0 and configurable port via env var
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port ${STREAMLIT_SERVER_PORT:-8501}"]