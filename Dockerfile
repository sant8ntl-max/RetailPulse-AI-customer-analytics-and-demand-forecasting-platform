# ---------- stage 1: build wheels ----------
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements-app.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-app.txt

# ---------- stage 2: slim runtime ----------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
COPY requirements-app.txt .
RUN pip install --no-index --find-links=/wheels -r requirements-app.txt && pip install matplotlib seaborn
COPY src ./src
COPY app ./app
COPY data/processed ./data/processed
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
