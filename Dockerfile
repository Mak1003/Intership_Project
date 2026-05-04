# ─────────────────────────────────────────────
# Stage 1: Frontend (nginx serving index.html)
# ─────────────────────────────────────────────
FROM nginx:alpine AS frontend

COPY frontend/index.html /usr/share/nginx/html/index.html

EXPOSE 80

# ─────────────────────────────────────────────
# Stage 2: Backend (FastAPI + OpenCV)
# ─────────────────────────────────────────────
FROM python:3.11-slim AS backend

WORKDIR /app

# System dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]