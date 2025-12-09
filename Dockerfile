# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Production-ready Dockerfile for Vision Processing API

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_MODEL_PATH=yolov8n.pt \
    DEPTH_CHECKPOINT_PATH=/app/checkpoints/depth_pro_checkpoint.pt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libheif-dev \
    libde265-dev \
    libx265-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff5-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models /app/checkpoints && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copia primero los metadatos y el código para que pip tenga src/ y README.md disponibles
COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser src ./src

# Instala dependencias del proyecto y extras (no editable para evitar fallas de build)
RUN pip install --no-cache-dir --prefer-binary . && \
        pip install --no-cache-dir \
            fastapi \
            uvicorn[standard] \
            websockets \
            opencv-python-headless \
            ultralytics \
        && rm -rf /root/.cache/pip

# Copia los entrypoints de la app
COPY --chown=appuser:appuser vision_processor.py ./
COPY --chown=appuser:appuser api_server.py ./

USER appuser

# Descarga el modelo YOLOv8n en build (tolerante a fallo)
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLOv8n model downloaded')" || \
    echo 'Warning: YOLOv8n model download failed, will retry at runtime'

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]