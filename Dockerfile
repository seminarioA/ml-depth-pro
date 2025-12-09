# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Production-ready Dockerfile for Vision Processing API

# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_MODEL_PATH=yolov8n.pt \
    DEPTH_CHECKPOINT_PATH=/app/checkpoints/depth_pro_checkpoint.pt

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models /app/checkpoints && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=appuser:appuser pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    websockets \
    opencv-python-headless \
    ultralytics \
    && rm -rf /root/.cache/pip

# Copy application files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser vision_processor.py ./
COPY --chown=appuser:appuser api_server.py ./

# Download YOLOv8n model at build time
USER appuser
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLOv8n model downloaded')" || \
    echo "Warning: YOLOv8n model download failed, will retry at runtime"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Run the application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
