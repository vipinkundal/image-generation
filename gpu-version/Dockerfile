# GPU-Optimized Dockerfile for AI Image Generator
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Set environment variables for GPU performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# HuggingFace cache settings
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HUB_CACHE=/app/cache
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# Network timeout settings
ENV HF_HUB_DOWNLOAD_TIMEOUT=600

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .


# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --upgrade pip

# Install all requirements (including torch/torchvision/torchaudio)
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Create cache directories
RUN mkdir -p /app/cache /root/.cache/huggingface

# Expose port
EXPOSE 7860

# Run application with verbose output
CMD ["python3", "-u", "app.py"]
