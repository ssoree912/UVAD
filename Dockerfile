# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3-pip \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.7 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data if they don't exist
RUN mkdir -p features/avenue/test features/avenue/train \
    features/ped2/test features/ped2/train \
    features/shanghaitech/test features/shanghaitech/train \
    patches/avenue/test patches/avenue/train \
    patches/ped2/test patches/ped2/train \
    patches/shanghaitech/test patches/shanghaitech/train \
    meta

# Make shell scripts executable
RUN chmod +x run1_pseudo_anomaly_scores.sh run2_evaluate.sh

# Default command
CMD ["/bin/bash"]
