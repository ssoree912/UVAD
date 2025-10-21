# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_FORCE_IPV4=false

# 기본 도구 + 런타임 라이브러리
RUN printf 'Acquire::http::Proxy "false";\nAcquire::https::Proxy "false";\n' > /etc/apt/apt.conf.d/99no-proxy && \
    apt-get update -o Acquire::ForceIPv4=${APT_FORCE_IPV4} && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl bzip2 \
      git screen \
      ffmpeg \
      libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Miniforge(Conda) 설치
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/mf.sh && \
    bash /tmp/mf.sh -b -p /opt/conda && rm /tmp/mf.sh
ENV PATH=/opt/conda/bin:$PATH
SHELL ["bash", "-lc"]

# Python 3.7 환경 생성 (+ pip/setuptools 호환 버전 고정)
RUN conda create -y -n py37 python=3.7 && \
    conda run -n py37 python -V && \
    conda run -n py37 python -m pip install --upgrade "pip==23.2.1" "setuptools<69" "wheel<0.42"

# 기본은 쉘로 진입 (요구사항 직접 설치/실행 전제)
CMD ["bash"]