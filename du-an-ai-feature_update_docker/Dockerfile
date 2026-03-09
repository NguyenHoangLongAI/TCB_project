# =========================================================
# BASE IMAGE: CUDA 12.4 + cuDNN9 (RTX 50xx REQUIRED)
# =========================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# ---------------------------------------------------------
# ENV
# ---------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=300 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ---------------------------------------------------------
# INSTALL PYTHON 3.11
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# SYSTEM DEPENDENCIES (OCR / PDF / OPENCV)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-vie \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl wget git ca-certificates dos2unix \
    build-essential \
    libgomp1 \
    libmagic1 \
    libpoppler-cpp-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# WORKDIR
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# NUMPY (Torch-safe)
# ---------------------------------------------------------
RUN pip install "numpy<2.0"
RUN pip install hf_transfer
# ---------------------------------------------------------
# PYTORCH CUDA 12.4 (RTX 50xx SUPPORT)
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------
# QUICK GPU CHECK (NO ASSERT – SAFE)
# ---------------------------------------------------------
RUN python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY

# ---------------------------------------------------------
# EASYOCR (USE TORCH CUDA)
# ---------------------------------------------------------
RUN pip install easyocr==1.7.1

# ---------------------------------------------------------
# APP DEPENDENCIES
# ---------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN dos2unix /app/requirements.txt || true \
 && sed -i '1s/^\xEF\xBB\xBF//' /app/requirements.txt \
 && pip install -r /app/requirements.txt

# ---------------------------------------------------------
# OPTIONAL: PRE-DOWNLOAD EASYOCR MODELS
# ---------------------------------------------------------
RUN python - <<'PY' || echo "⚠️ EasyOCR model preload skipped"
import easyocr
easyocr.Reader(['vi','en'], gpu=True)
PY

# ---------------------------------------------------------
# COPY SOURCE
# ---------------------------------------------------------
COPY . /app

RUN mkdir -p /app/uploads /app/models

# ---------------------------------------------------------
# EXPOSE
# ---------------------------------------------------------
EXPOSE 8000 8501

# ---------------------------------------------------------
# RUNTIME CHECK (FAIL ONLY AT RUNTIME, NOT BUILD)
# ---------------------------------------------------------
CMD ["bash", "-lc", "python - <<'PY'\nimport torch\nprint('torch', torch.__version__)\nprint('cuda_build', torch.version.cuda)\nprint('cuda_available', torch.cuda.is_available())\nif torch.cuda.is_available():\n    print('GPU:', torch.cuda.get_device_name(0))\n    print('Capability:', torch.cuda.get_device_capability(0))\nPY"]
