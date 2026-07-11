FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EASYOCR_MODULE_PATH=/models/easyocr \
    TORCH_HOME=/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libjpeg-turbo8 \
        libpng16-16 \
        libtiff5 \
        libraw20 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1 torchvision==0.18.1 \
    && python3 -m pip install -r requirements.txt

COPY . .

RUN mkdir -p /models/easyocr /models/torch \
    && python3 -m compileall -q /app \
    && python3 - <<'PY'
import easyocr
easyocr.Reader(["en"], gpu=False, verbose=False)
print("EasyOCR model cache initialized.")
PY

CMD ["python3", "-u", "handler.py"]
