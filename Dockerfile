FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY handler.py .
COPY processor.py .

CMD ["python", "-u", "handler.py"]
