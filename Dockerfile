FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY handler.py .

CMD ["python", "handler.py"]
