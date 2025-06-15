# syntax=docker/dockerfile:1.4
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Utilisation du cache pip entre builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
