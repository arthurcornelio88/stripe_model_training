# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# ✅ Étape 0 : Installer Git (nécessaire pour MLflow)
RUN apt-get update && apt-get install -y git

# Étape 1 : Copier uniquement requirements.txt pour permettre le cache pip
COPY requirements.txt .

# Étape 2 : Installer les dépendances avec cache pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Étape 3 : Copier uniquement les fichiers nécessaires (pas tout le contexte)
COPY main.py .
COPY src/ ./src
COPY utils/ ./utils

# Étape 4 : Commande de démarrage de l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
