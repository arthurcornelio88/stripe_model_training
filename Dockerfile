# Dockerfile
FROM python:3.11-slim

# Variables d'environnement pour la production
ENV ENV=PROD
ENV PYTHONPATH=/app
ENV PORT=8000

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Créer le répertoire de travail et les dossiers nécessaires
WORKDIR /app
RUN mkdir -p /app/shared_data /app/models /app/mlruns

# Copier les requirements de tous les modules
COPY model_training_api/requirements.txt /app/requirements_training.txt
COPY mock_realtime_api/requirements.txt /app/requirements_mock.txt

# Installer toutes les dépendances Python + MLflow + Google Cloud
RUN pip install --no-cache-dir -r requirements_training.txt
RUN pip install --no-cache-dir -r requirements_mock.txt
RUN pip install --no-cache-dir mlflow[extras]==2.8.1 psycopg2-binary google-cloud-secret-manager

# Copier le code source des APIs
COPY model_training_api/ /app/model_training_api/
COPY mock_realtime_api/ /app/mock_realtime_api/

# Copier les données de référence
COPY shared_data/fraudTest.csv /app/shared_data/fraudTest.csv

# Script d'entrée et secrets manager
COPY deployment/ /app/deployment/
RUN chmod +x /app/deployment/entrypoint.sh

# Exposer les ports (8000 pour APIs, 5000 pour MLflow)
EXPOSE 8000
EXPOSE 5000

# Point d'entrée
ENTRYPOINT ["/app/deployment/entrypoint.sh"]