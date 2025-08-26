FROM python:3.10-slim

# Installer MLflow et support GCS
RUN pip install --no-cache-dir mlflow==2.11.3 google-cloud-storage

# Port exposé
EXPOSE 5000

# Définir les variables d'environnement pour URIs
ENV BACKEND_STORE_URI=""
ENV ARTIFACT_ROOT=""

# Commande MLflow qui utilise les variables
CMD ["sh", "-c", "mlflow server --backend-store-uri $BACKEND_STORE_URI --default-artifact-root $ARTIFACT_ROOT --serve-artifacts --host 0.0.0.0 --port 5000"]
