#!/bin/bash
# deployment/entrypoint.sh

echo "🚀 Starting MLOps API in $ENV mode..."

# Load secrets from Google Secret Manager in production
if [ "$ENV" = "PROD" ]; then
    echo "🔐 Loading secrets from Google Secret Manager..."
    python3 -c "
import sys
sys.path.append('/app')
sys.path.append('/app/deployment')
from deployment.secret_manager import setup_production_secrets
setup_production_secrets()
"
    echo "✅ Secrets loaded"
fi

# Configuration basée sur l'environnement
if [ "$ENV" = "PROD" ]; then
    echo "📊 Production mode: Using GCS for model storage"
    export SHARED_DATA_PATH="gs://${GCS_BUCKET}/shared_data"
    export MODEL_PATH="gs://${GCS_BUCKET}/models"
    
    # Configuration MLflow pour la production (secrets already loaded)
    export MLFLOW_ARTIFACT_URI="gs://${GCS_BUCKET}/mlflow-artifacts"
    export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"  # Simple backend for production
    # MLFLOW_EXPERIMENT is already loaded from secrets
    
else
    echo "🔧 Development mode: Using local storage"
    export SHARED_DATA_PATH="/app/shared_data"
    export MODEL_PATH="/app/models"
    
    # Configuration MLflow pour le développement
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    export MLFLOW_ARTIFACT_URI="./mlruns"
    export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"
    export MLFLOW_EXPERIMENT_NAME="fraud-detection-dev"
fi

# Créer les répertoires nécessaires en mode DEV
if [ "$ENV" != "PROD" ]; then
    mkdir -p /app/shared_data /app/models /app/mlruns
fi

# Démarrer l'API appropriée basée sur la variable SERVICE_TYPE
if [ "$SERVICE_TYPE" = "mock" ]; then
    echo "🔄 Starting Mock Realtime API..."
    cd /app/mock_realtime_api
    python main.py
elif [ "$SERVICE_TYPE" = "mlflow" ]; then
    echo "📊 Starting MLflow Tracking Server..."
    mlflow server \
        --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
        --default-artifact-root "${MLFLOW_ARTIFACT_URI}" \
        --host 0.0.0.0 \
        --port 5000
else
    echo "🧠 Starting Model Training API..."
    cd /app/model_training_api
    python main.py
fi