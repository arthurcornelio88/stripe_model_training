#!/bin/bash
# deployment/entrypoint.sh

echo "🚀 Starting MLOps API in $ENV mode..."

# Configuration centralisée des variables d'environnement
echo "� Setting up environment configuration..."
python3 -c "
import sys
sys.path.append('/app')
sys.path.append('/app/deployment')
from deployment.env_config import setup_environment, TRAINING_API_REQUIRED_VARS, MOCK_API_REQUIRED_VARS, MLFLOW_SERVER_REQUIRED_VARS

# Configuration centralisée
config = setup_environment()

# Validation selon le service
service_type = '$SERVICE_TYPE'
if service_type == 'training':
    config.validate_required_vars(TRAINING_API_REQUIRED_VARS)
elif service_type == 'mock':
    config.validate_required_vars(MOCK_API_REQUIRED_VARS)
elif service_type == 'mlflow':
    config.validate_required_vars(MLFLOW_SERVER_REQUIRED_VARS)
"

echo "✅ Environment configuration complete"

# Validation optionnelle (pour debug)
if [ "$DEBUG_ENV" = "true" ]; then
    echo "🔍 Running environment validation..."
    python3 /app/deployment/validate_env.py
fi

# Démarrer l'API appropriée basée sur la variable SERVICE_TYPE
if [ "$SERVICE_TYPE" = "mock" ]; then
    echo "🔄 Starting Mock Realtime API..."
    cd /app/mock_realtime_api
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
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
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
fi