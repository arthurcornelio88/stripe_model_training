#!/bin/bash
# deployment/deploy_all_services.sh

set -e

# Get project and region from gcloud config or secrets
PROJECT_ID=$(gcloud config get-value project)
REGION=$(gcloud secrets versions access latest --secret="gcp-region" 2>/dev/null || echo "europe-west1")
GCS_BUCKET=$(gcloud secrets versions access latest --secret="gcp-bucket")
MLFLOW_URI=$(gcloud secrets versions access latest --secret="mlflow-tracking-uri")

IMAGE_NAME="gcr.io/${PROJECT_ID}/mlops-api"

echo "🚀 Deploying all MLOps services..."
echo "📋 Project: $PROJECT_ID"
echo "📍 Region: $REGION" 
echo "🗂️  Bucket: $GCS_BUCKET"

# 1. Build l'image Docker depuis le répertoire parent
echo "📦 Building unified Docker image..."
cd ..  # Aller dans model_training/ où se trouve le Dockerfile

# Vérifier que le Dockerfile existe
if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile not found in $(pwd)"
    echo "💡 Make sure you're running this script from model_training/deployment/"
    exit 1
fi

# Copier le fichier fraudTest.csv dans le contexte de build
echo "📋 Copying fraudTest.csv to build context..."
mkdir -p shared_data

# Vérifier que le fichier source existe
if [ ! -f "../shared_data/fraudTest.csv" ]; then
    echo "❌ Source file ../shared_data/fraudTest.csv not found"
    exit 1
fi

# Copier le fichier
cp ../shared_data/fraudTest.csv shared_data/fraudTest.csv

# Vérifier que le fichier a été copié
if [ ! -f "shared_data/fraudTest.csv" ]; then
    echo "❌ Failed to copy fraudTest.csv"
    exit 1
fi
echo "✅ fraudTest.csv copied successfully (size: $(stat -c%s shared_data/fraudTest.csv) bytes)"

docker build -t ${IMAGE_NAME} .
docker push ${IMAGE_NAME}

# Nettoyer le fichier temporaire
echo "🧹 Cleaning up temporary files..."
rm -rf shared_data/

cd deployment  # Revenir dans deployment/

# 2. Deploy Model Training API
echo "🧠 Deploying Model Training API..."
gcloud run deploy mlops-training-api \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars ENV=PROD,SERVICE_TYPE=training \
  --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID} \
  --set-env-vars MLFLOW_TRACKING_URI=${MLFLOW_URI} \
  --memory 2Gi \
  --cpu 2 \
  --port 8000 \
  --timeout 600

# 3. Deploy Mock Realtime API
echo "🔄 Deploying Mock Realtime API..."
gcloud run deploy mlops-mock-api \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars ENV=PROD,SERVICE_TYPE=mock \
  --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID} \
  --memory 1Gi \
  --cpu 1 \
  --port 8000 \
  --timeout 300

# 4. Deploy MLflow Tracking Server
echo "📊 Deploying MLflow Tracking Server..."
gcloud run deploy mlops-mlflow \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars ENV=PROD,SERVICE_TYPE=mlflow \
  --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID} \
  --memory 1Gi \
  --cpu 1 \
  --port 5000

echo "✅ All services deployed successfully!"

# Get deployed service URLs
echo ""
echo "🌐 Service URLs:"
TRAINING_URL=$(gcloud run services describe mlops-training-api --region=${REGION} --format='value(status.url)' 2>/dev/null || echo "❌ Not found")
MOCK_URL=$(gcloud run services describe mlops-mock-api --region=${REGION} --format='value(status.url)' 2>/dev/null || echo "❌ Not found")
MLFLOW_URL=$(gcloud run services describe mlops-mlflow --region=${REGION} --format='value(status.url)' 2>/dev/null || echo "❌ Not found")

echo "🧠 Training API: ${TRAINING_URL}"
echo "🔄 Mock API:     ${MOCK_URL}"
echo "📊 MLflow UI:    ${MLFLOW_URL}"

echo ""
echo "🔧 Next steps:"
echo "1. Test services:"
echo "   curl ${TRAINING_URL}/health"
echo "   curl ${MOCK_URL}/transactions"
echo "   open ${MLFLOW_URL}"
echo ""
echo "2. Update secrets with production URLs:"
echo "   echo '${TRAINING_URL}' | gcloud secrets versions add prod-api-url --data-file=-"
echo "   echo '${MLFLOW_URL}' | gcloud secrets versions add mlflow-tracking-uri --data-file=-"
echo ""
echo "📖 See deployment/README.md for complete post-deployment guide"