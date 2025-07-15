#!/bin/bash
# deployment/deploy_training_only.sh

set -e

echo "🚀 Deploying MLOps Training API only..."

# Configuration
PROJECT_ID="jedha2024"
REGION="europe-west1"
BUCKET_NAME="fraud-detection-jedha2024"
TRAINING_IMAGE="gcr.io/${PROJECT_ID}/mlops-unified"

echo "📋 Project: $PROJECT_ID"
echo "📍 Region: $REGION"
echo "🗂️  Bucket: $BUCKET_NAME"

# Build Docker image
echo "📦 Building unified Docker image..."
echo "📋 Copying fraudTest.csv to build context..."
mkdir -p ../shared_data
cp ../../shared_data/fraudTest.csv ../shared_data/fraudTest.csv
ls -la ../shared_data/fraudTest.csv

# Build the image (from the model_training directory)
cd ..
docker build -t $TRAINING_IMAGE .
cd deployment

# Push to Google Container Registry
echo "📤 Pushing image to GCR..."
docker push $TRAINING_IMAGE

# Deploy Training API
echo "🧠 Deploying Training API..."
gcloud run deploy mlops-training-api \
  --image $TRAINING_IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 10 \
  --max-instances 3 \
  --set-env-vars ENV=PROD,SERVICE_TYPE=training,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCS_BUCKET=$BUCKET_NAME,MLFLOW_TRACKING_URI=$MLFLOW_URI \
  --service-account mlops-service-account@${PROJECT_ID}.iam.gserviceaccount.com

# Get the new URL
TRAINING_URL=$(gcloud run services describe mlops-training-api --region=$REGION --format="value(status.url)")

echo "✅ Training API deployed successfully!"
echo ""
echo "🌐 Service URL:"
echo "🧠 Training API: $TRAINING_URL"
echo ""
echo "🔧 Test the service:"
echo "curl $TRAINING_URL/health"
echo ""
echo "📖 Service is ready for training requests!"
