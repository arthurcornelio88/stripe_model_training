# 🔐 Google Cloud Commands for Fraud Detection MLOps

## Prerequisites
```bash
# Authenticate with GCP
gcloud auth login
gcloud auth application-default login

# Set your project (replace with your actual project ID)
gcloud config set project YOUR_ACTUAL_PROJECT_ID
```

## 1. Create GCS Bucket FIRST

```bash
# Enable Cloud Storage API
gcloud services enable storage.googleapis.com

# Create the bucket (replace with your actual bucket name)
# Valid EU regions: europe-west1, europe-west2, europe-west3, europe-west4, europe-west6, europe-central2
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l europe-west1 gs://your-actual-bucket-name

# OR if you prefer other regions:
# US regions: us-central1, us-east1, us-west1, us-west2
# Asia regions: asia-east1, asia-southeast1, asia-northeast1

# Create folder structure
gsutil cp /dev/null gs://$GCP_BUCKET/models/.keep
gsutil cp /dev/null gs://$GCP_BUCKET/shared_data/.keep
gsutil cp /dev/null gs://$GCP_BUCKET/artifacts/.keep
gsutil cp /dev/null gs://$GCP_BUCKET/reports/.keep
```

## 2. Create Google Secret Manager Secrets

Based on your .env file, create these secrets:

```bash
# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# In GCP Secrets, always PROD
gcloud secrets create env --data-file=- <<< "PROD"
# GCP Infos
gcloud secrets create gcp-bucket --data-file=- <<< $GCP_BUCKET
gcloud secrets create gcp-project --data-file=- <<< $GCP_PROJECT
gcloud secrets create gcp-region --data-file=- <<< $GCP_REGION
gcloud secrets create shared-data-path --data-file=- <<< $SHARED_DATA_PATH
gcloud secrets create model-path --data-file=- <<< $MODEL_PATH
# MLFLow
gcloud secrets create mlflow-tracking-uri --data-file=- <<< $MLFLOW_TRACKING_URI
gcloud secrets create mlflow-experiment --data-file=- <<< $MLFLOW_EXPERIMENT
# BigQuery variables
gcloud secrets create bq-project --data-file=- <<< $BQ_PROJECT
gcloud secrets create bq-raw-dataset --data-file=- <<< $BQ_RAW_DATASET
gcloud secrets create bq-predict-dataset --data-file=- <<< $BQ_PREDICT_DATASET
gcloud secrets create bq-location --data-file=- <<< $BQ_LOCATION
gcloud secrets create reset-bq-before-write --data-file=- <<< $RESET_BQ_BEFORE_WRITE
# For mock data
gcloud secrets create fetch-variability --data-file=- <<< $FETCH_VARIABILITY
# Original dataset
gcloud secrets create reference-data-path --data-file=- <<< $REFERENCE_DATA_PATH
# For monitoring
gcloud secrets create discord-webhook-url --data-file=- <<< $DISCORD_WEBHOOK_URL
# Princial URL API (see in GCloud Run after publishinh and running api image)
gcloud secrets create prod-api-url --data-file=- <<< $API_URL_PROD

```

## 3. Quick Commands to Update Secrets

```bash
# Update a secret value
echo "new-value" | gcloud secrets versions add SECRET_NAME --data-file=-

# View a secret (be careful, this shows the value!)
gcloud secrets versions access latest --secret="SECRET_NAME"

# List all secrets
gcloud secrets list
```

## 4. Quick Bucket Commands

```bash
# List bucket contents
gsutil ls gs://your-bucket-name

# Upload a file
gsutil cp local-file.txt gs://your-bucket-name/path/

# Download a file
gsutil cp gs://your-bucket-name/path/file.txt ./

# Sync a directory
gsutil -m rsync -r ./local-dir gs://your-bucket-name/remote-dir
```

## 5. Recommended .env Values

Create these environment variables in your `.env` file for local development:

### Model Training (.env values)
```bash
# Environment configuration
ENV=DEV
GCP_PROJECT=your-actual-project-id
GCP_REGION=europe-west1
GCP_BUCKET=your-actual-bucket-name

# Data paths (production values stored in secrets)
SHARED_DATA_PATH=/app/shared_data  # DEV: local, PROD: loaded from secret → "shared_data" 
MODEL_PATH=/app/models             # DEV: local, PROD: loaded from secret → "models"

# MLflow configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT=fraud_detection_experiment

# API URLs for production (use your actual endpoints)
API_URL_PROD=https://your-model-api-cloudrun-url.run.app
```

### DataOps (environment specific)
```bash
# BigQuery configuration (hardcoded in secrets)
BQ_PROJECT=jedha2024
BQ_RAW_DATASET=raw_api_data
BQ_PREDICT_DATASET=predictions
BQ_LOCATION=EU
RESET_BQ_BEFORE_WRITE=true
FETCH_VARIABILITY=0.1
REFERENCE_DATA_PATH=fraudTest.csv

# Discord webhook (replace with your actual webhook)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

## 6. Secret Values Explanation

| Secret Name | Description | DEV Value | PROD Value (in secret) |
|-------------|-------------|-----------|------------------------|
| `env` | Environment mode | `DEV` | `PROD` |
| `gcp-bucket` | GCS bucket name | Your bucket | Your bucket |
| `gcp-project` | GCP project ID | Your project | Your project |
| `gcp-region` | GCP region | `europe-west1` | `europe-west1` |
| `shared-data-path` | Data storage path | N/A (local) | `shared_data` |
| `model-path` | Model storage path | N/A (local) | `models` |
| `mlflow-tracking-uri` | MLflow server URL | Local | Cloud SQL/Redis |
| `mlflow-experiment` | Experiment name | Same | Same |

**Important**: In production, `SHARED_DATA_PATH` and `MODEL_PATH` are constructed by entrypoint.sh:
- `SHARED_DATA_PATH` → `gs://{GCS_BUCKET}/shared_data`
- `MODEL_PATH` → `gs://{GCS_BUCKET}/models`

The secret values (`shared_data`, `models`) are just the relative paths within the bucket.

## Replace These Values:
- `YOUR_ACTUAL_PROJECT_ID` → Your Google Cloud Project ID
- `your-actual-bucket-name` → Your desired bucket name
- `YOUR_DISCORD_WEBHOOK_URL` → Your Discord webhook URL
- `YOUR_PRODUCTION_API_URL` → Your production API URL

## 7. 🚀 Deployment Commands

### Prerequisites for Deployment

Before running the deployment script, ensure you have:

```bash
# 1. Docker installed and running
docker --version

# 2. Authenticated with GCP
gcloud auth login
gcloud auth configure-docker

# 3. Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 4. Set your project
gcloud config set project YOUR_ACTUAL_PROJECT_ID

# 5. Verify secrets are created (should show all secrets listed above)
gcloud secrets list
```

### Deploy All Services

```bash
# Navigate to the deployment directory
cd model_training/deployment/

# Make the script executable
chmod +x deploy_all_services.sh

# Run the deployment (script will build from ../Dockerfile)
./deploy_all_services.sh
```

**Important**: The script builds the Docker image from `model_training/Dockerfile`, not from the deployment directory. It also temporarily copies `../shared_data/fraudTest.csv` into the build context and cleans it up after the build.

### What the Script Deploys

The `deploy_all_services.sh` script will deploy 3 Cloud Run services:

| Service | Name | Purpose | Memory | CPU | Timeout |
|---------|------|---------|--------|-----|---------|
| **Model Training API** | `mlops-training-api` | Train/retrain fraud models | 2Gi | 2 CPU | 600s |
| **Mock Realtime API** | `mlops-mock-api` | Simulate transaction data | 1Gi | 1 CPU | 300s |
| **MLflow Tracking** | `mlops-mlflow` | Experiment tracking server | 1Gi | 1 CPU | - |

### Service URLs After Deployment

After successful deployment, you'll get URLs like:
```bash
# Training API
https://mlops-training-api-HASH-REGION.run.app

# Mock API
https://mlops-mock-api-HASH-REGION.run.app

# MLflow UI
https://mlops-mlflow-HASH-REGION.run.app
```

### Update Your .env with Production URLs

After deployment, update your local `.env` file with the actual URLs:

```bash
# Replace with your actual Cloud Run URLs
API_URL_PROD=https://mlops-training-api-HASH-europe-west1.run.app
MLFLOW_TRACKING_URI=https://mlops-mlflow-HASH-europe-west1.run.app
```

### Test Deployment

```bash
# Test Training API health
curl https://mlops-training-api-HASH-REGION.run.app/health

# Test Mock API
curl https://mlops-mock-api-HASH-REGION.run.app/transactions

# Access MLflow UI in browser
open https://mlops-mlflow-HASH-REGION.run.app
```

### Update Secrets with Production URLs

After deployment, update the secrets with actual URLs:

```bash
# Update production API URLs in secrets
echo "https://mlops-training-api-HASH-REGION.run.app" | gcloud secrets versions add prod-api-url --data-file=-
echo "https://mlops-mock-api-HASH-REGION.run.app" | gcloud secrets versions add prod-mock-url --data-file=-
echo "https://mlops-mlflow-HASH-REGION.run.app" | gcloud secrets versions add mlflow-tracking-uri --data-file=-
```

### Troubleshooting Deployment

```bash
# Check Cloud Run service logs
gcloud run services logs read mlops-training-api --region=europe-west1

# Check service status
gcloud run services list --region=europe-west1

# Redeploy a single service if needed
gcloud run deploy mlops-training-api \
  --image gcr.io/YOUR_PROJECT/mlops-api \
  --region europe-west1

# Check container build logs
gcloud builds list --limit=5
```

### Manual Service Management

```bash
# Stop all services (to save costs during development)
gcloud run services delete mlops-training-api --region=europe-west1 --quiet
gcloud run services delete mlops-mock-api --region=europe-west1 --quiet
gcloud run services delete mlops-mlflow --region=europe-west1 --quiet

# Scale down to zero (services remain but don't consume resources)
gcloud run services update mlops-training-api --region=europe-west1 --min-instances=0
gcloud run services update mlops-mock-api --region=europe-west1 --min-instances=0
gcloud run services update mlops-mlflow --region=europe-west1 --min-instances=0
```
