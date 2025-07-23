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
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l us-central1 gs://your-actual-bucket-name

# Create folder structure
gsutil cp /dev/null gs://your-actual-bucket-name/models/.keep
gsutil cp /dev/null gs://your-actual-bucket-name/data/.keep
gsutil cp /dev/null gs://your-actual-bucket-name/artifacts/.keep
gsutil cp /dev/null gs://your-actual-bucket-name/reports/.keep
```

## 2. Create Google Secret Manager Secrets

Based on your .env file, create these secrets:

```bash
# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# Create secrets from your .env values
gcloud secrets create gcp-bucket --data-file=- <<< $GCP_BUCKET
gcloud secrets create gcp-project --data-file=- <<< $GCP_PROJECT
gcloud secrets create gcp-region --data-file=- <<< $GCP_REGION
gcloud secrets create gcp-data-prefix --data-file=- <<< $GCP_DATA_PREFIX
gcloud secrets create mlflow-tracking-uri --data-file=- <<< $MLFLOW_TRACKING_URI
gcloud secrets create mlflow-experiment --data-file=- <<< $MLFLOW_EXPERIMENT

# Additional secrets you might need for production
gcloud secrets create discord-webhook-url --data-file=- <<< "YOUR_DISCORD_WEBHOOK_URL"
gcloud secrets create prod-api-url --data-file=- <<< "YOUR_PRODUCTION_API_URL"
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

## Replace These Values:
- `YOUR_ACTUAL_PROJECT_ID` → Your Google Cloud Project ID
- `your-actual-bucket-name` → Your desired bucket name
- `YOUR_DISCORD_WEBHOOK_URL` → Your Discord webhook URL
- `YOUR_PRODUCTION_API_URL` → Your production API URL
