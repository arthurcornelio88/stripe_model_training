# 🚀 MLOps Deployment Guide

## Quick Start

```bash
# 1. Ensure you're in the right directory
cd model_training/deployment/

# 2. Run the deployment script (builds from ../Dockerfile)
./deploy_all_services.sh
```

**Note**: The deployment script automatically:
1. Navigates to `../` to build from the main `Dockerfile`
2. Temporarily copies `../../shared_data/fraudTest.csv` into the build context
3. Builds the Docker image
4. Cleans up temporary files
5. Returns to the deployment directory

## Prerequisites Checklist

- [ ] Docker installed and running
- [ ] GCP CLI authenticated (`gcloud auth login`)
- [ ] Docker registry authenticated (`gcloud auth configure-docker`)
- [ ] All secrets created (see `../../gcp_commands.md`)
- [ ] Required APIs enabled
- [ ] GCS bucket created

## What Gets Deployed

The script deploys 3 unified Cloud Run services from the same Docker image:

### 🧠 Model Training API (`mlops-training-api`)
- **Purpose**: Train/retrain fraud detection models
- **Resources**: 2GB RAM, 2 CPU cores
- **Timeout**: 10 minutes
- **Endpoints**:
  - `POST /train` - Train new model
  - `POST /fine-tune` - Fine-tune existing model
  - `POST /preprocess` - Preprocess data
  - `POST /predict` - Make predictions
  - `GET /health` - Health check

### 🔄 Mock Realtime API (`mlops-mock-api`)
- **Purpose**: Simulate real transaction data for testing
- **Resources**: 1GB RAM, 1 CPU core
- **Timeout**: 5 minutes
- **Endpoints**:
  - `GET /transactions` - Generate mock transactions
  - `GET /health` - Health check

### 📊 MLflow Tracking Server (`mlops-mlflow`)
- **Purpose**: Experiment tracking and model registry
- **Resources**: 1GB RAM, 1 CPU core
- **Port**: 8080
- **Interface**: Web UI for model tracking

## Environment Variables Set

Each service gets these environment variables:

```bash
ENV=PROD                           # Production mode
SERVICE_TYPE=training|mock|mlflow  # Service-specific behavior
GOOGLE_CLOUD_PROJECT=your-project  # For secret access
```

## Post-Deployment Steps

### 1. Get Service URLs

```bash
# List all deployed services
gcloud run services list --region=europe-west1

# Get specific service URL
gcloud run services describe mlops-training-api --region=europe-west1 --format='value(status.url)'
```

### 2. Test Services

```bash
# Test training API
curl -X GET "$(gcloud run services describe mlops-training-api --region=europe-west1 --format='value(status.url)')/health"

# Test mock API
curl -X GET "$(gcloud run services describe mlops-mock-api --region=europe-west1 --format='value(status.url)')/transactions"

# Access MLflow UI (open in browser)
gcloud run services describe mlops-mlflow --region=europe-west1 --format='value(status.url)'
```

### 3. Update Secrets with Production URLs

```bash
# Get the actual URLs
TRAINING_URL=$(gcloud run services describe mlops-training-api --region=europe-west1 --format='value(status.url)')
MOCK_URL=$(gcloud run services describe mlops-mock-api --region=europe-west1 --format='value(status.url)')
MLFLOW_URL=$(gcloud run services describe mlops-mlflow --region=europe-west1 --format='value(status.url)')

# Update secrets
echo "${TRAINING_URL}" | gcloud secrets versions add prod-api-url --data-file=-
echo "${TRAINING_URL}/preprocess" | gcloud secrets versions add preprocess-endpoint --data-file=-
echo "${TRAINING_URL}/predict" | gcloud secrets versions add predict-url-prod --data-file=-
echo "${TRAINING_URL}/monitor" | gcloud secrets versions add monitor-url-prod --data-file=-
echo "${MOCK_URL}/transactions" | gcloud secrets versions add transaction-url-prod --data-file=-
echo "${MLFLOW_URL}" | gcloud secrets versions add mlflow-tracking-uri --data-file=-
```

## Troubleshooting

### Check Logs

```bash
# Real-time logs
gcloud run services logs tail mlops-training-api --region=europe-west1

# Recent logs
gcloud run services logs read mlops-training-api --region=europe-west1 --limit=50
```

### Common Issues

#### 1. **Build Fails**
```bash
# Check if you're in the right directory
pwd  # Should be: .../model_training/deployment/

# Check Dockerfile exists in parent directory
ls -la ../Dockerfile

# Manual build test
cd .. && docker build -t test-image . && cd deployment
```

#### 2. **Secret Access Errors**
```bash
# Verify secrets exist
gcloud secrets list

# Test secret access
gcloud secrets versions access latest --secret="gcp-bucket"
```

#### 3. **Service Won't Start**
```bash
# Check service configuration
gcloud run services describe mlops-training-api --region=europe-west1

# Check container logs
gcloud run services logs read mlops-training-api --region=europe-west1
```

#### 4. **Memory/Timeout Issues**
```bash
# Increase resources if needed
gcloud run services update mlops-training-api \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --region=europe-west1
```

## Cost Management

### Stop Services (Delete)
```bash
gcloud run services delete mlops-training-api --region=europe-west1 --quiet
gcloud run services delete mlops-mock-api --region=europe-west1 --quiet
gcloud run services delete mlops-mlflow --region=europe-west1 --quiet
```

### Scale to Zero (Keep Services, No Cost)
```bash
gcloud run services update mlops-training-api --region=europe-west1 --min-instances=0
gcloud run services update mlops-mock-api --region=europe-west1 --min-instances=0
gcloud run services update mlops-mlflow --region=europe-west1 --min-instances=0
```

## Next Steps

After successful deployment:

1. **Update Airflow DAGs** with production URLs
2. **Configure monitoring** and alerting
3. **Set up CI/CD** for automatic deployments
4. **Test end-to-end workflow** with real data
5. **Monitor costs** in GCP Console

## Support

- **Logs**: `gcloud run services logs read SERVICE_NAME --region=europe-west1`
- **Status**: `gcloud run services list --region=europe-west1`
- **Configuration**: Check `entrypoint.sh` and `secret_manager.py`
- **Documentation**: See `../../gcp_commands.md` for complete setup guide

## 📚 Documentation

### API Usage Guides
- **[📘 Complete API Production Guide](../docs/api-production-guide.md)** - Comprehensive guide with local and production examples
- **[📘 Model API Endpoints](../docs/model-api_endpoints.md)** - Updated endpoint documentation with production URLs
- **[📊 MLflow Integration](../docs/)** - MLflow tracking and experiment management

### Quick Reference
- **Training API**: https://mlops-training-api-bxzifydblq-ew.a.run.app
- **MLflow UI**: https://mlops-mlflow-bxzifydblq-ew.a.run.app
- **Mock API**: https://mlops-mock-api-bxzifydblq-ew.a.run.app

### API Testing
```bash
# Test production API health
curl https://mlops-training-api-bxzifydblq-ew.a.run.app/health

# Test training endpoint
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "20250715_195232", "learning_rate": 0.1, "epochs": 10}' \
  --max-time 600
```
