# 🚀 MLOps Fraud Detection API - Complete Guide

## 📋 Table of Contents

1. [Configuration and Deployment](#configuration-and-deployment)
2. [Service URLs](#service-urls)
3. [Authentication](#authentication)
4. [API Endpoints](#api-endpoints)
5. [Usage Examples](#usage-examples)
6. [Monitoring and Debug](#monitoring-and-debug)
7. [Troubleshooting](#troubleshooting)

---

## 📦 Configuration and Deployment

### Development Mode (Local)
```bash
# Start all services locally
cd model_training/
docker compose up

# Available services:
# - Training API: http://localhost:8000
# - Mock data API: http://localhost:8000
# - MLflow UI: http://localhost:5000
```

### Production Mode (Cloud Run)
```bash
# Deploy all services
cd model_training/deployment/
./deploy_all_services.sh

# Or deploy only the training API
./deploy_training_only.sh
```

---

## 🌐 Service URLs

### 🔧 Development (Local)
```bash
# Training API and mock data streaming
TRAINING_API_URL="http://localhost:8000"

# MLflow UI
MLFLOW_UI_URL="http://localhost:5000"
```

### 🚀 Production (Cloud Run)
```bash
# Training API
TRAINING_API_URL="https://mlops-training-api-bxzifydblq-ew.a.run.app"

# Mock API (test data generation)
MOCK_API_URL="https://mlops-mock-api-bxzifydblq-ew.a.run.app"

# MLflow UI
MLFLOW_UI_URL="https://mlops-mlflow-bxzifydblq-ew.a.run.app"
```

---

## 🔐 Authentication

### Local
Depending on the services used, you'll need to retrieve and set up the .json files with permissions (google_credentials).

### Production
Cloud Run services are configured with `--allow-unauthenticated` to simplify access. For real production, you should configure IAM authentication.

---

## 🔌 API Endpoints

### 1. 🏥 Health Check

**Endpoint**: `GET /ping`
**Purpose**: Check if the API is operational

#### 💻 Local
```bash
curl http://localhost:8000/ping
```

#### ☁️ Production
```bash
curl https://mlops-training-api-bxzifydblq-ew.a.run.app/ping
```

**Response**:
```json
{"status": "alive"}
```

> 📸 **Screenshot needed**: Terminal output showing successful health check response

---

### 2. 🔄 Preprocessing

**Endpoint**: `POST /preprocess`
**Purpose**: Preprocess raw data for training

#### 📝 Request Body
```json
{
  "input_path": "data/raw/fraudTest.csv",
  "output_dir": "data/processed",
  "log_amt": true
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "data/raw/fraudTest.csv",
    "output_dir": "data/processed",
    "log_amt": true
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/fraudTest.csv",
    "output_dir": "gs://fraud-detection-jedha2024/shared_data/processed",
    "log_amt": true
  }' \
  --max-time 300
```

> 📸 **Screenshot needed**: Terminal output showing preprocessing progress and completion

---

### 3. 🤖 Training

**Endpoint**: `POST /train`
**Purpose**: Train a fraud detection model

#### 📝 Request Body
```json
{
  "timestamp": "20250715_195232", # corresponds to datetime of training and test datasets to use
  "learning_rate": 0.1,
  "epochs": 100,
  "test": false
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250715_195232",
    "learning_rate": 0.1,
    "epochs": 100,
    "test": false
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250715_195232",
    "learning_rate": 0.1,
    "epochs": 100,
    "test": false
  }' \
  --max-time 600
```

**Response**:
```json
{
  "status": "training complete",
  "mode": "full_train",
  "model_path": "gs://fraud-detection-jedha2024/models/catboost_model_20250715_195232.cbm",
  "metrics": {
    "auc": 0.9234,
    "f1_score": 0.8567,
    "precision": 0.8901,
    "recall": 0.8245
  }
}
```

> 📸 **Screenshot needed**: Terminal output showing training progress and final metrics

---

### 4. 🔍 Validation

**Endpoint**: `POST /validate`
**Purpose**: Evaluate model performance

#### 📝 Request Body
```json
{
  "model_name": "catboost_model_20250715_195232.cbm",
  "timestamp": "20250715_195232"
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "catboost_model_20250715_195232.cbm",
    "timestamp": "20250715_195232"
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "catboost_model_20250715_195232.cbm",
    "timestamp": "20250715_195232"
  }' \
  --max-time 300
```

> 📸 **Screenshot needed**: Terminal output showing validation metrics and results

---

### 5. 🔮 Prediction

**Endpoint**: `POST /predict`
**Purpose**: Make predictions on new data

#### 📝 Request Body
```json
{
  "input_path": "data/processed/X_pred_20250715_195232.csv",
  "model_name": "catboost_model_20250715_195232.cbm",
  "output_path": "data/predictions.csv"
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "data/processed/X_pred_20250715_195232.csv",
    "model_name": "catboost_model_20250715_195232.cbm",
    "output_path": "data/predictions.csv"
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250715_195232.csv",
    "model_name": "catboost_model_20250715_195232.cbm",
    "output_path": "gs://fraud-detection-jedha2024/shared_data/predictions.csv"
  }' \
  --max-time 300
```

> 📸 **Screenshot needed**: Terminal output showing prediction process and completion

---

### 6. 📊 Monitoring (Data Drift)

**Endpoint**: `POST /monitor`
**Purpose**: Detect data drift between two datasets

#### 📝 Request Body
```json
{
  "reference_path": "data/processed/X_test_20250715_195232.csv",
  "current_path": "data/processed/X_pred_20250715_195232.csv",
  "output_html": "reports/data_drift.html"
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "data/processed/X_test_20250715_195232.csv",
    "current_path": "data/processed/X_pred_20250715_195232.csv",
    "output_html": "reports/data_drift.html"
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "gs://fraud-detection-jedha2024/shared_data/processed/X_test_20250715_195232.csv",
    "current_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250715_195232.csv",
    "output_html": "gs://fraud-detection-jedha2024/shared_data/reports/data_drift.html"
  }' \
  --max-time 300
```

**Response**:
```json
{
  "drift_summary": {
    "drift_detected": false,
    "drift_score": 0.1234,
    "features_drifted": []
  },
  "report_path": "gs://fraud-detection-jedha2024/shared_data/reports/data_drift.html"
}
```

> 📸 **Screenshot needed**: Terminal output showing drift detection analysis and results

---

## 🎯 Usage Examples

### Complete Production Workflow

```bash
# 1. Check API health
curl https://mlops-training-api-bxzifydblq-ew.a.run.app/ping

# 2. Preprocess data
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/fraudTest.csv",
    "output_dir": "gs://fraud-detection-jedha2024/shared_data/processed",
    "log_amt": true
  }' \
  --max-time 300

# 3. Train the model
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250715_195232",
    "learning_rate": 0.1,
    "epochs": 50
  }' \
  --max-time 600

# 4. Make predictions
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250715_195232.csv",
    "model_name": "catboost_model_20250715_195232.cbm",
    "output_path": "gs://fraud-detection-jedha2024/shared_data/predictions.csv"
  }' \
  --max-time 300

# 5. Monitor data drift
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "gs://fraud-detection-jedha2024/shared_data/processed/X_test_20250715_195232.csv",
    "current_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250715_195232.csv",
    "output_html": "gs://fraud-detection-jedha2024/shared_data/reports/data_drift.html"
  }' \
  --max-time 300
```

> 📸 **Screenshot needed**: Terminal output showing complete workflow execution from health check to drift monitoring

---

## 📊 Monitoring and Debug

### Swagger Interface
- **Local**: http://localhost:8000/docs
- **Production**: https://mlops-training-api-bxzifydblq-ew.a.run.app/docs

> 📸 **Screenshot needed**: Swagger UI showing all available endpoints

### MLflow UI
- **Local**: http://localhost:5000
- **Production**: https://mlops-mlflow-bxzifydblq-ew.a.run.app

> 📸 **Screenshot needed**: MLflow UI showing experiment tracking and model metrics

### Log Verification
```bash
# Cloud Run logs
gcloud run services logs read mlops-training-api --region=europe-west1

# Real-time logs
gcloud run services logs tail mlops-training-api --region=europe-west1
```

### Debug Environment Variables
```bash
# Enable environment variable debug
gcloud run services update mlops-training-api \
  --set-env-vars DEBUG_ENV=true \
  --region=europe-west1
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Error 500 - Internal Server Error
```bash
# Check logs
gcloud run services logs read mlops-training-api --region=europe-west1 --limit=50

# Check environment variables
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/debug/env
```

#### 2. Training Timeout
```bash
# Increase request timeout
curl ... --max-time 900  # 15 minutes

# Reduce epochs for testing
{
  "epochs": 5,
  "learning_rate": 0.1
}
```

#### 3. MLflow Connection Issue
```bash
# Check MLflow service status
curl https://mlops-mlflow-bxzifydblq-ew.a.run.app/health

# Check secrets
gcloud secrets versions access latest --secret="mlflow-tracking-uri"
```

#### 4. GCS Data Access Issue
```bash
# Check service account permissions
gcloud projects get-iam-policy jedha2024 \
  --flatten="bindings[].members" \
  --format="value(bindings.role)" \
  --filter="bindings.members:mlops-service-account@jedha2024.iam.gserviceaccount.com"
```

### Diagnostic Commands

```bash
# Restart service
gcloud run services update mlops-training-api \
  --region=europe-west1 \
  --set-env-vars RESTART=$(date +%s)

# Check metrics
gcloud run services describe mlops-training-api \
  --region=europe-west1 \
  --format="value(status.url,status.latestReadyRevisionName)"

# Test connectivity
curl -I https://mlops-training-api-bxzifydblq-ew.a.run.app/ping
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [CatBoost Documentation](https://catboost.ai/docs/)

---

## 🆕 What's New

### Version 2.0 - January 2025
- ✅ **Centralized Environment Variable Management**
- ✅ **Production MLflow Connection Fixed**
- ✅ **Automatic Required Variable Validation**
- ✅ **Optimized Deployment Scripts**
- ✅ **Integrated Debug Mode with `DEBUG_ENV=true`**
- ✅ **Robust Fallbacks for Missing Variables**

### Migration from v1.0
API calls remain the same, but:
- Production URLs have changed
- Environment variables are now auto-validated
- MLflow now works correctly in production
- Better timeout handling

---

*Last updated: July 16, 2025*
