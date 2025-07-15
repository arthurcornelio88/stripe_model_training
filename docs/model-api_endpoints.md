# 📘 API Usage Guide – model-api (Fraud Detection)

This document explains how to interact with the FastAPI-powered `model-api` service deployed in the `model_training` project.

## 🌐 Base URLs

**Local Development**:
```
http://localhost:8000
```

**Production (Cloud Run)**:
```
https://mlops-training-api-bxzifydblq-ew.a.run.app
```

---

## 🔹 0. Launch API

### Local Development
```bash
docker compose up
```

### Production Deployment
```bash
cd deployment/
./deploy_training_only.sh
```

## 🔹 1. `/preprocess` — Clean and encode raw dataset

**Purpose:** Apply preprocessing logic to a raw CSV (including feature engineering + target encoding).

### 🔸 JSON Body

```json
{
  "input_path": "data/raw/fraudTest.csv",
  "output_dir": "data/processed",
  "log_amt": true
}
```

### ✅ Examples

**Local (cURL)**:
```bash
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d '{"input_path": "data/raw/fraudTest.csv", "output_dir": "data/processed", "log_amt": true}'
```

**Production (cURL)**:
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

---

## 🔹 2. `/train` — Train the ML model

**Purpose:** Run `train.py` logic on preprocessed files with a given timestamp.

### 🔸 JSON Body

```json
{
  "timestamp": "20250615_204742",
  "test": false,
  "learning_rate": 0.1,
  "epochs": 100
}
```

### ✅ Examples

**Local (cURL)**:
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "20250615_204742", "test": false, "learning_rate": 0.1, "epochs": 100}'
```

**Production (cURL)**:
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250615_204742",
    "test": false,
    "learning_rate": 0.1,
    "epochs": 100
  }' \
  --max-time 600
```

**Response**:
```json
{
  "status": "training complete",
  "mode": "full_train",
  "model_path": "gs://fraud-detection-jedha2024/models/catboost_model_20250615_204742.cbm",
  "metrics": {
    "auc": 0.9234,
    "f1_score": 0.8567,
    "precision": 0.8901,
    "recall": 0.8245
  }
}
```

---

## 🔹 3. `/validate` — Evaluate a trained model

**Purpose:** Measure model performance (F1, AUC, precision, etc) on test set.

### 🔸 JSON Body

```json
{
  "model_name": "catboost_model.cbm",
  "timestamp": "20250615_204742"
}
```

### ✅ Examples

**Local (cURL)**:
```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "catboost_model.cbm", "timestamp": "20250615_204742"}'
```

**Production (cURL)**:
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "catboost_model.cbm",
    "timestamp": "20250615_204742"
  }' \
  --max-time 300
```

> **Note:** Make sure to use the correct timestamp that matches your dated data files.

---

## 🔹 4. `/predict` — Run batch inference

**Purpose:** Predict fraud on a fully preprocessed CSV (e.g. `X_pred_*.csv`).

### 🔸 JSON Body

```json
{
  "input_path": "data/processed/X_pred_20250615_204742.csv",
  "model_name": "catboost_model.cbm",
  "output_path": "data/predictions.csv"
}
```

### ✅ Examples

**Local (cURL)**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_path": "data/processed/X_pred_20250615_204742.csv", "model_name": "catboost_model.cbm", "output_path": "data/predictions.csv"}'
```

**Production (cURL)**:
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250615_204742.csv",
    "model_name": "catboost_model.cbm",
    "output_path": "gs://fraud-detection-jedha2024/shared_data/predictions.csv"
  }' \
  --max-time 300
```

> **Note:** In local mode, predictions will be saved to `data/predictions.csv`.

---

## 🔹 5. `/monitor` — Compare reference and production data for drift

**Purpose:** Generate an HTML report and exploitable JSON on data drift between two batches. If data drift is detected, it will be signaled in JSON response like `"drift_summary":{"drift_detected":false}`

### 🔸 JSON Body

```json
{
  "reference_path": "data/processed/X_test_20250615_204742.csv",
  "current_path": "data/processed/X_pred_20250615_204742.csv",
  "output_html": "reports/data_drift.html"
}
```

### ✅ Examples

**Local (cURL)**:

```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "data/processed/X_test_20250615_204742.csv",
    "current_path": "data/processed/X_pred_20250615_204742.csv",
    "output_html": "reports/data_drift.html"
  }'
```

**Production (cURL)**:
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "gs://fraud-detection-jedha2024/shared_data/processed/X_test_20250615_204742.csv",
    "current_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250615_204742.csv",
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

---

## 🔹 Additional Endpoints

### Health Check

**Endpoint**: `GET /health` or `GET /ping`

**Examples**:

**Local**:
```bash
curl http://localhost:8000/health
```

**Production**:
```bash
curl https://mlops-training-api-bxzifydblq-ew.a.run.app/health
```

**Response**:
```json
{"status": "alive"}
```

### API Documentation

**Swagger UI**:
- **Local**: http://localhost:8000/docs
- **Production**: https://mlops-training-api-bxzifydblq-ew.a.run.app/docs

---

## 🚀 Production Deployment

### Deploy Training API Only
```bash
cd deployment/
./deploy_training_only.sh
```

### Deploy All Services
```bash
cd deployment/
./deploy_all_services.sh
```

### Monitor Logs
```bash
# View recent logs
gcloud run services logs read mlops-training-api --region=europe-west1

# Follow logs in real-time
gcloud run services logs tail mlops-training-api --region=europe-west1
```

---

## 🔧 Production URLs

- **Training API**: https://mlops-training-api-bxzifydblq-ew.a.run.app
- **MLflow UI**: https://mlops-mlflow-bxzifydblq-ew.a.run.app
- **Mock API**: https://mlops-mock-api-bxzifydblq-ew.a.run.app

---

## 📊 MLflow Integration

The training API is fully integrated with MLflow for experiment tracking:

- **Local MLflow**: http://localhost:5000
- **Production MLflow**: https://mlops-mlflow-bxzifydblq-ew.a.run.app

All training runs are automatically logged with:
- Model parameters (learning_rate, epochs, etc.)
- Performance metrics (AUC, F1, precision, recall)
- Model artifacts and predictions
- Environment information

---

*Last updated: July 15, 2025*

## 🔹 6. `/ping` — Health check

```bash
curl http://localhost:8000/ping
```

Response:

```json
{"status": "alive"}
```

---

> ✅ You can use Swagger UI as well: open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.