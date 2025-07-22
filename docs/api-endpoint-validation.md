# 🚀 MLOps API - Production Endpoint Documentation

This document outlines the main endpoints available in the `mlops-training-api` used for preprocessing, prediction, and model validation. All endpoints are deployed on **Cloud Run** and designed to interact with **GCS-based pipelines** for scalable fraud detection.

## 🌐 Base URL

```

[https://mlops-training-api-](https://mlops-training-api-)<your-project-suffix>.a.run.app

````

---

## 📦 `/preprocess`  
**Purpose**: Triggers preprocessing on a raw dataset stored in GCS or local.  
**Method**: `POST`

### 🔧 Request Body
```json
{
  "input_path": "gs://your-bucket-name/shared_data/raw/fraud_data.csv",
  "output_dir": "shared_data/preprocessed",
  "log_amt": true,
  "for_prediction": true
}
````

### ✅ Example Response

```json
{
  "status": "done",
  "timestamp": "20250721_153507"
}
```

---

## 📥 `/predict`

**Purpose**: Run inference on preprocessed data using a trained model stored in GCS.
**Method**: `POST`

### 🔧 Request Body

```json
{
  "input_path": "gs://your-bucket/shared_data/preprocessed/X_pred_<timestamp>.csv",
  "model_name": "catboost_model_<timestamp>.cbm",
  "output_path": "gs://your-bucket/shared_data/predictions/predictions_<timestamp>.csv"
}
```

### ✅ Example Response

```json
{
  "status": "prediction complete",
  "output": "gs://your-bucket/shared_data/predictions/predictions_20250721.csv"
}
```

---

## 🧪 `/validate`

**Purpose**: Validate a trained model on test data (historical mode) or predictions (production mode).
**Method**: `POST`

### 🔧 Historical Mode Request Body

```json
{
  "model_name": "catboost_model_20250715_210730.cbm",
  "timestamp": "20250721_153507"
}
```

### ✅ Example Response

```json
{
  "status": "model validated",
  "auc": 0.9271,
  "report_path": "gs://your-bucket/shared_data/reports/validation_report.json",
  "validation_type": "historical",
  "data_source": "local",
  "n_samples": 555719
}
```

### 🔧 Production Mode (optional alternative)

```json
{
  "validation_mode": "production",
  "production_data": {
    "y_true": [...],
    "y_pred_proba": [...],
    "y_pred_binary": [...]
  }
}
```

---

## 📊 `/monitor` *(Coming soon)*

**Purpose**: Monitor data drift between reference and current datasets using Evidently.

📍 *More details to be added in future release...*

---

## 🔐 Auth & Environment Notes

* ENV must be set to `PROD` with all secrets loaded from **Google Secret Manager**
* GCS files must be publicly accessible to the service account or authorized via workload identity
* MLflow logs only occur for **local files**; GCS paths will skip artifact logging

---

💬 For more information, contact the MLOps maintainer or refer to your Cloud Run logs.

```

---