# 📘 API Usage Guide – model-api (Fraud Detection)

This document explains how to interact with the FastAPI-powered `model-api` service deployed in the `model_training` project.

All requests are made via HTTP POST to endpoints running on:

```
http://localhost:8000
```

---

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

### ✅ Example (cURL)

```bash
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d '{"input_path": "data/raw/fraudTest.csv", "output_dir": "data/processed", "log_amt": true}'
```

---

## 🔹 2. `/train` — Train the ML model

**Purpose:** Run `train.py` logic on preprocessed files with a given timestamp.

### 🔸 JSON Body

```json
{
  "timestamp": "20250615_204742",
  "test": false
}
```

### ✅ Example

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "20250615_204742", "test": false}'
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

### ✅ Example

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "catboost_model.cbm", "timestamp": "20250615_204742"}'
```

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

### ✅ Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_path": "data/processed/X_pred_20250615_204742.csv", "model_name": "catboost_model.cbm", "output_path": "data/predictions.csv"}'
```

---

## 🔹 5. `/monitor` — Compare reference and production data for drift

**Purpose:** Generate an HTML report on data drift between two batches.

### 🔸 JSON Body

```json
{
  "reference_path": "data/processed/X_test_20250615_204742.csv",
  "current_path": "data/processed/X_pred_20250616_153842.csv",
  "output_html": "reports/data_drift.html"
}
```

### ✅ Example

```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "data/processed/X_test_20250615_204742.csv",
    "current_path": "data/processed/X_pred_20250616_153842.csv",
    "output_html": "reports/data_drift.html"
  }'
```

---

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