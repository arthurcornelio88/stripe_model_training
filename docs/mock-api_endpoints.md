# 📘 API Usage Guide – mock-api (Fraud Detection)

This document explains how to interact with the FastAPI-powered `mock-api` service deployed in the `model_training` project. Its main goal is to create mocked raw data for feeding the MLOps workflow for predictions and model retraining if data drift is detected.

All requests are made via HTTP POST to endpoints running on:

```
http://localhost:8001
```

---
## 🔹 0. Launch API

It launches `mock-api`, with all endpoints, for mock data creation.
```bash
docker compose up
```

## 🔹 1. `/transactions` — Create mock raw transactions 

**Purpose:** Create mocked data for new predictions and model retraining, if data drift detected in the MLOps workflow.

### 🔸 JSON Body

```json
{
  "n": 100 # (rows)
}
```

### ✅ Example (cURL)

```bash
curl -X GET "http://localhost:8001/transactions?n=100"
```