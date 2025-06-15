# 🧱 Modular Architecture — Fraud Detection (Block 3)

---

## 📦 `model_training/` — ML API Service

This repository contains all machine learning logic (preprocessing, training, batch inference, validation, drift monitoring) exposed via a unified FastAPI service.

### 🔌 Exposed Endpoints (FastAPI)

| Endpoint      | Method | Description                               |
| ------------- | ------ | ----------------------------------------- |
| `/ping`       | GET    | Liveness check                            |
| `/preprocess` | POST   | Cleans and encodes a batch of raw data    |
| `/train`      | POST   | Triggers model training                   |
| `/validate`   | POST   | Evaluates model performance (on test set) |
| `/predict`    | POST   | Performs batch inference                  |
| `/monitor`    | POST   | Compares two datasets to detect drift     |

➡️ Deployed as a container named `model-api` in `docker-compose`.

---

## 📦 `mock_realtime_api/` — Synthetic Transaction Generator

A second FastAPI service that generates realistic mock transactions, replicating the raw dataset structure.

### 🔌 Endpoint

| Endpoint        | Method | Description                                             |
| --------------- | ------ | ------------------------------------------------------- |
| `/transactions` | GET    | Returns a list of fake credit card transactions in JSON |

➡️ Deployed as a container named `mock-api` in `docker-compose`.

---

## 📦 `dataops/` — Airflow Orchestration (separate repo)

This repo manages the **data pipelines** (not ML logic). Contains:

* `dags/` — ingestion, preprocessing, batch prediction, drift validation
* `docker-compose-airflow.yml` — local orchestration setup

> Airflow never directly calls the ML code. It only interacts with the `model-api` via HTTP:

```python
# Airflow DAG example
requests.post("http://model-api:8000/predict", json={...})
```

---

## 📂 Folder Breakdown in `model_training/`

```
model_training/
├── model_training_api/         ← FastAPI logic (main.py with all endpoints)
│   └── main.py
├── mock_realtime_api/          ← Mock transaction generator
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── src/                        ← Core ML modules
│   ├── train.py
│   ├── predict.py
│   ├── validate_model.py
│   ├── preprocessing.py
├── Dockerfile                  ← for model-api container
├── docker-compose.yml          ← runs model-api + mock-api
└── pyproject.toml / uv.lock    ← dependencies managed with `uv`
```

---

## ⚙️ Dev vs Prod Behavior

| Component     | Development (Local)         | Production (Cloud / Orchestrated)           |
| ------------- | --------------------------- | ------------------------------------------- |
| Data Source   | Local `data/` folder        | GCS Bucket via `gcsfs`                      |
| Model Storage | Saved to `models/` locally  | Stored as MLflow artifact in GCS            |
| Ingestion     | Pulled from `mock-api`      | Comes from live system / webhook            |
| Preprocessing | Triggered via `/preprocess` | Automated in Airflow pipeline               |
| Training      | Via CLI or `/train`         | Triggered via API (possibly retraining DAG) |
| Prediction    | `/predict` with local model | `/predict` using model from GCS + MLflow    |

---

## ✅ Recap

| Component             | Role                                 |
| --------------------- | ------------------------------------ |
| `model-api` container | Hosts all ML logic + endpoints       |
| `mock-api` container  | Simulates real-time fraud data       |
| `dataops` + Airflow   | Pipeline orchestration (external)    |
| FastAPI everywhere    | Makes it testable, scalable, unified |
