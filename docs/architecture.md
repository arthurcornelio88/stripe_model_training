# 🧱 Modular Architecture — Fraud Detection (Block 3)

---

## 📦 `model_training/` — ML API Service

Contient toute la logique Machine Learning exposée via FastAPI :

* preprocessing
* entraînement & fine-tuning
* batch prediction
* validation
* drift monitoring

### 🔌 FastAPI Endpoints

| Endpoint      | Method | Description                     |
| ------------- | ------ | ------------------------------- |
| `/ping`       | GET    | Health check                    |
| `/preprocess` | POST   | Preprocess raw transaction data |
| `/train`      | POST   | Train or fine-tune a model      |
| `/validate`   | POST   | Evaluate a trained model        |
| `/predict`    | POST   | Run batch prediction            |
| `/monitor`    | POST   | Detect dataset drift            |

➡️ Déployé dans le conteneur `model-api` (Docker / Cloud Run).

---

## 📦 `mock_realtime_api/` — Synthetic Transaction Generator

Expose une API FastAPI qui génère des transactions synthétiques similaires au dataset réel.

### 🔌 Endpoint

| Endpoint        | Method | Description                      |
| --------------- | ------ | -------------------------------- |
| `/transactions` | GET    | Returns fake transactions (JSON) |

➡️ Déployé dans le conteneur `mock-api`.

---

## 📦 `dataops/` — Airflow Orchestration (separate repo)

Ce dépôt gère l’orchestration des pipelines de données avec Apache Airflow.

* `dags/`: ingestion, preprocessing, prediction, drift monitoring
* `docker-compose-airflow.yml`: setup local

> 💡 Airflow n'appelle jamais directement le code ML. Il utilise l’API HTTP exposée par `model-api`.

```python
requests.post("http://model-api:8000/predict", json={...})
```

---

## 📁 Repo Breakdown — `model_training/`

```
model_training/
├── model_training_api/       ← FastAPI backend (routes, utils)
├── mock_realtime_api/        ← Fake transaction generator
├── src/                      ← ML logic (training, preprocessing, etc.)
│   ├── train.py
│   ├── validate_model.py
│   ├── predict.py
│   ├── preprocessing.py
├── Dockerfile                ← model-api container
├── docker-compose.yml        ← runs model-api + mock-api
└── pyproject.toml / uv.lock  ← dependencies
```

---

## ⚙️ DEV vs PROD Comparison

| Feature        | DEV Mode                 | PROD Mode (Cloud)                       |
| -------------- | ------------------------ | --------------------------------------- |
| Data Source    | Local `/app/shared_data` | GCS Bucket (via `gcsfs`)                |
| Model Storage  | Local `models/`          | GCS + MLflow Artifacts                  |
| Triggering     | Manual (CLI or curl)     | Triggered via API calls or Airflow DAGs |
| Prediction     | Model from local folder  | Model loaded via MLflow + GCS           |
| Authentication | Local credentials file   | IAM & GCP service accounts              |

---

## ✅ System Recap

| Component   | Role                                  |
| ----------- | ------------------------------------- |
| `model-api` | Expose ML endpoints (FastAPI)         |
| `mock-api`  | Generate fake transactions            |
| `mlflow`    | Track experiments and store artifacts |
| `dataops`   | Orchestrate pipeline via DAGs         |
| FastAPI     | Enables unified, testable interfaces  |

---