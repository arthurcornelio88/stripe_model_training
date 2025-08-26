# 🥷 Automatic Fraud Detection — MLOps Project

This project implements a full-stack pipeline for **credit card fraud detection**, designed to run both locally and in production. It includes preprocessing, model training and fine-tuning, validation, prediction, and drift monitoring — all orchestrated via APIs and optionally integrated with **Airflow** and **GCP (Cloud Run + GCS + Secret Manager)**.

---

## 🗂️ Project Structure

```bash
model_training/
├── src/                    # Core ML logic
│   ├── preprocessing.py
│   ├── train.py
│   ├── check_preproc_data.py
│   └── ...
├── model_training_api/     # FastAPI service for ML logic
├── mock_realtime_api/      # Fake transaction generator API
├── deployment/             # GCP deploy scripts
├── docker-compose.yml      # Local dev stack
└── docs/                   # Project documentation
```

---

## 🚀 Key Features

* Feature engineering with datetime & location
* Preprocessing pipeline (log scale, target encoding)
* CatBoost model with MLflow experiment tracking
* Unified FastAPI backend for all ML endpoints
* Cloud deployment via Docker + GCP (Cloud Run + GCS)
* Automated data drift detection

---

## ⚙️ How to Run

### 🔹 Locally

```bash
docker compose up --build
```

Access:

* API: [http://localhost:8000/docs](http://localhost:8000/docs)
* MLflow: [http://localhost:5000](http://localhost:5000)

### 🔹 In Production

Run:

```bash
bash deployment/deploy_all_services.sh
```

Requirements:

* GCP credentials
* `.env` with required variables
* Secrets created in GCP (see [`gcp_commands.md`](docs/gcp_commands.md))

---

## 📦 Main API Pipeline

| Step        | Endpoint        | Description                        |
| ----------- | --------------- | ---------------------------------- |
| 0. Health   | `/ping`         | Service health check               |
| 1. Data     | `/transactions` | Generate fake transaction samples  |
| 2. Prep     | `/preprocess`   | Preprocessing for training/predict |
| 3. Train    | `/train`        | Train or fine-tune a model         |
| 4. Validate | `/validate`     | Evaluate model on test data        |
| 5. Predict  | `/predict`      | Batch prediction                   |
| 6. Drift    | `/monitor`      | Drift detection on new data        |

See full usage examples in [📘 `api-production-guide.md`](docs/api-production-guide.md)

---

## 📚 Documentation

| Topic             | Doc File                                                  |
| ----------------- | --------------------------------------------------------- |
| Architecture      | [`architecture.md`](docs/architecture.md)                 |
| EDA Insights      | [`eda.md`](docs/eda.md)                                   |
| Preprocessing     | [`preprocessing.md`](docs/preprocessing.md)               |
| Training Pipeline | [`train.md`](docs/train.md)                               |
| API Guide (full)  | [`api-production-guide.md`](docs/api-production-guide.md) |
| GCP Deployment    | [`gcp_commands.md`](docs/gcp_commands.md)                 |
| Dev instructions  | [`instructions.md`](docs/instructions.md)                 |

---

## 🔧 Configuration & Secrets

* `.env` file for local development
* GCP Secret Manager for production values
* Deployment buckets and variables described in [`gcp_commands.md`](docs/gcp_commands.md)

---