
# 🧠 `train.py` — CatBoost Fraud Detection Trainer

This script trains a **CatBoostClassifier** on preprocessed data to detect fraud. It is designed to work both locally (DEV) and in production (PROD) with full support for **versioning, MLflow, and GCS**.

---


## 🧰 Main Features

✅ Automatic environment management (local or GCP)
✅ Loading of versioned data by timestamp
✅ 3 training modes (`full`, `--fast`, `--test`)
✅ Full logging in **MLflow** (params, metrics, artifacts)
✅ Model registration in the **MLflow Registry**

---


## ⚙️ Prerequisites

### Local (`ENV=DEV`)

Minimal required `.env`:

```env
ENV=DEV
MLFLOW_TRACKING_URI=http://localhost:5000
```

Start the local MLflow server:

```bash
mlflow server \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000 \
  --serve-artifacts
```

### Production (`ENV=PROD`)

Production `.env` (secrets must exist on GCP):

```env
ENV=PROD
GCP_BUCKET=my-bucket-name
GCP_DATA_PREFIX=data/fraud-detection
MLFLOW_TRACKING_URI=https://mlflow.mycompany.com
```

Make sure the GCP service has the correct permissions (`GOOGLE_APPLICATION_CREDENTIALS` configured).

---


## 🚀 Launch a Training Run

```bash
python src/train.py [OPTIONS]
```

### Available Options:

| Option            | Description                                   | Default              |
| ----------------- | --------------------------------------------- | -------------------- |
| `--iterations`    | Number of CatBoost iterations                 | `500`                |
| `--learning_rate` | Learning rate                                 | `0.05`               |
| `--depth`         | Maximum tree depth                            | `6`                  |
| `--model_name`    | Output model filename                         | `catboost_model.cbm` |
| `--timestamp`     | Use a specific data version                   | *(most recent)*      |
| `--test`          | Fast test mode (10 iterations)                | `False`              |
| `--fast`          | Fast dev mode (fewer iterations than prod)    | `False`              |

---


## ⚡ Training Modes

| Mode | Command        | Purpose                            | Simplified Config |
| ---- | -------------- | ---------------------------------- | ----------------- |
| PROD | *(default)*    | Best performance, full logging     | ❌                 |
| FAST | `--fast`       | Fewer iterations, quick logging    | ✅                 |
| TEST | `--test`       | Ultra-fast for pipeline validation | ✅                 |

---


## 🧪 Examples

Train with the most recent data:

```bash
python train.py
```

Train with a specific data version:

```bash
python train.py --timestamp 20240721_1800
```

Fast mode:

```bash
python train.py --fast
```

Test mode:

```bash
python train.py --test
```

Custom model name:

```bash
python train.py --model_name catboost_fraud_v1.cbm
```

---


## 📤 Generated Artifacts

| Type                   | Location                                                   |
| ---------------------- | ---------------------------------------------------------- |
| Trained model          | `models/<model_name>.cbm` (or GCS bucket if `ENV=PROD`)    |
| Training logs          | MLflow (`params`, `metrics`, `artifacts`, `model`)         |
| Classification report  | `reports/classification_report.json` + `.html` in MLflow   |
| Model registration     | MLflow Registry (`CatBoostFraudDetector`)                  |

---


## 🧠 What the Script Does

1. 🔍 Loads preprocessed datasets (by timestamp or latest available)
2. 🏋️ Trains a `CatBoostClassifier` (config depends on mode)
3. 📈 Evaluates the model (`AUC`, `F1`, etc.)
4. 📝 Logs to MLflow:

   * Hyperparameters
   * Metrics
   * HTML/JSON report
   * Model (MLflow + Registry)
5. 💾 Saves the model locally or to GCS

---