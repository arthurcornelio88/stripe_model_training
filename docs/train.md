# 🧠 `train.py` — CatBoost Fraud Detection Trainer

This script trains a **CatBoostClassifier** on preprocessed fraud detection data, using environment-aware logic to handle:

* Local (DEV) or Cloud (PROD) data access
* Versioned datasets via timestamped CSVs
* Fast prototyping (`--test`)
* Full MLflow experiment tracking

---

## ✅ Before Running the Script

To track training runs and manage model versions, **MLflow** must be properly configured depending on your environment:

### 🔧 DEV Mode

Ensure your `.env` file includes:

```env
ENV=DEV
MLFLOW_TRACKING_URI=http://localhost:5000
```

This will:

* Track experiments locally at `http://localhost:5000`
* Save artifacts to the local `mlruns/` folder
* Store trained models in `models/`

To spin up a local MLflow server:

```bash
mlflow server \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000 \
  --serve-artifacts
```

---

### ☁️ PROD Mode

In production, data and models are stored in Google Cloud Storage and tracked on a remote MLflow server. Set your `.env` like this:

```env
ENV=PROD
GCP_BUCKET=my-bucket-name
GCP_DATA_PREFIX=data/fraud-detection
MLFLOW_TRACKING_URI=https://mlflow.mycompany.com
```

This will:

* Load input datasets from `gs://my-bucket-name/data/fraud-detection/processed/`
* Save models to `gs://my-bucket-name/models/`
* Track MLflow runs at the provided remote URI

> ⚠️ Make sure your environment has the right permissions (e.g. Google service account credentials via `GOOGLE_APPLICATION_CREDENTIALS`).

---

## 🚀 Usage

```bash
python src/train.py [options]
```

### Available CLI options:

| Option            | Description                                                    | Default                  |
| ----------------- | -------------------------------------------------------------- | ------------------------ |
| `--iterations`    | Number of boosting rounds                                      | `500`                    |
| `--learning_rate` | Learning rate for CatBoost                                     | `0.05`                   |
| `--depth`         | Tree depth                                                     | `6`                      |
| `--model_name`    | Name of the output model file                                  | `catboost_model.cbm`     |
| `--timestamp`     | Use a specific timestamped dataset (format: `YYYYmmdd_HHMMSS`) | *(use latest available)* |
| `--test`          | Enable fast training mode with minimal config                  | `False`                  |
| `--fast`          | Run in fast dev mode (not full test, not full prod)            | `False`                  |

---

## 🛠️ Behavior by Environment

The script adapts based on `ENV` defined in your `.env` file:

### 🔧 DEV Mode (`ENV=DEV`)

* Loads CSVs from `data/processed/X_train_<timestamp>.csv` (or latest version)
* Saves model locally in `models/`
* Uses local MLflow tracking URI

```env
ENV=DEV
MLFLOW_TRACKING_URI=http://localhost:5000
```

### ☁️ PROD Mode (`ENV=PROD`)

* Reads datasets from Google Cloud Storage bucket
* Logs experiment to remote MLflow server
* Saves model in `gs://<bucket>/models/`

```env
ENV=PROD
GCP_BUCKET=my-bucket-name
GCP_DATA_PREFIX=data/fraud-detection
MLFLOW_TRACKING_URI=https://mlflow.mycompany.com
```

---

## ⚡ Fast Training Mode

Use the `--test` flag to quickly validate the pipeline with minimal config:

```bash
python train.py --test
```

Uses:

```python
iterations = 10
learning_rate = 0.1
depth = 3
class_weights = [1, 10]
```

---

## 🧪 Example Runs

Train latest version:

```bash
python train.py
```

Train specific version:

```bash
python train.py --timestamp 20240611_1542
```

Train with custom model name:

```bash
python train.py --model_name fraud_v1.cbm
```

Train in test mode:

```bash
python train.py --test
```

---

## 📁 Output

| Artifact       | Destination                               |
| -------------- | ----------------------------------------- |
| Trained Model  | `models/<model_name>.cbm` (or GCS bucket) |
| MLflow Logs    | Tracked at `MLFLOW_TRACKING_URI`          |
| Dataset Inputs | Loaded from local or GCS (preprocessed)   |

---

## 🧩 Next Steps

* Add API serving: `serve.py`
* Automate version tagging
* Upload evaluation report to MLflow artifacts
