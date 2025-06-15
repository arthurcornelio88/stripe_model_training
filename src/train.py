import pandas as pd
import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from dotenv import load_dotenv
import os
import argparse
from glob import glob
from mlflow.tracking import MlflowClient
import requests
from urllib.parse import urljoin
import json
import shutil



# Load environment
load_dotenv()

ENV = os.getenv("ENV", "DEV")
BUCKET = os.getenv("GCP_BUCKET")
PREFIX = os.getenv("GCP_DATA_PREFIX")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI").strip("/")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")


def gcs_path(filename):
    return f"gs://{BUCKET}/{PREFIX}/{filename}"


def get_latest_file(pattern):
    """
    Return the most recently modified file matching the pattern.
    """
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def resolve_path(name, io="input", timestamp=None):
    """
    Resolve the correct path depending on ENV and optional timestamp.

    - DEV: loads latest or specific timestamped CSV from local folder
    - PROD: uses GCS fixed path
    """
    if ENV == "PROD":
        return gcs_path(f"processed/{name}")

    # DEV
    base_dir = "data/raw/" if io == "input" else "data/processed/"
    if timestamp:
        filename = name.replace(".csv", f"_{timestamp}.csv")
        return os.path.join(base_dir, filename)
    else:
        pattern = os.path.join(base_dir, name.replace(".csv", "_*.csv"))
        return get_latest_file(pattern)


def load_data(timestamp=None, test_mode=False, sample_size=5000):
    """
    Load and optionally subsample preprocessed data.

    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"🔄 ENV = {ENV} | Loading data...")

    def read(name):
        path = resolve_path(name, io="output", timestamp=timestamp)
        print(f"🔄 Resolving latest path for {name}: {path}")
        return pd.read_csv(path)

    X_train = read("X_train.csv")
    X_test = read("X_test.csv")
    y_train = read("y_train.csv").squeeze()
    y_test = read("y_test.csv").squeeze()

    print(f"✅ Data loaded: {len(X_train)} train samples, {len(X_test)} test samples")

    if test_mode:
        print(f"⚡ Sampling {sample_size} rows for fast testing")
        X_train = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)
        X_test = X_test.sample(n=min(sample_size // 4, len(X_test)), random_state=42)
        y_train = y_train.loc[X_train.index]
        y_test = y_test.loc[X_test.index]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_val, y_val, params):
    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=100,
        early_stopping_rounds=50
    )
    print("🔄 Training CatBoost model...")
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probas)
    print("🔄 Evaluating model...")
    return report, auc

def check_mlflow_server(mlflow_uri, experiment_name="default"):
    """
    Ping MLflow server with a safe GET to verify availability.
    Uses a valid API endpoint to avoid 404s.
    """
    try:
        uri = mlflow_uri.rstrip("/") + "/"
        health_url = urljoin(uri, f"api/2.0/mlflow/experiments/get-by-name?experiment_name={experiment_name}")
        response = requests.get(health_url, timeout=3)
        if response.status_code not in (200, 404):  # 404 here may still be valid (experiment not found)
            raise RuntimeError(f"MLflow server responded with {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"❌ Could not connect to MLflow server at {mlflow_uri}: {e}")

def save_and_log_report(report_dict, run_id, output_dir="reports"):
    """
    Sauvegarde le rapport au format JSON/HTML, puis l'upload dans MLflow.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "classification_report.json")
    html_path = os.path.join(output_dir, "classification_report.html")

    # Enregistrement local
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    html_content = "<html><head><title>Classification Report</title></head><body>"
    html_content += "<h2>Classification Report</h2><table border='1'>"
    html_content += "<tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>"

    for label, scores in report_dict.items():
        if isinstance(scores, dict):
            html_content += f"<tr><td>{label}</td><td>{scores.get('precision', 0):.4f}</td>"
            html_content += f"<td>{scores.get('recall', 0):.4f}</td><td>{scores.get('f1-score', 0):.4f}</td>"
            html_content += f"<td>{scores.get('support', 0):.0f}</td></tr>"

    html_content += "</table></body></html>"

    with open(html_path, "w") as f:
        f.write(html_content)

    # Upload dans MLflow (dans artifacts/reports)
    mlflow.log_artifact(json_path, artifact_path="reports")
    mlflow.log_artifact(html_path, artifact_path="reports")

    shutil.rmtree(output_dir)



def log_mlflow(model, params, metrics, report):
    """
    Log model, parameters and metrics to MLflow.
    Handles experiment creation and safe logging based on ENV.
    """
    print("🔄 Setting up MLflow logging...")

    experiment_name = EXPERIMENT or "Fraud Detection CatBoost"
    check_mlflow_server(MLFLOW_URI, experiment_name=experiment_name)

    mlflow.set_tracking_uri(MLFLOW_URI)

    if ENV == "PROD":
        artifact_location = f"gs://{BUCKET}/mlflow-artifacts"
    else:
        artifact_location = "file:./mlruns"

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(name=experiment_name, artifact_location=artifact_location)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    print(f"🔄 Logging to MLflow: {MLFLOW_URI} | Experiment: {experiment_name}")

    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.catboost.log_model(model, artifact_path="model")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name="CatBoostFraudDetector"
        )

        save_and_log_report(report, mlflow.active_run().info.run_id)



def save_model(model, model_name="catboost_model.cbm"):
    output_path = (
        gcs_path(f"models/{model_name}") if ENV == "PROD"
        else os.path.join("models", model_name)
    )
    if ENV == "DEV":
        os.makedirs("models", exist_ok=True)
    model.save_model(output_path)
    print(f"💾 Model saved to: {output_path}")


def main():
    """
    Train a CatBoostClassifier for fraud detection.

    Reads data from GCS or local disk depending on ENV.
    Supports timestamp-based data versioning.
    Logs training to MLflow and saves model artifact.

    CLI:
    --iterations     Number of boosting rounds
    --learning_rate  Learning rate
    --depth          Tree depth
    --model_name     Output filename for the model
    --test           Run fast training with small params
    --timestamp      Timestamp (YYYYmmdd_HHMMSS) to target a specific dataset version
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--model_name", type=str, default="catboost_model.cbm")
    parser.add_argument("--test", action="store_true", help="Use minimal params for fast testing")
    parser.add_argument("--fast", action="store_true", help="Run in fast dev mode (not full test, not full prod)")
    parser.add_argument("--timestamp", type=str, help="Timestamp to load specific preprocessed data")

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(timestamp=args.timestamp,test_mode=args.test)

    if args.test:
        print("⚡️ Running in TEST mode: minimal CatBoost config")
        params = {
            "iterations": 10,
            "learning_rate": 0.1,
            "depth": 3,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 0,
            "random_seed": 42,
            "class_weights": [1, 10]
        }

    elif args.fast:
        print("🚀 Running in FAST DEV mode: semi-prod CatBoost config")
        params = {
            "iterations": 150,
            "learning_rate": 0.07,
            "depth": 5,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 100,
            "random_seed": 42,
            "class_weights": [1, 15]
        }
    else:
        print("🏗️ Running in FULL PROD mode: full CatBoost config")
        params = {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 100,
            "random_seed": 42,
            "class_weights": [1, 25]
        }

    model = train_model(X_train, y_train, X_test, y_test, params)
    report, auc = evaluate_model(model, X_test, y_test)

    metrics = {
        "roc_auc": auc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    }

    log_mlflow(model, params, metrics, report)
    save_model(model, model_name=args.model_name)

    print("✅ Training complete.")
    print(f"📊 AUC: {auc:.4f} | F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
