import pandas as pd
import numpy as np
import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from dotenv import load_dotenv
import os
import argparse
from glob import glob

# Load environment
load_dotenv()

ENV = os.getenv("ENV", "DEV")
BUCKET = os.getenv("GCP_BUCKET")
PREFIX = os.getenv("GCP_DATA_PREFIX")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
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


def log_mlflow(model, params, metrics):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    print(f"🔄 Logging to MLflow: {MLFLOW_URI} | Experiment: {EXPERIMENT}")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.catboost.log_model(model, artifact_path="catboost_model")


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
    else:
        params = {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 0,
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

    log_mlflow(model, params, metrics)
    save_model(model, model_name=args.model_name)

    print("✅ Training complete.")
    print(f"📊 AUC: {auc:.4f} | F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
