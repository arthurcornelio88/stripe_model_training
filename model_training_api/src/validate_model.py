import os
import argparse
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from src.train import load_data
from dotenv import load_dotenv
import mlflow
import json
import shutil
import gcsfs

load_dotenv()
ENV = os.getenv("ENV", "DEV")
BUCKET = os.getenv("GCP_BUCKET")
PREFIX = os.getenv("GCP_DATA_PREFIX")

def gcs_path(filename):
    return f"gs://{BUCKET}/{PREFIX}/models/{filename}"

def load_model(model_path):
    model = CatBoostClassifier()
    if model_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(model_path, "rb") as f:
            model.load_model(f)
    else:
        model.load_model(model_path)
    print(f"✅ Model loaded from {model_path}")
    return model

def evaluate_model(model, X_test, y_test):
    print("🔍 Running evaluation...")
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probas)
    return report, auc

def save_report(report, auc, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "validation_report.json")

    report["roc_auc"] = auc
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Report saved to {json_path}")
    return json_path

def run_validation(model_name="catboost_model.cbm", timestamp=None) -> dict:
    print(f"🌍 ENV = {ENV}")

    # Resolve model path
    model_path = gcs_path(model_name) if ENV == "PROD" else os.path.join("models", model_name)
    model = load_model(model_path)

    # Load data
    _, X_test, _, y_test = load_data(timestamp=timestamp)

    # Evaluate
    report, auc = evaluate_model(model, X_test, y_test)
    print(f"📊 AUC: {auc:.4f}")

    # Save report
    report_path = save_report(report, auc)

    if ENV == "PROD":
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Fraud Detection CatBoost"))
        with mlflow.start_run(run_name="validation"):
            mlflow.log_metric("val_auc", auc)
            mlflow.log_artifact(report_path, artifact_path="validation")
            print("📡 Logged to MLflow.")
        shutil.rmtree("reports")

    return {
        "auc": auc,
        "report_path": report_path
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="catboost_model.cbm")
    parser.add_argument("--timestamp", type=str)
    args = parser.parse_args()

    run_validation(model_name=args.model_name, timestamp=args.timestamp)

if __name__ == "__main__":
    main()
