import os
import argparse
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
import mlflow
import json
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

def run_inference(model, df):
    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1]
    result = df.copy()
    result["is_fraud_pred"] = preds
    result["fraud_score"] = probas

    return result

def run_prediction(input_path: str, model_name: str, output_path: str):
    # Déduire le bon chemin du modèle
    model_path = gcs_path(model_name) if ENV == "PROD" else os.path.join("models", model_name)
    model = load_model(model_path)

    # Charger les données d'entrée (via read_csv_flexible dans main.py)
    df = pd.read_csv(input_path)
    print(f"📥 Loaded input: {df.shape} rows from {input_path}")

    # Prédire
    result = run_inference(model, df)

    # Sauvegarder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"📤 Saved predictions to {output_path}")

    # Journaliser dans MLflow si en PROD
    if ENV == "PROD":
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Fraud Detection CatBoost"))
        with mlflow.start_run(run_name="batch_prediction"):
            mlflow.log_artifact(input_path, artifact_path="inputs")
            mlflow.log_artifact(output_path, artifact_path="outputs")
            print("📡 Logged artifacts to MLflow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/processed/new_data.csv")
    parser.add_argument("--model_name", type=str, default="catboost_model.cbm")
    parser.add_argument("--output_path", type=str, default="data/predictions.csv")
    args = parser.parse_args()

    run_prediction(args.input_path, args.model_name, args.output_path)


if __name__ == "__main__":
    main()
