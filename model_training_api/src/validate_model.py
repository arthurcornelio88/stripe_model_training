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
from model_training_api.utils.file_io import read_csv_flexible
from model_training_api.utils.storage_path import get_storage_path


load_dotenv()
ENV = os.getenv("ENV", "DEV")
BUCKET = os.getenv("GCP_BUCKET")
SHARED_DATA_PATH = os.getenv("SHARED_DATA_PATH")

def resolve_path(relative_path):
    if ENV == "PROD":
        if SHARED_DATA_PATH.startswith("gs://"):
            return f"{SHARED_DATA_PATH}/{relative_path}"
        else:
            return f"gs://{BUCKET}/{SHARED_DATA_PATH}/{relative_path}"
    else:
        return f"/app/shared_data/{relative_path}"

def get_file_path(subfolder, filename: str) -> str:
    return get_storage_path(subfolder, filename)

# def load_model(model_path):
#     model = CatBoostClassifier()
#     if model_path.startswith("gs://"):
#         fs = gcsfs.GCSFileSystem(skip_instance_cache=True, cache_timeout=0)

#         with fs.open(model_path, "rb") as f:
#             model.load_model(f)
#     else:
#         model.load_model(model_path)
#     print(f"✅ Model loaded from {model_path}")
#     return model

def load_model(model_path: str):
    model = CatBoostClassifier()
    
    if model_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem(skip_instance_cache=True, cache_timeout=0)
        
        # Téléchargement temporaire dans /tmp
        local_tmp = f"/tmp/model.cbm"
        fs.get(model_path, local_tmp)

        model.load_model(local_tmp)
        print(f"✅ Model loaded from GCS: {model_path}")
    
    else:
        model.load_model(model_path)
        print(f"✅ Model loaded locally from: {model_path}")

    return model


def evaluate_model(model, X_test, y_test):
    print("🔍 Running evaluation...")
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probas)
    return report, auc

def evaluate_predictions(y_true, y_pred_proba, y_pred_binary):
    """Évaluation directe avec des prédictions pré-calculées"""
    print("🔍 Running evaluation on pre-calculated predictions...")
    report = classification_report(y_true, y_pred_binary, output_dict=True)
    auc = roc_auc_score(y_true, y_pred_proba)
    return report, auc

def save_report(report, auc, output_dir="shared_data/reports"):
    json_filename = "validation_report.json"
    json_path = os.path.join(output_dir, json_filename)

    report["roc_auc"] = auc

    if ENV == "PROD" and json_path.startswith("gs://"):
        # 1. Sauvegarde locale temporaire
        local_temp = f"/tmp/{json_filename}"
        with open(local_temp, "w") as f:
            json.dump(report, f, indent=2)

        # 2. Upload GCS
        fs = gcsfs.GCSFileSystem()
        with fs.open(json_path, "w") as f:
            with open(local_temp, "r") as tmpf:
                f.write(tmpf.read())

        print(f"📄 Report saved to GCS: {json_path}")
        return local_temp, json_path  # tuple (local_path, gcs_path)

    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved locally to: {json_path}")
        return json_path, json_path



def run_validation(
    model_name="catboost_model.cbm",
    timestamp=None,
    source="local",
    bq_date=None,
    X_test=None,
    y_test=None,
    validation_type=None,
    validation_mode="historical",
    production_data=None
) -> dict:
    print(f"🌍 ENV = {ENV} | Source = {source} | Mode = {validation_mode}")

    # === Mode Production : prédictions déjà calculées
    if validation_mode == "production" and production_data:
        print("🎯 Production validation mode - using pre-calculated predictions")
        y_true = production_data["y_true"]
        y_pred_proba = production_data["y_pred_proba"]
        y_pred_binary = production_data["y_pred_binary"]

        report, auc = evaluate_predictions(y_true, y_pred_proba, y_pred_binary)
        return {
            "auc": auc,
            "validation_type": "production",
            "data_source": source,
            "n_samples": len(y_true),
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"]
        }

    # === Mode normal : on charge modèle + données
    model_path = get_file_path("models", model_name) if ENV == "PROD" else os.path.join("models", model_name)
    model = load_model(model_path)

    if X_test is not None and y_test is not None:
        print("🎯 Using directly provided test data")
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)
        validation_type = validation_type or "manual"
    else:
        print(f"🔄 Loading test or prediction data using timestamp: {timestamp}")

        # Try loading test data first
        try:
            x_path = get_file_path("shared_data/preprocessed", f"X_test_{timestamp}.csv")
            y_path = get_file_path("shared_data/preprocessed", f"y_test_{timestamp}.csv")
            X_test = read_csv_flexible(x_path, env=ENV)
            y_test = read_csv_flexible(y_path, env=ENV).squeeze()
            validation_type = "historical"
            print("✅ Loaded X_test and y_test")
        except Exception as e1:
            print(f"⚠️ Could not find X_test/y_test: {e1}")
            print("🔄 Trying to load X_pred/y_pred instead...")
            try:
                x_path = get_file_path("shared_data/preprocessed", f"X_pred_{timestamp}.csv")
                y_path = get_file_path("shared_data/preprocessed", f"y_pred_{timestamp}.csv")
                X_test = read_csv_flexible(x_path, env=ENV)
                y_test = read_csv_flexible(y_path, env=ENV).squeeze()
                validation_type = "prediction"
                print("✅ Fallback: Loaded X_pred and y_pred")
            except Exception as e2:
                raise FileNotFoundError(f"❌ Could not find any test or prediction data for timestamp '{timestamp}'\n{e2}")

    print(f"📊 Test shape: {X_test.shape} | y_test: {y_test.shape}")
    report, auc = evaluate_model(model, X_test, y_test)
    local_path, report_path = save_report(report, auc, output_dir="shared_data/reports")

    # Optionally log to MLflow
    if ENV == "PROD":
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Fraud Detection CatBoost"))
        with mlflow.start_run(run_name=f"validation_{validation_type}"):
            mlflow.log_metric("val_auc", auc)
            mlflow.log_param("validation_type", validation_type)
            mlflow.log_artifact(local_path, artifact_path="validation")
            print("📡 Logged to MLflow")

    return {
        "auc": auc,
        "report_path": report_path,
        "validation_type": validation_type,
        "data_source": source,
        "n_samples": len(X_test),
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="catboost_model.cbm")
    parser.add_argument("--timestamp", type=str)
    args = parser.parse_args()

    run_validation(model_name=args.model_name, timestamp=args.timestamp)

if __name__ == "__main__":
    main()
