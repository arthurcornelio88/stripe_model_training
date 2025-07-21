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
SHARED_DATA_PATH = os.getenv("SHARED_DATA_PATH")

def resolve_path(relative_path):
    """Resolve file path based on environment"""
    if ENV == "PROD":
        return f"gs://{BUCKET}/{SHARED_DATA_PATH}/{relative_path}"
    else:
        return f"/app/shared_data/{relative_path}"

def get_file_path(filename, subfolder=""):
    """Get file path with optional subfolder"""
    if subfolder:
        return resolve_path(f"{subfolder}/{filename}")
    else:
        return resolve_path(filename)

def load_model(model_path):
    model = CatBoostClassifier()
    if model_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem(skip_instance_cache=True, cache_timeout=0)

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

    # === Mode Production : Validation directe avec données pré-calculées
    if validation_mode == "production" and production_data:
        print("🎯 Production validation mode - using pre-calculated predictions")
        
        y_true = production_data["y_true"]
        y_pred_proba = production_data["y_pred_proba"]
        y_pred_binary = production_data["y_pred_binary"]
        
        # 🔍 DEBUG : Analyser les données
        print(f"🔍 DEBUG - Data shapes: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}, y_pred_binary={len(y_pred_binary)}")
        print(f"🔍 DEBUG - y_true unique values: {list(set(y_true))}")
        print(f"🔍 DEBUG - y_pred_binary unique values: {list(set(y_pred_binary))}")
        print(f"🔍 DEBUG - y_pred_proba range: [{min(y_pred_proba):.3f}, {max(y_pred_proba):.3f}]")
        print(f"🔍 DEBUG - Fraud cases in y_true: {sum(y_true)} / {len(y_true)} ({100*sum(y_true)/len(y_true):.1f}%)")
        print(f"🔍 DEBUG - Fraud predictions in y_pred_binary: {sum(y_pred_binary)} / {len(y_pred_binary)} ({100*sum(y_pred_binary)/len(y_pred_binary):.1f}%)")
        
        # Utiliser la fonction d'évaluation existante
        report, auc = evaluate_predictions(y_true, y_pred_proba, y_pred_binary)
        
        n_samples = len(y_true)
        precision = report['1']['precision'] if '1' in report else 0
        recall = report['1']['recall'] if '1' in report else 0
        f1 = report['1']['f1-score'] if '1' in report else 0
        
        print(f"📊 Production AUC: {auc:.4f}")
        print(f"📈 Métriques (n={n_samples}): Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Pas de sauvegarde de rapport pour le mode production
        return {
            "auc": auc,
            "validation_type": "production",
            "data_source": source,
            "n_samples": n_samples,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # === Mode Historical : Validation classique avec modèle
    # Charger le modèle
    model_path = get_file_path(model_name, "models") if ENV == "PROD" else os.path.join("models", model_name)
    model = load_model(model_path)

    # Charger ou utiliser les données
    if X_test is not None and y_test is not None:
        print(f"🔄 Using provided data: {len(X_test)} samples")
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)
        validation_type = validation_type or "production"
        print(f"🎯 Production validation mode with {len(X_test)} samples")
    else:
        print(f"🔄 ENV = {ENV} | Loading historical data...")
        _, X_test, _, y_test = load_data(timestamp=timestamp)
        validation_type = validation_type or "historical"

    # Évaluation
    report, auc = evaluate_model(model, X_test, y_test)
    print(f"📊 AUC ({validation_type}): {auc:.4f}")

    # Sauvegarde du rapport
    # report_path = save_report(report, auc, output_dir="reports")
    local_path, report_path = save_report(report, auc, output_dir="shared_data/reports")

    # Log MLflow si PROD
    if ENV == "PROD":
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Fraud Detection CatBoost"))
        with mlflow.start_run(run_name=f"validation_{validation_type}"):
            mlflow.log_metric("val_auc", auc)
            mlflow.log_param("validation_type", validation_type)
            mlflow.log_artifact(local_path, artifact_path="validation")
            print("📡 Logged to MLflow.")
        
        # Nettoyage uniquement si on a écrit localement dans un répertoire
        if local_path.startswith("/tmp/") and os.path.exists(os.path.dirname(local_path)):
            shutil.rmtree(os.path.dirname(local_path), ignore_errors=True)

    return {
        "auc": auc,
        "report_path": report_path,
        "validation_type": validation_type,
        "data_source": source,
        "n_samples": len(X_test)
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="catboost_model.cbm")
    parser.add_argument("--timestamp", type=str)
    args = parser.parse_args()

    run_validation(model_name=args.model_name, timestamp=args.timestamp)

if __name__ == "__main__":
    main()
