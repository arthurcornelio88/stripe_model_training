import os
import argparse
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
import mlflow
import json
import gcsfs
from model_training_api.utils.storage_path import get_storage_path

load_dotenv()

ENV = os.getenv("ENV", "DEV")

print("For commit 2")

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


def run_inference(model, df):
    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1]
    result = df.copy()
    result["is_fraud_pred"] = preds
    result["fraud_score"] = probas

    return result

def run_prediction(input_path: str, model_name: str, output_path: str):
    # 🔍 Charger le modèle
    model_path = get_storage_path("models", model_name)
    model = load_model(model_path)

    # 📥 Charger les données d'entrée
    df = pd.read_csv(input_path)
    print(f"📥 Loaded input: {df.shape} rows from {input_path}")

    # 🧠 Inférence
    result = run_inference(model, df)

    # 💾 Sauvegarde des prédictions
    if output_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(output_path, "w") as f:
            result.to_csv(f, index=False)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to {output_path}")

    # 📡 Logging MLflow uniquement si chemins sont locaux
    if ENV == "PROD":
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Fraud Detection CatBoost"))
        with mlflow.start_run(run_name="batch_prediction"):
            if not input_path.startswith("gs://"):
                mlflow.log_artifact(input_path, artifact_path="inputs")
            else:
                print(f"⚠️ Skipped MLflow log: input_path is on GCS ({input_path})")

            if not output_path.startswith("gs://"):
                mlflow.log_artifact(output_path, artifact_path="outputs")
            else:
                print(f"⚠️ Skipped MLflow log: output_path is on GCS ({output_path})")

            print("📡 MLflow logging complete")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=get_storage_path("shared_data/preprocessed", "new_data.csv"))
    parser.add_argument("--model_name", type=str, default="catboost_model.cbm")
    parser.add_argument("--output_path", type=str, default=get_storage_path("shared_data/predictions", "predictions.csv"))
    args = parser.parse_args()

    run_prediction(args.input_path, args.model_name, args.output_path)


if __name__ == "__main__":
    main()
