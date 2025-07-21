from urllib3 import request
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import os
import re
import pandas as pd
import time
import random
from uuid import uuid4
import gcsfs
from utils.file_io import wait_for_gcs

from model_training_api.src.train import run_training, run_fine_tuning
from model_training_api.src.predict import run_prediction
from model_training_api.src.validate_model import run_validation
from model_training_api.src.preprocessing import run_preprocessing

from model_training_api.utils.file_io import read_csv_flexible
from model_training_api.utils.storage_path import get_storage_path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


app = FastAPI()
ENV = os.getenv("ENV", "DEV")

@app.get("/ping")
def ping():
    return {"status": "alive"}


class PreprocessRequest(BaseModel):
    input_path: str
    output_dir: Optional[str] = "data/processed"
    log_amt: Optional[bool] = True
    for_prediction: Optional[bool] = True

@app.post("/preprocess")
def preprocess_endpoint(request: PreprocessRequest):
    timestamp = run_preprocessing(
        input_path=request.input_path,
        output_dir=request.output_dir,
        log_amt=request.log_amt,
        for_prediction=request.for_prediction
    )
    return {"status": "done", "timestamp": timestamp}

class PreprocessDirectRequest(BaseModel):
    data: List[Dict]
    output_dir: Optional[str] = "data/processed"
    log_amt: Optional[bool] = True
    for_prediction: Optional[bool] = True

@app.post("/preprocess_direct")
def preprocess_direct(request: PreprocessDirectRequest):
    # 1. Génération du fichier CSV temporaire localement
    df = pd.DataFrame(request.data).reset_index(drop=True)
    local_tmp = f"/tmp/raw_input_{uuid4().hex}.csv"
    df.to_csv(local_tmp, index=False)

    # 2. Résolution du chemin cible (via storage path)
    tmp_input = get_storage_path("shared_data/tmp", os.path.basename(local_tmp))

    # 3. Upload vers GCS si nécessaire
    if tmp_input.startswith("gs://"):
        try:
            fs = gcsfs.GCSFileSystem()
            with fs.open(tmp_input, "w") as f:
                with open(local_tmp, "r") as local_f:
                    f.write(local_f.read())
            print(f"✅ Uploaded tmp_input to GCS: {tmp_input}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload to GCS: {str(e)}")
    else:
        os.makedirs(os.path.dirname(tmp_input), exist_ok=True)
        shutil.copy2(local_tmp, tmp_input)
        print(f"✅ Saved tmp_input locally: {tmp_input}")

    # 4. Résolution du dossier de sortie
    output_dir = get_storage_path("shared_data/preprocessed", "")
    if ENV != "PROD":
        os.makedirs(output_dir, exist_ok=True)

    # 5. Lancer le préprocessing
    timestamp = run_preprocessing(
        input_path=tmp_input,
        output_dir=output_dir,
        log_amt=request.log_amt,
        for_prediction=request.for_prediction
    )

    return {"status": "done", "timestamp": timestamp}



class TrainRequest(BaseModel):
    timestamp: Optional[str] = Field(
        default=None,
        description="Timestamp au format YYYYmmdd_HHMMSS ou null pour utiliser le plus récent!"
    )
    test: Optional[bool] = False
    fast: Optional[bool] = False
    model_name: Optional[str] = "catboost_model.cbm"
    mode: Optional[str] = "full_train"  # "full_train" ou "fine_tune"
    learning_rate: Optional[float] = 0.1  # LR pour fine-tuning
    epochs: Optional[int] = 50  # Nombre d'epochs

    @validator("timestamp")
    def validate_timestamp(cls, value):
        if value in (None, "", "latest"):
            return None
        pattern = r"^\d{8}_\d{6}$"
        if not re.match(pattern, value):
            raise ValueError(f"⛔ Le timestamp '{value}' n'est pas valide. Format attendu : YYYYmmdd_HHMMSS")
        return value


@app.post("/train")
def train_endpoint(request: TrainRequest):
    if request.mode == "fine_tune":
        # 🧠 FINE-TUNING MODE - REAL IMPLEMENTATION
        print(f"🧠 Fine-tuning mode activated!")
        print(f"📊 Parameters: lr={request.learning_rate}, epochs={request.epochs}")
        
        try:
            result = run_fine_tuning(
                model_name=request.model_name,
                timestamp=request.timestamp,
                learning_rate=request.learning_rate,
                epochs=request.epochs
            )
            
            print(f"🔍 DEBUG: run_fine_tuning returned: {result}")  # 🔧 Debug log
            print(f"🔍 DEBUG: result keys: {list(result.keys())}")  # 🔧 Debug log
            print(f"🔍 DEBUG: model_path value: {result.get('model_path', 'NOT_FOUND')}")  # 🔧 Debug log
            
            # 🚨 VALIDATION: Vérifier que model_path existe
            if "model_path" not in result:
                print("❌ ERROR: model_path not found in result!")
                raise KeyError("model_path not found in fine-tuning result")
            
            if result["model_path"] is None:
                print("❌ ERROR: model_path is None!")
                raise ValueError("model_path is None in fine-tuning result")
            
            response_data = {
                "status": "fine_tuning_complete",
                "mode": "fine_tune",
                "model_updated": result["model_updated"],
                "auc": result["auc"],
                "auc_improvement": result["auc"] - 0.74,  # Estimation basée sur l'AUC précédent
                "metrics": result["metrics"],
                "model_path": result["model_path"],  # 🔧 Add model path to response
                "parameters": {
                    "learning_rate": request.learning_rate,
                    "epochs": request.epochs
                }
            }
            
            print(f"🔍 DEBUG: API response will be: {response_data}")  # 🔧 Debug log
            print(f"🔍 DEBUG: model_path in response: {response_data.get('model_path', 'MISSING')}")  # 🔧 Debug log
            
            return response_data
            
        except Exception as e:
            return {"status": "error", "message": f"Fine-tuning failed: {str(e)}"}
    
    else:
        # 🏋️ FULL TRAINING MODE (mode original)
        print("🏋️ Full training mode")
        run_training(
            timestamp=request.timestamp,
            test=request.test,
            fast=request.fast,
            model_name=request.model_name
        )
        return {"status": "training complete", "mode": "full_train"}

class ValidateRequest(BaseModel):
    model_name: Optional[str] = "catboost_model.cbm"
    timestamp: Optional[str] = None
    source: Optional[str] = "local"
    bq_date: Optional[str] = None
    X_test: Optional[List[Dict]] = None
    y_test: Optional[List[int]] = None
    validation_type: Optional[str] = None
    validation_mode: Optional[str] = "historical"  # "historical" ou "production"
    production_data: Optional[Dict] = None  # Pour le mode production

@app.post("/validate")
def validate_model_endpoint(request: ValidateRequest):
    result = run_validation(
        model_name=request.model_name,
        timestamp=request.timestamp,
        source=request.source,
        bq_date=request.bq_date,
        X_test=request.X_test,
        y_test=request.y_test,
        validation_type=request.validation_type,
        validation_mode=request.validation_mode,
        production_data=request.production_data
    )
    return {"status": "model validated", **result}


class PredictRequest(BaseModel):
    input_path: str
    model_name: Optional[str] = "catboost_model.cbm"
    output_path: Optional[str] = get_storage_path("shared_data/predictions", "predictions.csv")

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    print(f"📥 Predict input = {request.input_path}")

    # ✅ Nouveau bloc : attendre la propagation GCS
    if ENV == "PROD" and request.input_path.startswith("gs://"):
        wait_for_gcs(request.input_path, timeout=30)
        time.sleep(2)  # 🧯 marge de sécurité
    else:
        os.makedirs(os.path.dirname(request.output_path), exist_ok=True)

    df = read_csv_flexible(request.input_path, env=ENV)
    df.shape

    run_prediction(
        input_path=request.input_path,
        model_name=request.model_name,
        output_path=request.output_path
    )

    return {"status": "prediction complete", "output": request.output_path}

class DriftRequest(BaseModel):
    reference_path: str
    current_path: str
    output_html: Optional[str] = "reports/data_drift.html"

@app.post("/monitor")
def monitor_drift(request: DriftRequest):
    import json

    # Charger les données
    ref = read_csv_flexible(request.reference_path, env=ENV)
    curr = read_csv_flexible(request.current_path, env=ENV)

    # 🧹 Clean colonnes parasites
    if "Unnamed: 0" in ref.columns:
        ref = ref.drop(columns=["Unnamed: 0"])

    # ⚡️ Sous-échantillonner la référence pour correspondre à la taille du current
    curr = read_csv_flexible(request.current_path, env=ENV)
    if "Unnamed: 0" in curr.columns:
        curr = curr.drop(columns=["Unnamed: 0"])

    sample_size = min(len(ref), 5 * len(curr))  # Ex: 2500 lignes max si curr en a 500
    ref = ref.sample(n=sample_size, random_state=42)

    print(f"📂 REF PATH = {request.reference_path}")
    print(f"📂 CURR PATH = {request.current_path}")
    print(f"📂 ENV = {ENV}")

    print("📑 REF columns:", ref.columns.tolist())
    print("📑 CURR columns:", curr.columns.tolist())

    # Générer le rapport
    # Exemple : plus sensible (20%)
    preset = DataDriftPreset(drift_share=0.2)
    report = Report(metrics=[preset])
    report.run(reference_data=ref, current_data=curr)

    # Définir le nom du rapport HTML  
    output_html_rel = request.output_html
    shared_data_path = os.getenv("SHARED_DATA_PATH", "/app/shared_data")
    abs_path_html = os.path.join(shared_data_path, output_html_rel)

    # Sauvegarde HTML
    if not abs_path_html.startswith("gs://"):
        os.makedirs(os.path.dirname(abs_path_html), exist_ok=True)

    report.save_html(abs_path_html)
    print(f"📄 Drift report saved to: {abs_path_html}")

    # Sauvegarder JSON
    json_path = abs_path_html.replace(".html", ".json")
    report_dict = report.as_dict()
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # Extraire les résultats
    drift_summary = {}
    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DataDriftTable":
            result = metric.get("result", {})
            drift_summary = {
                "drift_detected": result.get("dataset_drift", False),
                "n_drifted_columns": result.get("number_of_drifted_columns", 0),
                "share_drifted_columns": result.get("share_of_drifted_columns", 0.0),
                "drifted_columns": list(result.get("drift_by_columns", {}).keys()),
            }
            break

    return {
        "status": "drift report generated",
        "report_html": abs_path_html,
        "report_json": json_path,
        "drift_summary": drift_summary
    }
