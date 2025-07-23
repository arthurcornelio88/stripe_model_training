
import os
import re
import time
import pandas as pd
import random
from uuid import uuid4
from typing import Optional, List, Dict
import gcsfs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from utils.file_io import wait_for_gcs
from model_training_api.src.train import run_training, run_fine_tuning
from model_training_api.src.predict import run_prediction
from model_training_api.src.validate_model import run_validation
from model_training_api.src.preprocessing import run_preprocessing
from model_training_api.utils.file_io import read_csv_flexible
from model_training_api.utils.storage_path import get_storage_path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# FastAPI app with metadata for docs
app = FastAPI(
    title="My Automatic Fraud Detection API - Bloc B3 Certificate",
    description="Final project to obtain the RNCP 7 level in Architecte of AI, from Jedha (France)",
    version="1.0.1"
)

ENV = os.getenv("ENV", "DEV")


@app.get("/", tags=["Health"])
def get_index():
    """Returns greetings"""
    return {"greetings": "welcome"}

@app.get("/ping", tags=["Health"])
def ping():
    """Health check endpoint."""
    return {"status": "alive"}



class PreprocessRequest(BaseModel):
    input_path: str
    output_dir: Optional[str] = "data/processed"
    log_amt: Optional[bool] = True
    for_prediction: Optional[bool] = True

@app.post("/preprocess", tags=["Preprocessing"])
def preprocess_endpoint(request: PreprocessRequest):
    """Preprocesses a CSV file and saves the output to the specified directory."""
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

@app.post("/preprocess_direct", tags=["Preprocessing"])
def preprocess_direct(request: PreprocessDirectRequest):
    """Preprocesses data sent directly as JSON and saves the output."""
    df = pd.DataFrame(request.data).reset_index(drop=True)
    local_tmp = f"/tmp/raw_input_{uuid4().hex}.csv"
    df.to_csv(local_tmp, index=False)

    tmp_input = get_storage_path("shared_data/tmp", os.path.basename(local_tmp))

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
        import shutil
        shutil.copy2(local_tmp, tmp_input)
        print(f"✅ Saved tmp_input locally: {tmp_input}")

    output_dir = get_storage_path("shared_data/preprocessed", "")
    if ENV != "PROD":
        os.makedirs(output_dir, exist_ok=True)

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
    timestamp_model_finetune: Optional[str] = None  # Timestamp du modèle à fine-tuner
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

@app.post("/train", tags=["Training"])
def train_endpoint(request: TrainRequest):
    """Trains or fine-tunes the CatBoost model. Use mode='fine_tune' for incremental learning."""
    if request.mode == "fine_tune":
        print(f"🧠 Fine-tuning mode activated!")
        print(f"📊 Parameters: lr={request.learning_rate}, epochs={request.epochs}")
        try:
            result = run_fine_tuning(
                model_name=request.model_name,
                timestamp=request.timestamp,
                timestamp_model_finetune=request.timestamp_model_finetune,
                learning_rate=request.learning_rate,
                epochs=request.epochs
            )
            print(f"🔍 DEBUG: run_fine_tuning returned: {result}")
            print(f"🔍 DEBUG: result keys: {list(result.keys())}")
            print(f"🔍 DEBUG: model_path value: {result.get('model_path', 'NOT_FOUND')}")
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
                "auc_improvement": result["auc"] - 0.74,
                "metrics": result["metrics"],
                "model_path": result["model_path"],
                "parameters": {
                    "learning_rate": request.learning_rate,
                    "epochs": request.epochs
                }
            }
            print(f"🔍 DEBUG: API response will be: {response_data}")
            print(f"🔍 DEBUG: model_path in response: {response_data.get('model_path', 'MISSING')}")
            return response_data
        except Exception as e:
            return {"status": "error", "message": f"Fine-tuning failed: {str(e)}"}
    else:
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

@app.post("/validate", tags=["Validation"])
def validate_model_endpoint(request: ValidateRequest):
    """Validates a model on provided or historical data."""
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

@app.post("/predict", tags=["Prediction"])
def predict_endpoint(request: PredictRequest):
    """Runs batch prediction on the input data using the specified model."""
    print(f"📥 Predict input = {request.input_path}")
    if ENV == "PROD" and request.input_path.startswith("gs://"):
        wait_for_gcs(request.input_path, timeout=30)
        time.sleep(2)
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

@app.post("/monitor", tags=["Monitoring"])
def monitor_drift(request: DriftRequest):
    """Detects data drift between a reference and current dataset, saves a report, and returns drift summary."""
    import json
    ref = read_csv_flexible(request.reference_path, env=ENV)
    curr = read_csv_flexible(request.current_path, env=ENV)
    if "Unnamed: 0" in ref.columns:
        ref = ref.drop(columns=["Unnamed: 0"])
    curr = read_csv_flexible(request.current_path, env=ENV)
    if "Unnamed: 0" in curr.columns:
        curr = curr.drop(columns=["Unnamed: 0"])
    sample_size = min(len(ref), 5 * len(curr))
    ref = ref.sample(n=sample_size, random_state=42)
    print(f"📂 REF PATH = {request.reference_path}")
    print(f"📂 CURR PATH = {request.current_path}")
    print(f"📂 ENV = {ENV}")
    print("📑 REF columns:", ref.columns.tolist())
    print("📑 CURR columns:", curr.columns.tolist())
    preset = DataDriftPreset(drift_share=0.2)
    report = Report(metrics=[preset])
    report.run(reference_data=ref, current_data=curr)
    output_html_rel = request.output_html
    shared_data_path = os.getenv("SHARED_DATA_PATH", "/app/shared_data")
    abs_path_html = os.path.join(shared_data_path, output_html_rel)
    if abs_path_html.startswith("gs://"):
        tmp_html = "/tmp/data_drift.html"
        report.save_html(tmp_html)
        fs = gcsfs.GCSFileSystem()
        with fs.open(abs_path_html, "w") as f:
            with open(tmp_html, "r") as local_f:
                f.write(local_f.read())
        print(f"📄 Drift report uploaded to GCS: {abs_path_html}")
    else:
        os.makedirs(os.path.dirname(abs_path_html), exist_ok=True)
        report.save_html(abs_path_html)
        print(f"📄 Drift report saved locally: {abs_path_html}")
    json_path = abs_path_html.replace(".html", ".json")
    report_dict = report.as_dict()
    if json_path.startswith("gs://"):
        tmp_json = "/tmp/data_drift.json"
        with open(tmp_json, "w") as f:
            json.dump(report_dict, f, indent=2)
        fs = gcsfs.GCSFileSystem()
        with fs.open(json_path, "w") as f:
            with open(tmp_json, "r") as tmpf:
                f.write(tmpf.read())
        print(f"📄 Drift JSON uploaded to GCS: {json_path}")
    else:
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"📄 Drift JSON saved locally: {json_path}")
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
