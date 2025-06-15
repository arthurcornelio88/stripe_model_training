from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import pandas as pd

from model_training_api.src import preprocessing
from model_training_api.src.train import run_training
from model_training_api.src.predict import run_prediction
from model_training_api.src.validate_model import run_validation

from model_training_api.utils.file_io import read_csv_flexible

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


app = FastAPI()
ENV = os.getenv("ENV", "DEV")

@app.get("/ping")
def ping():
    return {"status": "alive"}


from model_training_api.src.preprocessing import run_preprocessing

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

from pydantic import BaseModel, Field, validator
from typing import Optional
import re

class TrainRequest(BaseModel):
    timestamp: Optional[str] = Field(
        default=None,
        description="Timestamp au format YYYYmmdd_HHMMSS ou null pour utiliser le plus récent"
    )
    test: Optional[bool] = False
    fast: Optional[bool] = False
    model_name: Optional[str] = "catboost_model.cbm"

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
    run_training(
        timestamp=request.timestamp,
        test=request.test,
        fast=request.fast,
        model_name=request.model_name
    )
    return {"status": "training complete"}

class ValidateRequest(BaseModel):
    model_name: Optional[str] = "catboost_model.cbm"
    timestamp: Optional[str] = None

@app.post("/validate")
def validate_model_endpoint(request: ValidateRequest):
    result = run_validation(
        model_name=request.model_name,
        timestamp=request.timestamp
    )
    return {"status": "model validated", **result}



class PredictRequest(BaseModel):
    input_path: str
    model_name: Optional[str] = "catboost_model.cbm"
    output_path: Optional[str] = "data/predictions.csv"

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    # Assure que fichier existe (lecture pour validation)
    df = read_csv_flexible(request.input_path, env=ENV)
    df.shape  # déclenche l’erreur si non lisible

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

    ref = read_csv_flexible(request.reference_path, env=ENV)
    curr = read_csv_flexible(request.current_path, env=ENV)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)
    os.makedirs(os.path.dirname(request.output_html), exist_ok=True)
    report.save_html(request.output_html)
    return {"status": "drift report generated", "report_path": request.output_html}
