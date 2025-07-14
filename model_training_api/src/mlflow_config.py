# model_training_api/src/mlflow_config.py
import mlflow
import os
from typing import Optional

class MLflowConfig:
    def __init__(self):
        self.env = os.getenv("ENV", "DEV")
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Configure MLflow selon l'environnement"""
        if self.env == "PROD":
            # Configuration production avec Cloud SQL + GCS
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "fraud-detection-prod"))
        else:
            # Configuration développement locale
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("fraud-detection-dev")
    
    def log_model_training(self, model, metrics: dict, params: dict, model_path: str):
        """Log un entraînement de modèle dans MLflow"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.catboost.log_model(
                model,
                "model",
                registered_model_name="fraud-detection-model"
            )
            
            # Log artifacts
            mlflow.log_artifact(model_path, "model_files")
            
            print(f"✅ Model logged to MLflow: {mlflow.active_run().info.run_id}")
            return mlflow.active_run().info.run_id