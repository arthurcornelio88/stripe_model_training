# model_training_api/src/mlflow_config.py
import mlflow
import os
from typing import Optional

class MLflowConfig:
    def __init__(self):
        self.env = os.getenv("ENV", "DEV")
        self.tracking_uri = self._get_tracking_uri()
        self.experiment_name = self._get_experiment_name()
        self.setup_mlflow()
    
    def _get_tracking_uri(self) -> str:
        """Récupère l'URI de tracking MLflow selon l'environnement"""
        uri = os.getenv("MLFLOW_TRACKING_URI", "")
        
        if not uri:
            # Fallback selon l'environnement
            if self.env == "PROD":
                print("⚠️ MLFLOW_TRACKING_URI not set in production, using sqlite fallback")
                return "sqlite:///mlflow.db"
            else:
                return "http://localhost:5000"
        
        # Nettoyer l'URI
        uri = uri.strip().rstrip("/")
        print(f"🔍 Using MLflow tracking URI: {uri}")
        return uri
    
    def _get_experiment_name(self) -> str:
        """Récupère le nom de l'expérience MLflow"""
        experiment = os.getenv("MLFLOW_EXPERIMENT", "")
        
        if not experiment:
            # Fallback selon l'environnement
            if self.env == "PROD":
                return "fraud-detection-prod"
            else:
                return "fraud-detection-dev"
        
        return experiment.strip()
    
    def setup_mlflow(self):
        """Configure MLflow selon l'environnement"""
        try:
            print(f"🔧 Setting up MLflow for {self.env} environment...")
            print(f"   Tracking URI: {self.tracking_uri}")
            print(f"   Experiment: {self.experiment_name}")
            
            # Configurer MLflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            
            print("✅ MLflow configuration complete")
            
        except Exception as e:
            print(f"❌ Error setting up MLflow: {e}")
            raise
    
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