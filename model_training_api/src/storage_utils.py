# model_training_api/src/storage_utils.py
import os
from google.cloud import storage
from typing import Optional
import pandas as pd

class StorageManager:
    def __init__(self):
        self.env = os.getenv("ENV", "DEV")
        self.gcs_bucket = os.getenv("GCS_BUCKET") if self.env == "PROD" else None
        self.shared_data_path = os.getenv("SHARED_DATA_PATH", "/app/shared_data")
        
    def get_data_path(self, filename: str) -> str:
        """Retourne le chemin approprié selon l'environnement"""
        return os.path.join(self.shared_data_path, filename)
    
    def get_model_path(self, model_name: str) -> str:
        """Retourne le chemin du modèle selon l'environnement"""
        model_path = os.getenv("MODEL_PATH", "/app/models")
        return os.path.join(model_path, model_name)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str):
        """Sauvegarde un DataFrame selon l'environnement"""
        if self.env == "PROD":
            # Upload vers GCS
            client = storage.Client()
            bucket = client.bucket(self.gcs_bucket)
            blob = bucket.blob(f"shared_data/{filename}")
            blob.upload_from_string(df.to_csv(index=False), content_type='text/csv')
        else:
            # Sauvegarde locale
            os.makedirs("/app/shared_data", exist_ok=True)
            df.to_csv(f"/app/shared_data/{filename}", index=False)
    
    def load_dataframe(self, filename: str) -> pd.DataFrame:
        """Charge un DataFrame selon l'environnement"""
        if self.env == "PROD":
            # Download depuis GCS
            client = storage.Client()
            bucket = client.bucket(self.gcs_bucket)
            blob = bucket.blob(f"shared_data/{filename}")
            return pd.read_csv(blob.download_as_text())
        else:
            # Lecture locale
            return pd.read_csv(f"/app/shared_data/{filename}")