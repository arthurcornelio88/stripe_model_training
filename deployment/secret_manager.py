# deployment/secret_manager.py
import os
from google.cloud import secretmanager
from typing import Dict, Optional

class SecretManager:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
        
    def get_secret(self, secret_id: str) -> str:
        """Récupère un secret depuis Google Secret Manager"""
        try:
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            print(f"❌ Error accessing secret {secret_id}: {e}")
            return None
    
    def load_secrets_to_env(self, secret_mapping: Dict[str, str]):
        """Charge les secrets dans les variables d'environnement"""
        for env_var, secret_id in secret_mapping.items():
            secret_value = self.get_secret(secret_id)
            if secret_value:
                os.environ[env_var] = secret_value
                print(f"✅ Loaded secret {secret_id} → {env_var}")
                
                # Create aliases for backward compatibility
                if env_var == "GCS_BUCKET":
                    os.environ["GCP_BUCKET"] = secret_value  # Alias for legacy code
                    print(f"✅ Created alias GCP_BUCKET → {env_var}")
            else:
                print(f"⚠️ Failed to load secret {secret_id}")

# Configuration des secrets pour la production
PROD_SECRETS = {
    # Model training secrets (standardized names)
    "GCS_BUCKET": "gcp-bucket",  # Will be used as both GCS_BUCKET and GCP_BUCKET
    "GCP_PROJECT": "gcp-project", 
    "GCP_REGION": "gcp-region",
    # Note: SHARED_DATA_PATH and MODEL_PATH are set in entrypoint.sh based on environment
    "MLFLOW_TRACKING_URI": "mlflow-tracking-uri",
    "MLFLOW_EXPERIMENT": "mlflow-experiment",
    
    # DataOps secrets
    "BQ_PROJECT": "bq-project",
    "BQ_DATASET": "bq-dataset",
    "BQ_PREDICT_DATASET": "bq-predict-dataset",
    "BQ_LOCATION": "bq-location",
    "DISCORD_WEBHOOK_URL": "discord-webhook-url",
    "REFERENCE_DATA_PATH": "reference-data-path",
    
    # API URLs
    "PROD_API_URL": "prod-api-url",
    "MONITOR_URL_PROD": "monitor-url-prod",
    "PREPROCESS_ENDPOINT": "preprocess-endpoint",
    
    # Add other secrets as needed
}

def setup_production_secrets():
    """Configure les secrets pour la production"""
    if os.getenv("ENV") == "PROD":
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if project_id:
            secret_manager = SecretManager(project_id)
            secret_manager.load_secrets_to_env(PROD_SECRETS)