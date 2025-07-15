# deployment/env_config.py
"""
Configuration centralisée des variables d'environnement
Résout les problèmes de variables incohérentes et manquantes
"""
import os
from typing import Dict, Optional
from .secret_manager import SecretManager

class EnvironmentConfig:
    """Gestionnaire centralisé des variables d'environnement"""
    
    def __init__(self):
        self.env = os.getenv("ENV", "DEV")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
    def setup_environment_variables(self):
        """Configure toutes les variables d'environnement selon l'environnement"""
        print(f"🔧 Setting up environment variables for {self.env} mode...")
        
        if self.env == "PROD":
            self._setup_production_env()
        else:
            self._setup_development_env()
            
        # Debug: afficher les variables importantes
        self._debug_environment_vars()
        
    def _setup_production_env(self):
        """Configuration production avec secrets Google Cloud"""
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set for production mode")
            
        # Charger les secrets
        secret_manager = SecretManager(self.project_id)
        
        # Mapping des secrets - VERSION RÉVISÉE
        secret_mapping = {
            "GCS_BUCKET": "gcp-bucket",
            "MLFLOW_TRACKING_URI": "mlflow-tracking-uri", 
            "MLFLOW_EXPERIMENT": "mlflow-experiment",
            "BQ_PROJECT": "bq-project",
            "BQ_DATASET": "bq-dataset",
            "BQ_PREDICT_DATASET": "bq-predict-dataset",
            "BQ_LOCATION": "bq-location",
            "DISCORD_WEBHOOK_URL": "discord-webhook-url",
            "PROD_API_URL": "prod-api-url",
        }
        
        # Charger les secrets
        for env_var, secret_id in secret_mapping.items():
            secret_value = secret_manager.get_secret(secret_id)
            if secret_value:
                # Nettoyer les valeurs (enlever les \n etc.)
                clean_value = secret_value.strip()
                os.environ[env_var] = clean_value
                print(f"✅ Loaded secret {secret_id} → {env_var} = {clean_value[:20]}...")
            else:
                print(f"⚠️ Failed to load secret {secret_id}")
        
        # Créer les alias pour compatibilité
        if "GCS_BUCKET" in os.environ:
            os.environ["GCP_BUCKET"] = os.environ["GCS_BUCKET"]  # Alias legacy
            
        # Configuration des chemins production
        bucket = os.environ.get("GCS_BUCKET", "fraud-detection-jedha2024")
        os.environ["SHARED_DATA_PATH"] = f"gs://{bucket}/shared_data"
        os.environ["MODEL_PATH"] = f"gs://{bucket}/models"
        os.environ["MLFLOW_ARTIFACT_URI"] = f"gs://{bucket}/mlflow-artifacts"
        
        # Valeurs par défaut pour les variables manquantes
        self._set_default_if_missing("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        self._set_default_if_missing("MLFLOW_EXPERIMENT", "fraud-detection-prod")
        
    def _setup_development_env(self):
        """Configuration développement locale"""
        print("🔧 Setting up development environment...")
        
        # Chemins locaux
        os.environ["SHARED_DATA_PATH"] = "/app/shared_data"
        os.environ["MODEL_PATH"] = "/app/models"
        os.environ["GCS_BUCKET"] = "dev-bucket"
        os.environ["GCP_BUCKET"] = "dev-bucket"  # Alias
        
        # Configuration MLflow développement
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
        os.environ["MLFLOW_EXPERIMENT"] = "fraud-detection-dev"
        os.environ["MLFLOW_ARTIFACT_URI"] = "./mlruns"
        
        # Créer les répertoires nécessaires
        import pathlib
        pathlib.Path("/app/shared_data").mkdir(parents=True, exist_ok=True)
        pathlib.Path("/app/models").mkdir(parents=True, exist_ok=True)
        pathlib.Path("/app/mlruns").mkdir(parents=True, exist_ok=True)
        
    def _set_default_if_missing(self, env_var: str, default_value: str):
        """Définit une valeur par défaut si la variable n'existe pas"""
        if not os.environ.get(env_var):
            os.environ[env_var] = default_value
            print(f"⚠️ {env_var} not set, using default: {default_value}")
            
    def _debug_environment_vars(self):
        """Affiche les variables d'environnement importantes pour debug"""
        important_vars = [
            "ENV",
            "GOOGLE_CLOUD_PROJECT", 
            "GCS_BUCKET",
            "GCP_BUCKET",
            "SHARED_DATA_PATH",
            "MODEL_PATH",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_EXPERIMENT",
            "MLFLOW_ARTIFACT_URI",
        ]
        
        print("\n🔍 DEBUG: Environment variables:")
        for var in important_vars:
            value = os.environ.get(var, "NOT SET")
            # Masquer les URLs complètes pour la sécurité
            if "URI" in var or "URL" in var:
                display_value = value[:50] + "..." if len(value) > 50 else value
            else:
                display_value = value
            print(f"  {var} = {display_value}")
        print()
        
    def get_env_var(self, var_name: str, default: Optional[str] = None) -> Optional[str]:
        """Récupère une variable d'environnement avec gestion d'erreur"""
        value = os.environ.get(var_name, default)
        if value is None:
            print(f"⚠️ Environment variable {var_name} is not set")
        return value
        
    def validate_required_vars(self, required_vars: list):
        """Valide que les variables requises sont définies"""
        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
                
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
            
        print(f"✅ All required environment variables are set: {required_vars}")

# Fonction utilitaire pour initialiser l'environnement
def setup_environment():
    """Point d'entrée principal pour configurer l'environnement"""
    config = EnvironmentConfig()
    config.setup_environment_variables()
    return config

# Variables requises selon le service
TRAINING_API_REQUIRED_VARS = [
    "ENV",
    "GCS_BUCKET", 
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT",
    "SHARED_DATA_PATH",
    "MODEL_PATH"
]

MOCK_API_REQUIRED_VARS = [
    "ENV",
    "GCS_BUCKET",
]

MLFLOW_SERVER_REQUIRED_VARS = [
    "ENV",
    "MLFLOW_ARTIFACT_URI",
]
