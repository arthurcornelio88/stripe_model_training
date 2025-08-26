#!/usr/bin/env python3
# deployment/validate_env.py
"""
Script de validation des variables d'environnement
À utiliser pour déboguer les problèmes de configuration
"""

import os
import sys
sys.path.append('/app')
sys.path.append('/app/deployment')

def validate_environment():
    """Valide la configuration des variables d'environnement"""
    print("🔍 VALIDATION DES VARIABLES D'ENVIRONNEMENT")
    print("=" * 50)
    
    # Variables critiques
    critical_vars = [
        "ENV",
        "GOOGLE_CLOUD_PROJECT",
        "GCS_BUCKET", 
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT",
        "SHARED_DATA_PATH",
        "MODEL_PATH",
        "SERVICE_TYPE"
    ]
    
    print("📋 Variables critiques:")
    for var in critical_vars:
        value = os.environ.get(var)
        status = "✅" if value else "❌"
        print(f"  {status} {var}: {value}")
    
    print("\n🔍 Variables d'environnement complètes:")
    env_vars = dict(os.environ)
    mlops_vars = {k: v for k, v in env_vars.items() if any(keyword in k.lower() for keyword in ['mlflow', 'gcs', 'bucket', 'path', 'experiment', 'gcp'])}
    
    for var, value in sorted(mlops_vars.items()):
        print(f"  {var}: {value}")
    
    print("\n🧪 Test de chargement des modules:")
    try:
        from deployment.env_config import EnvironmentConfig
        print("✅ env_config importé avec succès")
        
        config = EnvironmentConfig()
        print("✅ EnvironmentConfig initialisé")
        
        config.setup_environment_variables()
        print("✅ Variables d'environnement configurées")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🔍 Test MLflow:")
    try:
        import mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI", "NOT SET")
        print(f"  URI configuré: {uri}")
        
        # Test simple de connection
        if uri != "NOT SET" and uri != "sqlite:///mlflow.db":
            import requests
            try:
                response = requests.get(f"{uri}/health", timeout=5)
                print(f"  Test connexion: {response.status_code}")
            except Exception as e:
                print(f"  Test connexion: ÉCHEC ({e})")
        
        mlflow.set_tracking_uri(uri)
        print("✅ MLflow URI configuré")
        
    except Exception as e:
        print(f"❌ Erreur MLflow: {e}")

if __name__ == "__main__":
    validate_environment()
