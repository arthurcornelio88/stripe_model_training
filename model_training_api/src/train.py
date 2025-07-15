import pandas as pd
import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
import glob
from mlflow.tracking import MlflowClient
import requests
from urllib.parse import urljoin
import json
import shutil
import argparse
import time
from google.cloud import storage
from .mlflow_config import MLflowConfig
from .storage_utils import StorageManager
import datetime

load_dotenv()

ENV = os.getenv("ENV", "DEV")
BUCKET = os.getenv("GCP_BUCKET")
# Removed PREFIX - using SHARED_DATA_PATH directly
GCS_BUCKET = os.getenv("GCS_BUCKET", "your-mlops-bucket")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000").strip("/")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraud_detection_experiment")

# Get environment-aware paths
SHARED_DATA_PATH = os.getenv("SHARED_DATA_PATH", "/app/shared_data")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")

# Initialiser les managers
storage_manager = StorageManager()
mlflow_config = MLflowConfig()

def get_file_path(filename, io="input"):
    """
    Return environment-aware file path
    """
    if ENV == "PROD":
        return os.path.join(SHARED_DATA_PATH, filename)
    else:
        base_dir = "data/raw/" if io == "input" else SHARED_DATA_PATH
        return os.path.join(base_dir, filename)


def get_latest_file(pattern):
    """
    Return the most recently modified file matching the pattern.
    """
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def resolve_path(name, io="input", timestamp=None):
    """
    Resolve the correct path depending on ENV and optional timestamp.

    - DEV: loads latest or specific timestamped CSV from local folder
    - PROD: uses environment-aware shared data path
    """
    if ENV == "PROD":
        if io == "output":
            return os.path.join(SHARED_DATA_PATH, "preprocessed", name)
        else:
            return os.path.join(SHARED_DATA_PATH, name)

    # DEV
    base_dir = "data/raw/" if io == "input" else os.path.join(SHARED_DATA_PATH, "preprocessed")
    if timestamp:
        filename = name.replace(".csv", f"_{timestamp}.csv")
        return os.path.join(base_dir, filename)
    else:
        pattern = os.path.join(base_dir, name.replace(".csv", "_*.csv"))
        return get_latest_file(pattern)


def load_data(timestamp=None, test_mode=False, sample_size=5000):
    """
    Load and optionally subsample preprocessed data.

    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"🔄 ENV = {ENV} | Loading data...")

    def read(name):
        path = resolve_path(name, io="output", timestamp=timestamp)
        print(f"🔄 Resolving latest path for {name}: {path}")
        df = pd.read_csv(path)
        
        # print(f"🔍 DEBUG {name} loaded with columns: {list(df.columns)}")
        # print(f"🔍 DEBUG {name} shape: {df.shape}")
        
        # 🧹 Nettoyer les colonnes d'index potentielles
        if "Unnamed: 0" in df.columns:
            print(f"🧹 Removing index column 'Unnamed: 0' from {name}")
            df = df.drop(columns=["Unnamed: 0"])
        
        # Nettoyer autres colonnes d'index
        index_cols = [col for col in df.columns if col.startswith("Unnamed:")]
        if index_cols:
            print(f"🧹 Removing index columns {index_cols} from {name}")
            df = df.drop(columns=index_cols)
        
        # print(f"🔍 DEBUG {name} final columns: {list(df.columns)}")
        return df

    X_train = read("X_train.csv")
    X_test = read("X_test.csv")
    y_train = read("y_train.csv").squeeze()
    y_test = read("y_test.csv").squeeze()

    print(f"✅ Data loaded: {len(X_train)} train samples, {len(X_test)} test samples")

    if test_mode:
        print(f"⚡ Sampling {sample_size} rows for fast testing")
        X_train = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)
        X_test = X_test.sample(n=min(sample_size // 4, len(X_test)), random_state=42)
        y_train = y_train.loc[X_train.index]
        y_test = y_test.loc[X_test.index]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_val, y_val, params):
    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=100,
        early_stopping_rounds=50
    )
    print("🔄 Training CatBoost model...")
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probas)
    print("🔄 Evaluating model...")
    return report, auc

def check_mlflow_server(mlflow_uri, experiment_name="default"):
    """
    Ping MLflow server with a safe GET to verify availability.
    Uses a valid API endpoint to avoid 404s.
    """
    try:
        uri = mlflow_uri.rstrip("/") + "/"
        health_url = urljoin(uri, f"api/2.0/mlflow/experiments/get-by-name?experiment_name={experiment_name}")
        response = requests.get(health_url, timeout=3)
        if response.status_code not in (200, 404):  # 404 here may still be valid (experiment not found)
            raise RuntimeError(f"MLflow server responded with {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"❌ Could not connect to MLflow server at {mlflow_uri}: {e}")

def save_and_log_report(report_dict, run_id, output_dir="reports"):
    """
    Sauvegarde le rapport au format JSON/HTML, puis l'upload dans MLflow.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "classification_report.json")
    html_path = os.path.join(output_dir, "classification_report.html")

    # Enregistrement local
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    html_content = "<html><head><title>Classification Report</title></head><body>"
    html_content += "<h2>Classification Report</h2><table border='1'>"
    html_content += "<tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>"

    for label, scores in report_dict.items():
        if isinstance(scores, dict):
            html_content += f"<tr><td>{label}</td><td>{scores.get('precision', 0):.4f}</td>"
            html_content += f"<td>{scores.get('recall', 0):.4f}</td><td>{scores.get('f1-score', 0):.4f}</td>"
            html_content += f"<td>{scores.get('support', 0):.0f}</td></tr>"

    html_content += "</table></body></html>"

    with open(html_path, "w") as f:
        f.write(html_content)

    # Upload dans MLflow (dans artifacts/reports)
    mlflow.log_artifact(json_path, artifact_path="reports")
    mlflow.log_artifact(html_path, artifact_path="reports")

    shutil.rmtree(output_dir)



def log_mlflow(model, params, metrics, report):
    """
    Log model, parameters and metrics to MLflow.
    Handles experiment creation and safe logging based on ENV.
    """
    print("🔄 Setting up MLflow logging...")

    experiment_name = EXPERIMENT or "Fraud Detection CatBoost"
    check_mlflow_server(MLFLOW_URI, experiment_name=experiment_name)

    mlflow.set_tracking_uri(MLFLOW_URI)

    if ENV == "PROD":
        artifact_location = f"gs://{BUCKET}/mlflow-artifacts"
    else:
        artifact_location = "file:./mlruns"

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(name=experiment_name, artifact_location=artifact_location)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    print(f"🔄 Logging to MLflow: {MLFLOW_URI} | Experiment: {experiment_name}")

    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.catboost.log_model(model, artifact_path="model")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name="CatBoostFraudDetector"
        )

        save_and_log_report(report, mlflow.active_run().info.run_id)


# def save_model(model, model_name="catboost_model.cbm"):
#     # 🔧 FIX: Sauvegarder dans shared_data pour éviter les problèmes de permissions
#     if ENV == "PROD":
#         output_path = gcs_path(f"models/{model_name}")
#     else:
#         # En DEV, sauvegarder dans shared_data qui est accessible en écriture
#         os.makedirs("/app/shared_data/models", exist_ok=True)
#         output_path = os.path.join("/app/shared_data/models", model_name)
        
#     print(f"💾 Saving model to: {output_path}")
    
#     try:
#         model.save_model(output_path)
#         print(f"✅ Model saved successfully to: {output_path}")
        
#         # 🔧 SUPPRIMÉ: Pas de copie vers models/ car problème de permissions  
#         # On garde seulement le modèle dans shared_data
        
#         return output_path  # Retourner le chemin pour Discord
                
#     except Exception as e:
#         print(f"❌ Failed to save model: {e}")
#         raise

def save_model(model, model_name="catboost_model.cbm"):
    """Sauvegarde le modèle selon l'environnement avec horodatage"""
    from datetime import datetime
    
    # Horodater le nom du fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = model_name.split('.')[0]  # "catboost_model"
    extension = model_name.split('.')[-1]  # "cbm"
    timestamped_name = f"{base_name}_{timestamp}.{extension}"  # "catboost_model_20250714_143025.cbm"
    
    if ENV == "PROD":
        # Pour GCS, on sauvegarde d'abord localement puis on upload
        temp_path = f"/tmp/{timestamped_name}"
        model.save_model(temp_path)
        
        # Upload vers GCS
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"models/{timestamped_name}")
        blob.upload_from_filename(temp_path)
        
        # Créer aussi un lien "latest" pour faciliter les prédictions
        latest_blob = bucket.blob(f"models/{model_name}")  # Sans timestamp
        latest_blob.upload_from_filename(temp_path)
        
        # Nettoyer le fichier temporaire
        os.remove(temp_path)
        output_path = f"gs://{GCS_BUCKET}/models/{timestamped_name}"
        
        print(f"✅ Model saved with timestamp: {output_path}")
        print(f"✅ Latest model link: gs://{GCS_BUCKET}/models/{model_name}")
        
    else:
        # Sauvegarde locale
        os.makedirs("/app/shared_data/models", exist_ok=True)
        timestamped_path = os.path.join("/app/shared_data/models", timestamped_name)
        latest_path = os.path.join("/app/shared_data/models", model_name)
        
        # Sauvegarder avec timestamp
        model.save_model(timestamped_path)
        
        # Créer une copie "latest" pour faciliter les prédictions
        import shutil
        shutil.copy2(timestamped_path, latest_path)
        
        output_path = timestamped_path
        print(f"✅ Model saved with timestamp: {timestamped_path}")
        print(f"✅ Latest model link: {latest_path}")
        
    return output_path

def run_training(
    timestamp: str = None,
    test: bool = False,
    fast: bool = False,
    model_name: str = "catboost_model.cbm"
):
    X_train, X_test, y_train, y_test = load_data(timestamp=timestamp, test_mode=test)

    if test:
        print("⚡️ Running in TEST mode: minimal CatBoost config")
        params = {
            "iterations": 10,
            "learning_rate": 0.1,
            "depth": 3,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 0,
            "random_seed": 42,
            "class_weights": [1, 10]
        }

    elif fast:
        print("🚀 Running in FAST DEV mode: semi-prod CatBoost config")
        params = {
            "iterations": 150,
            "learning_rate": 0.07,
            "depth": 5,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 100,
            "random_seed": 42,
            "class_weights": [1, 15]
        }

    else:
        print("🏗️ Running in FULL PROD mode: full CatBoost config")
        params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 100,
            "random_seed": 42,
            "class_weights": [1, 25]
        }

    model = train_model(X_train, y_train, X_test, y_test, params)
    report, auc = evaluate_model(model, X_test, y_test)

    metrics = {
        "roc_auc": auc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    }

    log_mlflow(model, params, metrics, report)
    save_model(model, model_name=model_name)

    print("✅ Training complete.")
    print(f"📊 AUC: {auc:.4f} | F1: {metrics['f1']:.4f}")

def fine_tune_model(existing_model_path, X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=10):
    """
    Fine-tune an existing CatBoost model with new data.
    
    Args:
        existing_model_path: Path to the existing .cbm model
        X_train, y_train: New training data
        X_val, y_val: Validation data
        learning_rate: Lower learning rate for fine-tuning
        epochs: Number of additional training iterations
    """
    print(f"🧠 Fine-tuning existing model: {existing_model_path}")
    
    # Charger le modèle existant
    existing_model = CatBoostClassifier()
    existing_model.load_model(existing_model_path)
    
    # 🔍 DEBUG: Comparer les features
    model_features = existing_model.feature_names_
    data_features = list(X_train.columns)
    
    # print(f"🔍 Model expects features: {model_features}")
    # print(f"🔍 Data provides features: {data_features}")
    
    # 🧹 ALIGNER LES COLONNES avec le modèle original
    if model_features != data_features:
        print("🔧 Aligning column order with existing model...")
        
        # Vérifier que toutes les features du modèle sont présentes
        missing_features = [f for f in model_features if f not in data_features]
        extra_features = [f for f in data_features if f not in model_features]
        
        if missing_features:
            # 🔧 FIX: Créer les colonnes manquantes si c'est 'Unnamed: 0' (colonne d'index)
            for missing_col in missing_features:
                if missing_col.startswith('Unnamed:'):
                    print(f"🔧 Creating missing index column: {missing_col}")
                    X_train[missing_col] = range(len(X_train))  # Créer un index factice
                    X_val[missing_col] = range(len(X_val))
                else:
                    raise ValueError(f"❌ Missing required feature: {missing_col}")
        
        if extra_features:
            print(f"⚠️ Extra features in data (will be ignored): {extra_features}")
        
        # Réorganiser les colonnes dans l'ordre attendu par le modèle
        X_train = X_train[model_features]
        X_val = X_val[model_features]
        
        print(f"✅ Columns aligned. New order: {list(X_train.columns)}")
    
    print(f"✅ Model loaded. Starting fine-tuning with lr={learning_rate}, epochs={epochs}")
    
    # 🔧 Créer un nouveau modèle pour le fine-tuning avec des paramètres adaptés
    fine_tuned_model = CatBoostClassifier(
        iterations=epochs,
        learning_rate=learning_rate,
        verbose=5,  # Moins verbeux
        early_stopping_rounds=max(2, epochs//3),  # Plus de patience pour éviter l'overfitting prématuré
        use_best_model=True,
        # Garder les mêmes paramètres que le modèle original mais plus conservateurs
        depth=4,  # Réduire la profondeur pour éviter l'overfitting
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        class_weights=[1, 15],  # Réduire le poids des fraudes pour éviter l'overfitting
        # 🔧 FIX: Désactiver les logs CatBoost pour éviter les problèmes de permissions
        train_dir=None,
        allow_writing_files=False,
        # 🔧 Paramètres pour réduire l'overfitting
        l2_leaf_reg=10,  # Régularisation L2
        bootstrap_type='Bayesian',  # Régularisation par bootstrap
        bagging_temperature=1.0
        # 🔧 FIX: Enlever od_type et od_wait car conflictuel avec early_stopping_rounds
    )
    
    # Fine-tuning = entraîner avec init_model
    try:
        fine_tuned_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            init_model=existing_model  # 🔥 Utiliser le modèle existant comme point de départ
        )
    except Exception as e:
        print(f"⚠️ Fine-tuning with init_model failed: {e}")
        print("🔄 Trying alternative approach without init_model...")
        
        # Fallback: entraîner un nouveau modèle avec plus d'itérations
        fallback_model = CatBoostClassifier(
            iterations=epochs * 3,  # Plus d'itérations
            learning_rate=learning_rate * 0.5,  # Learning rate plus faible
            verbose=5,
            early_stopping_rounds=epochs,
            use_best_model=True,
            depth=4,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            class_weights=[1, 15],
            train_dir=None,
            allow_writing_files=False,
            l2_leaf_reg=10,
            bootstrap_type='Bayesian',
            bagging_temperature=1.0
        )
        
        fallback_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val)
        )
        
        fine_tuned_model = fallback_model
    
    print("✅ Fine-tuning complete!")
    return fine_tuned_model

def run_fine_tuning(
    model_name: str = "catboost_model.cbm",
    timestamp: str = None,
    learning_rate: float = 0.01,
    epochs: int = 10
):
    """
    Run fine-tuning on an existing model with preprocessed data.
    
    Le DAG s'occupe de :
    - Récupérer les données BigQuery
    - Les preprocesser avec /preprocess_direct
    
    L'API s'occupe de :
    - Charger les données déjà preprocessées
    - Faire le fine-tuning
    """

    """Fine-tuning avec MLflow tracking"""
    start_time = time.time()

    print(f"🧠 Starting fine-tuning for model: {model_name}")
    print(f"🔍 Using preprocessed data with timestamp: {timestamp}")
    print(f"💾 Expected files will be: X_train_{timestamp}.csv, X_test_{timestamp}.csv, etc.")
    
    # Charger les données préprocessées (déjà préparées par le DAG)
    X_train, X_test, y_train, y_test = load_data(timestamp=timestamp, test_mode=False)
    
    print(f"� Loaded preprocessed data: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"🔍 Columns: {list(X_train.columns[:5])}..." if len(X_train.columns) > 5 else f"🔍 Columns: {list(X_train.columns)}")
    
    # Vérifier la distribution des classes dans les données d'entraînement
    fraud_count = y_train.sum()
    total_count = len(y_train)
    fraud_ratio = fraud_count / total_count
    
    print(f"📊 Training data distribution:")
    print(f"    Total samples: {total_count}")
    print(f"    Fraud samples: {fraud_count}")
    print(f"    Fraud ratio: {fraud_ratio:.4f}")
    
    # Vérifier si on a assez de données pour fine-tuning
    if fraud_count < 2:
        raise ValueError(f"❌ Insufficient fraud samples for fine-tuning: {fraud_count}. Need at least 2.")
    
    # Échantillonnage stratifié pour préserver les classes
    
    # Prendre au maximum 2000 échantillons, mais garder au moins 5 fraudes si possible
    max_samples = min(2000, len(X_train))
    min_fraud_samples = min(5, fraud_count)  # Garder au moins 5 fraudes ou tout ce qu'on a
    
    if fraud_count <= min_fraud_samples:
        # Si on a très peu de fraudes, on les prend toutes
        fraud_indices = y_train[y_train == 1].index
        non_fraud_indices = y_train[y_train == 0].index
        
        # Prendre toutes les fraudes + échantillon de non-fraudes
        remaining_samples = max_samples - len(fraud_indices)
        if remaining_samples > 0 and len(non_fraud_indices) > 0:
            selected_non_fraud = non_fraud_indices.to_series().sample(n=min(remaining_samples, len(non_fraud_indices)), random_state=42)
            selected_indices = fraud_indices.union(selected_non_fraud)
        else:
            selected_indices = fraud_indices
    else:
        # Échantillonnage stratifié normal
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train, y_train,
            train_size=max_samples,
            stratify=y_train,
            random_state=42
        )
        selected_indices = X_train_sample.index
    
    X_train_sample = X_train.loc[selected_indices]
    y_train_sample = y_train.loc[selected_indices]
    
    final_fraud_count = y_train_sample.sum()
    final_total = len(y_train_sample)
    final_fraud_ratio = final_fraud_count / final_total
    
    print(f"📊 Fine-tuning sample:")
    print(f"    Total samples: {final_total}")
    print(f"    Fraud samples: {final_fraud_count}")
    print(f"    Fraud ratio: {final_fraud_ratio:.4f}")
    
    if final_fraud_count < 1:
        raise ValueError(f"❌ No fraud samples in fine-tuning data after sampling!")
    
    # 🔍 DEBUG: Vérifier les colonnes avant fine-tuning
    # print(f"🔍 DEBUG X_train_sample columns before fine-tuning: {list(X_train_sample.columns)}")
    # print(f"🔍 DEBUG X_test columns before fine-tuning: {list(X_test.columns)}")
    # print(f"🔍 DEBUG X_train_sample dtypes: {X_train_sample.dtypes.to_dict()}")
    
    # Chemin du modèle existant
    if ENV == "PROD":
        existing_model_path = resolve_path(f"models/{model_name}")
    else:
        # 🔧 Chercher d'abord dans shared_data (où sont sauvegardés les nouveaux modèles)
        shared_model_path = os.path.join("/app/shared_data", model_name)
        models_model_path = os.path.join("models", model_name)
        
        if os.path.exists(shared_model_path):
            existing_model_path = shared_model_path
            print(f"🔍 Using model from shared_data: {existing_model_path}")
        elif os.path.exists(models_model_path):
            existing_model_path = models_model_path
            print(f"🔍 Using model from models/: {existing_model_path}")
        else:
            raise FileNotFoundError(f"❌ Model not found in either {shared_model_path} or {models_model_path}")
    
    if not os.path.exists(existing_model_path):
        raise FileNotFoundError(f"❌ Model not found: {existing_model_path}")
    
    print(f"✅ Found existing model at: {existing_model_path}")
    
    # 🔍 DEBUG: Charger le modèle pour voir ses features
    # print(f"🔍 DEBUG: Checking existing model features...")
    temp_model = CatBoostClassifier()
    temp_model.load_model(existing_model_path)
    model_features = temp_model.feature_names_
    # print(f"🔍 DEBUG: Model expects features: {model_features}")
    # print(f"🔍 DEBUG: Data has features: {list(X_train_sample.columns)}")
    
    with mlflow.start_run():
        # Parameters à logger
        params = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "model_name": model_name,
            "timestamp": timestamp or "latest",
            "training_samples": len(X_train_sample),
            "fraud_ratio": float(y_train_sample.mean()),
            "environment": ENV
        }
        mlflow.log_params(params)
    
    # Fine-tuning
    model = fine_tune_model(
        existing_model_path=existing_model_path,
        X_train=X_train_sample,
        y_train=y_train_sample,
        X_val=X_test,
        y_val=y_test,
        learning_rate=learning_rate,
        epochs=epochs
    )
    
    # Évaluation
    report, auc = evaluate_model(model, X_test, y_test)
    
    # Sauvegarde du modèle fine-tuné
    model_path = save_model(model, model_name=model_name)
    
    metrics = {
        "roc_auc": auc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "training_time": time.time() - start_time
    }

    # Log model dans MLflow
    mlflow.catboost.log_model(
        model,
        "model",
        registered_model_name="fraud-detection-model"
    )

    run_id = mlflow.active_run().info.run_id
    print(f"📊 MLflow Run ID: {run_id}")
    
    print(f"✅ Fine-tuning complete!")
    print(f"🔍 DEBUG: model_path in response: {model_path}")
    print(f"📊 New AUC: {auc:.4f} | F1: {metrics['f1']:.4f}")
    
    return {
        "auc": auc,
        "metrics": metrics,
        "model_updated": True,
        "model_path": model_path,
        "mlflow_run_id": run_id,
        "model_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--model_name", type=str, default="catboost_model.cbm")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--timestamp", type=str)

    args = parser.parse_args()

    run_training(
        timestamp=args.timestamp,
        test=args.test,
        fast=args.fast,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()
