import pandas as pd
import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from dotenv import load_dotenv
import os
import argparse
from glob import glob
from mlflow.tracking import MlflowClient
import requests
from urllib.parse import urljoin
import json
import shutil



# Load environment
load_dotenv()

ENV = os.getenv("ENV", "DEV")
BUCKET = os.getenv("GCP_BUCKET")
PREFIX = os.getenv("GCP_DATA_PREFIX")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI").strip("/")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")


def gcs_path(filename):
    return f"gs://{BUCKET}/{PREFIX}/{filename}"


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
    - PROD: uses GCS fixed path
    """
    if ENV == "PROD":
        return gcs_path(f"processed/{name}")

    # DEV
    base_dir = "data/raw/" if io == "input" else "/app/shared_data/"
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
        return pd.read_csv(path)

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



def save_model(model, model_name="catboost_model.cbm"):
    output_path = (
        gcs_path(f"models/{model_name}") if ENV == "PROD"
        else os.path.join("models", model_name)
    )
    if ENV == "DEV":
        os.makedirs("models", exist_ok=True)
    model.save_model(output_path)
    print(f"💾 Model saved to: {output_path}")

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
    model = CatBoostClassifier()
    model.load_model(existing_model_path)
    
    print(f"✅ Model loaded. Starting fine-tuning with lr={learning_rate}, epochs={epochs}")
    
    # 🔧 IMPORTANT: Pour le fine-tuning, on ne peut pas passer 'iterations' à fit()
    # Il faut créer un nouveau modèle avec les bons paramètres
    fine_tuned_model = CatBoostClassifier(
        iterations=epochs,
        learning_rate=learning_rate,
        verbose=10,
        early_stopping_rounds=5,
        use_best_model=True,
        # Garder les mêmes paramètres que le modèle original
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        class_weights=[1, 25],
        # 🔧 FIX: Désactiver les logs CatBoost pour éviter les problèmes de permissions
        train_dir=None,  # Pas de répertoire de travail
        allow_writing_files=False  # Pas d'écriture de fichiers
    )
    
    # Fine-tuning = entraîner avec init_model
    fine_tuned_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        init_model=model  # 🔥 Utiliser le modèle existant comme point de départ
    )
    
    print("✅ Fine-tuning complete!")
    return fine_tuned_model

def run_fine_tuning(
    model_name: str = "catboost_model.cbm",
    timestamp: str = None,
    learning_rate: float = 0.01,
    epochs: int = 10
):
    """
    Run fine-tuning on an existing model with recent data.
    """
    print(f"🧠 Starting fine-tuning for model: {model_name}")
    
    # 🔧 PREPROCESSING D'ABORD ! Appeler l'endpoint de preprocessing
    print("🔄 First, calling preprocessing endpoint to prepare fresh data...")
    
    try:
        # Appeler l'endpoint de preprocessing
        preprocess_url = "http://localhost:8000/preprocess"
        response = requests.post(preprocess_url, json={}, timeout=300)  # 5 minutes timeout
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Preprocessing completed: {result}")
            
            # Récupérer le timestamp des données fraîches
            fresh_timestamp = result.get("timestamp")
            if fresh_timestamp:
                print(f"🔍 Using fresh preprocessed data with timestamp: {fresh_timestamp}")
            else:
                print("🔍 Using most recent preprocessed data")
        else:
            print(f"⚠️ Preprocessing failed with status {response.status_code}: {response.text}")
            print("🔄 Falling back to existing preprocessed data...")
            fresh_timestamp = None
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Could not call preprocessing endpoint: {e}")
        print("🔄 Falling back to existing preprocessed data...")
        fresh_timestamp = None
    
    # Charger les données (fraîches si preprocessing ok, sinon les plus récentes)
    X_train, X_test, y_train, y_test = load_data(timestamp=fresh_timestamp, test_mode=False)
    
    # 🧹 NETTOYER LES DONNÉES - Enlever les colonnes timestamp/string
    print("🧹 Cleaning training data for fine-tuning...")
    
    def clean_data(df):
        """Nettoyer les données comme dans le DAG"""
        df_clean = df.copy()
        cols_to_drop = []
        
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                # Garder seulement les colonnes catégorielles connues
                if col not in ["category", "merchant", "job", "state", "city_pop"]:
                    cols_to_drop.append(col)
        
        # Supprimer aussi les colonnes timestamp spécifiques
        cols_to_drop.extend(["ingestion_ts", "created_at", "updated_at"])
        cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
        
        if cols_to_drop:
            print(f"🧹 Removing columns: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        return df_clean
    
    # Nettoyer les datasets
    X_train = clean_data(X_train)
    X_test = clean_data(X_test)
    
    # 🔧 ALIGNEMENT EXACT DES COLONNES - Logique du DAG avec ORDRE PRÉSERVÉ
    # Charger le fichier de référence (fraudTest.csv) pour récupérer les colonnes du modèle original
    shared_dir = "/app/shared_data"
    ref_path = os.path.join(shared_dir, "fraudTest.csv")
    
    if os.path.exists(ref_path):
        print("📋 Loading reference data to match model columns...")
        df_ref = pd.read_csv(ref_path)
        
        # 🔧 ORDRE DE RÉFÉRENCE EXACT (selon le header du fichier)
        # Unnamed: 0,trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,street,city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,merch_long,is_fraud
        reference_order = [
            "Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
            "first", "last", "gender", "street", "city", "state", "zip", "lat", "long",
            "city_pop", "job", "dob", "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud"
        ]
        
        # Nettoyer le fichier de référence de la même façon
        df_ref_clean = clean_data(df_ref)
        
        # 🔧 CRITICAL: Utiliser l'ordre de référence exact, mais seulement les colonnes nettoyées
        # Trouver les colonnes communes dans l'ordre de référence
        common_cols = [col for col in reference_order if col in df_ref_clean.columns and col in X_train.columns]
        
        # Filtrer les données de fine-tuning pour avoir EXACTEMENT les mêmes colonnes dans le BON ORDRE
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        
        print(f"📊 Reference order: {reference_order[:10]}...")
        print(f"📊 Common columns for fine-tuning (in model order): {common_cols}")
        print(f"✅ Data aligned. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        
        # 🔍 Debug: Vérifier l'ordre des colonnes
        print(f"🔍 First 5 columns in order: {X_train.columns[:5].tolist()}")
        print(f"🔍 Reference first 5 columns: {df_ref_clean.columns[:5].tolist()}")
        
    else:
        print("⚠️ Reference file not found, falling back to basic cleaning...")
        # Fallback : supprimer uniquement les colonnes parasites
        if "Unnamed: 0" in X_train.columns:
            X_train = X_train.drop(columns=["Unnamed: 0"])
        if "Unnamed: 0" in X_test.columns:
            X_test = X_test.drop(columns=["Unnamed: 0"])
        
        print(f"✅ Data cleaned. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    print(f"🔍 Final X_train columns: {list(X_train.columns)}")
    
    # Prendre seulement une partie des données pour fine-tuning (ex: 20%)
    sample_size = min(len(X_train) // 5, 2000)  # Maximum 2000 samples
    X_train_sample = X_train.sample(n=sample_size, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
    
    print(f"📊 Using {len(X_train_sample)} samples for fine-tuning")
    
    # Chemin du modèle existant
    if ENV == "PROD":
        existing_model_path = gcs_path(f"models/{model_name}")
    else:
        existing_model_path = os.path.join("models", model_name)
    
    if not os.path.exists(existing_model_path):
        raise FileNotFoundError(f"❌ Model not found: {existing_model_path}")
    
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
    save_model(model, model_name=model_name)
    
    metrics = {
        "roc_auc": auc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    }
    
    print(f"✅ Fine-tuning complete!")
    print(f"📊 New AUC: {auc:.4f} | F1: {metrics['f1']:.4f}")
    
    return {
        "auc": auc,
        "metrics": metrics,
        "model_updated": True
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
