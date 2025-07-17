import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from dotenv import load_dotenv
import argparse
import os
from model_training_api.utils.storage_path import get_storage_path
from datetime import datetime

# Load environment variables
load_dotenv()

ENV = os.getenv("ENV", "DEV")
BUCKET = os.getenv("GCP_BUCKET")
SHARED_DATA_PATH = os.getenv("SHARED_DATA_PATH")


def resolve_base_path(relative_path):
    """Resolve file path based on environment"""
    if ENV == "PROD":
        return f"gs://{BUCKET}/{SHARED_DATA_PATH}/{relative_path}"
    else:
        return f"/app/shared_data/{relative_path}"


def resolve_path(filename, io="input"):
    """
    Resolves the correct path based on ENV.

    DEV: local path (data/raw or data/processed)
    PROD: GCS path using GCP_BUCKET and SHARED_DATA_PATH
    """
    sub = "raw/" if io == "input" else "processed/"
    return resolve_base_path(f"{sub}{filename}") if ENV == "PROD" else os.path.join(f"data/{sub}", filename)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes the haversine distance (in kilometers) between two geographic points.
    """
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def preprocess(df: pd.DataFrame, log_amt=True) -> pd.DataFrame:
    """
    Cleans and enriches the fraud dataset with feature engineering.

    - Drops irrelevant columns
    - Creates age, temporal features, and distance
    - Applies log transformation to 'amt' if enabled
    """
    # Supprimer seulement les colonnes qui existent
    cols_to_drop = ["first", "last", "street", "trans_num", "unix_time", "city"]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"🧹 Dropped columns: {existing_cols_to_drop}")
    
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    # Feature engineering
    df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month

    df = df.drop(columns=["trans_date_trans_time", "dob"])

    df["distance_km"] = haversine_distance(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    df = df.drop(columns=["lat", "long", "merch_lat", "merch_long"])

    if log_amt:
        df["amt"] = np.log1p(df["amt"])

    return df


def encode_and_split(df: pd.DataFrame, output_dir: str, test_size=0.2, random_state=42) -> str:
    """
    Splits the dataset and applies target encoding to categorical columns.
    Saves all files to output_dir with a timestamp suffix.

    Returns:
        str: The timestamp used for the file suffix
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Encoding
    cat_cols = ["category", "job", "state", "merchant", "gender"]
    encoder = TargetEncoder(cols=cat_cols)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    # Ensure output path exists if local
    if ENV == "DEV":
        os.makedirs(output_dir, exist_ok=True)

    # 🐛 DEBUG: Vérifier les colonnes avant sauvegarde
    print(f"🔍 DEBUG X_train columns before save: {list(X_train.columns)}")
    print(f"🔍 DEBUG X_train shape: {X_train.shape}")
    print(f"🔍 DEBUG X_train index: {X_train.index.name}")
    
    # Save files
    for name, data in {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }.items():
        data = data.reset_index(drop=True)
        filename = get_storage_path("shared_data/preprocessed", f"{name}_{timestamp}.csv")
        print(f"🔍 DEBUG Saving {name} with columns: {list(data.columns) if hasattr(data, 'columns') else 'Series'}")
        data.to_csv(filename, index=False)

    print(f"✅ Data saved to {output_dir}")
    print(f"➡️  Train size: {len(X_train)} | Test size: {len(X_test)} | Fraud ratio train: {y_train.mean():.4f}")
    return timestamp

def encode_full_data(df: pd.DataFrame, output_dir: str) -> str:
    """
    Encode the full dataset using target encoding, without splitting.
    Saves X_pred and y_pred.

    Returns:
        str: Timestamp used
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])

    cat_cols = ["category", "job", "state", "merchant", "gender"]
    encoder = TargetEncoder(cols=cat_cols)
    X[cat_cols] = encoder.fit_transform(X[cat_cols], y)

    os.makedirs(output_dir, exist_ok=True)
    print(f"🔄 Splitting and encoding data, saving to {output_dir}")

    X.to_csv(get_storage_path("shared_data/preprocessed", f"X_pred_{timestamp}.csv"), index=False)
    y.to_csv(get_storage_path("shared_data/preprocessed", f"y_pred_{timestamp}.csv"), index=False)

    print(f"✅ Prediction data saved to {output_dir}")
    print(f"➡️  Rows: {len(X)} | Positive ratio: {y.mean():.4f}")
    return timestamp

def run_preprocessing(
    input_path: str = "data/raw/fraudTest.csv",
    output_dir: str = "data/processed",
    log_amt: bool = True,
    for_prediction: bool = False
) -> str:
    """
    Fonction centrale pour API et CLI.
    Effectue le prétraitement (split ou non) selon le mode.
    """
    print(f"🔄 ENV = {ENV} | Reading from: {input_path}")

    df = pd.read_csv(input_path)
    df_clean = preprocess(df, log_amt=log_amt)

    if for_prediction:
        print("🟢 Prediction-only mode (no train/test split)")
        timestamp = encode_full_data(df_clean, output_dir)
    else:
        print("🧪 Training mode (split + encode)")
        timestamp = encode_and_split(df_clean, output_dir)

    return timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=get_storage_path("shared_data/raw", "fraudTest.csv"))
    parser.add_argument("--output_dir", type=str, default=get_storage_path("shared_data/preprocessed", ""))
    parser.add_argument("--no_log_amt", action="store_true")
    parser.add_argument("--for_prediction", action="store_true")

    args = parser.parse_args()

    run_preprocessing(
        input_path=args.input_path,
        output_dir=args.output_dir,
        log_amt=not args.no_log_amt,
        for_prediction=args.for_prediction
    )


if __name__ == "__main__":
    main()
