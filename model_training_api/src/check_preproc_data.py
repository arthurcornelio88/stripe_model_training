import pandas as pd
import numpy as np
from dotenv import load_dotenv
from glob import glob
import os
import argparse

load_dotenv()

ENV = os.getenv("ENV", "DEV")
BUCKET = os.getenv("GCP_BUCKET")
PREFIX = os.getenv("GCP_DATA_PREFIX")

def gcs_path(filename):
    return f"gs://{BUCKET}/{PREFIX}/{filename}"

def resolve_latest_path(name, io="output", timestamp=None):
    base_dir = "data/raw/" if io == "input" else "data/processed/"
    if ENV == "PROD":
        return gcs_path(f"processed/{name}")
    elif timestamp:
        filename = name.replace(".csv", f"_{timestamp}.csv")
        return os.path.join(base_dir, filename)
    else:
        pattern = os.path.join(base_dir, name.replace(".csv", "_*.csv"))
        files = glob(pattern)
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

def check_types(df, name):
    print(f"\n📌 Types in {name}:")
    print(df.dtypes.value_counts())
    print("\n🔍 Object columns:")
    print(df.select_dtypes(include='object').columns.tolist())

def check_cardinality(df):
    print("\n🔎 High-cardinality columns:")
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique > 100 and df[col].dtype == "object":
            print(f"⚠️  {col}: {n_unique} unique values")

def check_fraud_distribution(y_train, y_test):
    print("\n📊 is_fraud distribution:")
    print("Train set:")
    print(y_train.value_counts(normalize=True))
    print("Test set:")
    print(y_test.value_counts(normalize=True))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, help="Force specific timestamp version (YYYYmmdd_HHMMSS)")
    args = parser.parse_args()

    print(f"🌍 ENV: {ENV}")

    # Load data
    X_train = pd.read_csv(resolve_latest_path("X_train.csv", timestamp=args.timestamp))
    X_test = pd.read_csv(resolve_latest_path("X_test.csv", timestamp=args.timestamp))
    y_train = pd.read_csv(resolve_latest_path("y_train.csv", timestamp=args.timestamp)).squeeze()
    y_test = pd.read_csv(resolve_latest_path("y_test.csv", timestamp=args.timestamp)).squeeze()

    # Basic checks
    print("\n✅ Data loaded:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape} | Positive: {(y_train == 1).sum()}")
    print(f"y_test:  {y_test.shape} | Positive: {(y_test == 1).sum()}")

    # Type & integrity
    check_types(X_train, "X_train")
    check_types(X_test, "X_test")

    # Object columns & risk
    check_cardinality(X_train)

    # Fraud class balance
    check_fraud_distribution(y_train, y_test)

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
