import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

from src.preprocessing import preprocess, encode_and_split, haversine_distance


@pytest.fixture
def sample_data():
    """Create a minimal valid fake dataset"""
    data = {
        "trans_date_trans_time": ["2020-06-21 12:14:25"] * 4,
        "cc_num": [1234, 5678, 9101, 1121],
        "merchant": ["merchant_a", "merchant_b", "merchant_a", "merchant_c"],
        "category": ["grocery_pos", "shopping_net", "grocery_pos", "misc_net"],
        "amt": [100.0, 250.0, 15.0, 40.0],
        "first": ["John"] * 4,
        "last": ["Doe"] * 4,
        "gender": ["M", "F", "M", "F"],
        "street": ["123 St"] * 4,
        "city": ["CityX"] * 4,
        "state": ["CA", "TX", "CA", "NY"],
        "zip": [90001, 73301, 90001, 10001],
        "lat": [34.05, 30.26, 34.05, 40.71],
        "long": [-118.24, -97.74, -118.24, -74.00],
        "city_pop": [100000, 200000, 100000, 300000],
        "job": ["Engineer", "Doctor", "Engineer", "Artist"],
        "dob": ["1980-01-01", "1990-01-01", "1980-01-01", "1975-01-01"],
        "trans_num": ["txn1", "txn2", "txn3", "txn4"],
        "unix_time": [1234567890] * 4,
        "merch_lat": [34.06, 30.30, 34.06, 40.73],
        "merch_long": [-118.23, -97.78, -118.23, -73.99],
        "is_fraud": [0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_preprocess_structure(sample_data):
    df_clean = preprocess(sample_data.copy())
    assert isinstance(df_clean, pd.DataFrame)
    assert "age" in df_clean.columns
    assert "distance_km" in df_clean.columns
    assert "amt" in df_clean.columns
    assert "is_fraud" in df_clean.columns
    assert not {"first", "last", "street", "trans_num"}.intersection(df_clean.columns)


def test_haversine_output():
    lat1, lon1 = 34.05, -118.25
    lat2, lon2 = 34.06, -118.26
    d = haversine_distance(lat1, lon1, lat2, lon2)
    assert isinstance(d, float)
    assert d > 0


def test_encoding_output(sample_data, tmp_path):
    df_clean = preprocess(sample_data)
    encode_and_split(df_clean, output_dir=tmp_path, test_size=0.5)  # force 50/50 split

    for name in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        path = tmp_path / name
        assert path.exists()
        df = pd.read_csv(path)
        assert not df.empty


def test_stratified_split_preserves_class_ratio(sample_data):
    df_clean = preprocess(sample_data)
    y = df_clean["is_fraud"]
    X = df_clean.drop(columns=["is_fraud"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.5, random_state=42
    )
    assert abs(y_train.mean() - y.mean()) < 0.1
    assert abs(y_test.mean() - y.mean()) < 0.1
