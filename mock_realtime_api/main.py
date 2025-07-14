from fastapi import FastAPI, Query
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
from faker import Faker
from datetime import datetime
import random
import pandas as pd
import numpy as np
import os

app = FastAPI()
fake = Faker()

# Environment-aware data loading
ENV = os.getenv("ENV", "DEV")
if ENV == "PROD":
    # In production, data comes from GCS via StorageManager or mounted volume
    FRAUD_DATA_PATH = os.getenv("SHARED_DATA_PATH", "/app/shared_data") + "/fraudTest.csv"
else:
    # In development, use local path
    FRAUD_DATA_PATH = "/app/shared_data/fraudTest.csv"

print(f"🔄 Mock API loading data from: {FRAUD_DATA_PATH}")

# === Static data
try:
    fraud_df = pd.read_csv(FRAUD_DATA_PATH, dtype={"cc_num": str, "zip": str})
    print(f"✅ Loaded {len(fraud_df)} fraud records")
except Exception as e:
    print(f"❌ Error loading fraud data: {e}")
    # Fallback to minimal data
    fraud_df = pd.DataFrame({"category": ["grocery"], "job": ["engineer"], "state": ["CA"], 
                           "gender": ["M"], "is_fraud": [0], "merchant": ["Test"], 
                           "city": ["SF"], "lat": [37.7], "long": [-122.4]})

# === Precomputed distributions
category_dist = fraud_df["category"].value_counts(normalize=True)
job_dist = fraud_df["job"].value_counts(normalize=True)
state_dist = fraud_df["state"].value_counts(normalize=True)
gender_dist = fraud_df["gender"].value_counts(normalize=True)
fraud_dist = fraud_df["is_fraud"].value_counts(normalize=True)
merchant_pool = fraud_df["merchant"].dropna().unique().tolist()
city_pool = fraud_df[["city", "lat", "long"]].dropna().drop_duplicates()

# === Model
class Transaction(BaseModel):
    trans_date_trans_time: str
    cc_num: str
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: str
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    unix_time: int
    merch_lat: float
    merch_long: float
    is_fraud: int

# === Helpers
def perturb(value, scale):
    return value + random.gauss(0, scale)

def perturb_float(value, scale, min_value=0.01):
    return round(max(min_value, value + random.gauss(0, scale)), 2)

def generate_like(row: pd.Series, variability: float) -> dict:
    now = datetime.now()
    lat = perturb(row["lat"], variability * 0.001)
    long = perturb(row["long"], variability * 0.001)
    return {
        "trans_date_trans_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": row["cc_num"],
        "merchant": random.choices(merchant_pool, k=1)[0],
        "category": random.choices(category_dist.index, weights=category_dist.values)[0],
        "amt": perturb_float(row["amt"], variability * 10),
        "first": row["first"],
        "last": row["last"],
        "gender": random.choices(gender_dist.index, weights=gender_dist.values)[0],
        "street": row["street"],
        "city": row["city"],
        "state": random.choices(state_dist.index, weights=state_dist.values)[0],
        "zip": row["zip"],
        "lat": lat,
        "long": long,
        "city_pop": int(perturb(row["city_pop"], variability * 1000)),
        "job": random.choices(job_dist.index, weights=job_dist.values)[0],
        "dob": row["dob"],
        "trans_num": fake.uuid4().replace("-", ""),
        "unix_time": int(now.timestamp()),
        "merch_lat": lat + random.uniform(-0.001, 0.001),
        "merch_long": long + random.uniform(-0.001, 0.001),
        "is_fraud": random.choices(fraud_dist.index, weights=fraud_dist.values)[0],
    }

def generate_transactions(n: int, variability: float) -> List[dict]:
    synthetic = []
    rows = fraud_df.sample(n=n).to_dict(orient="records")
    
    for row in rows:
        tx = generate_like(pd.Series(row), variability)
        synthetic.append(tx)
    return synthetic

@app.get("/transactions", response_model=List[Transaction])
def get_transactions(
    n: int = Query(10, ge=1, le=1000, description="Number of transactions to generate"),
    variability: float = Query(0.0, ge=0.0, le=1.0, description="Variability: 0.0 = real, 1.0 = random synthetic")
):
    try:
        if variability <= 0.0:
            # Retourne des vraies lignes du CSV
            sampled = fraud_df.sample(n=n, random_state=42).to_dict(orient="records") # #fraud_df.sample(n=n).to_dict(orient="records")
            print(f"[VAR={variability:.2f}] Returning {n} real transactions.")
            return sampled

        elif variability >= 1.0:
            # Mode 100% aléatoire
            generated = [
                Transaction(
                    trans_date_trans_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    cc_num=fake.credit_card_number(),
                    merchant=random.choice(merchant_pool),
                    category=random.choice(category_dist.index.tolist()),
                    amt=round(random.uniform(1, 500), 2),
                    first=fake.first_name(),
                    last=fake.last_name(),
                    gender=random.choice(gender_dist.index.tolist()),
                    street=fake.street_address(),
                    city=random.choice(city_pool["city"].tolist()),
                    state=random.choice(state_dist.index.tolist()),
                    zip=fake.zipcode(),
                    lat=float(fake.latitude()),
                    long=float(fake.longitude()),
                    city_pop=random.randint(1000, 500000),
                    job=random.choice(job_dist.index.tolist()),
                    dob=fake.date_of_birth().strftime("%Y-%m-%d"),
                    trans_num=fake.uuid4().replace("-", ""),
                    unix_time=int(datetime.now().timestamp()),
                    merch_lat=float(fake.latitude()),
                    merch_long=float(fake.longitude()),
                    is_fraud=random.choices(fraud_dist.index, weights=fraud_dist.values)[0]
                ).dict()
                for _ in range(n)
            ]
            print(f"[VAR={variability:.2f}] Returning {n} fully synthetic transactions.")
            return generated

        else:
            # Génération réaliste par mutation légère
            generated = generate_transactions(n=n, variability=variability)
            print(f"[VAR={variability:.2f}] Returning {n} synthetic-like transactions.")
            return generated

    except Exception as e:
        print(f"❌ Internal error during generation: {e}")
        raise HTTPException(status_code=500, detail="Transaction generation failed.")
