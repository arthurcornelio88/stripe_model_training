from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from faker import Faker
from datetime import datetime
import random
import pandas as pd

app = FastAPI()
fake = Faker()

# === Static data (replace path if needed)
FRAUD_DATA_PATH = "/app/shared_data/fraudTest.csv"

# Load once at startup
fraud_df = pd.read_csv(FRAUD_DATA_PATH, dtype={"cc_num": str, "zip": str})

# === Faker categories
CATEGORIES = ["personal_care", "health_fitness", "misc_pos", "travel", "food", "shopping", "gas_transport"]
JOBS = ["Engineer", "Teacher", "Artist", "Doctor", "Sales"]
STATES = ["CA", "NY", "TX", "FL", "IL"]

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

def generate_transaction() -> Transaction:
    now = datetime.now()
    lat, long = float(fake.latitude()), float(fake.longitude())
    merch_lat, merch_long = lat + random.uniform(-0.5, 0.5), long + random.uniform(-0.5, 0.5)
    return Transaction(
        trans_date_trans_time=now.strftime("%Y-%m-%d %H:%M:%S"),
        cc_num=fake.credit_card_number(),
        merchant=f"fraud_{fake.company()}",
        category=random.choice(CATEGORIES),
        amt=round(random.uniform(1, 500), 2),
        first=fake.first_name(),
        last=fake.last_name(),
        gender=random.choice(["M", "F"]),
        street=fake.street_address(),
        city=fake.city(),
        state=random.choice(STATES),
        zip=fake.zipcode(),
        lat=lat,
        long=long,
        city_pop=fake.random_int(min=1000, max=500000),
        job=random.choice(JOBS),
        dob=fake.date_of_birth().strftime("%Y-%m-%d"),
        trans_num=fake.uuid4().replace("-", ""),
        unix_time=int(now.timestamp()),
        merch_lat=merch_lat,
        merch_long=merch_long,
        is_fraud=random.choices([0, 1], weights=[0.98, 0.02])[0]
    )

@app.get("/transactions", response_model=List[Transaction])
def get_transactions(
    n: int = Query(10, ge=1, le=1000, description="Number of transactions"),
    variability: str = Query("high", enum=["low", "medium", "high"])
):
    if variability == "low":
        sampled = fraud_df.sample(n=n).to_dict(orient="records")
        print(f"[LOW] Returning {n} real transactions.")
        return sampled

    elif variability == "high":
        generated = [generate_transaction().dict() for _ in range(n)]
        print(f"[HIGH] Returning {n} synthetic transactions.")
        return generated

    elif variability == "medium":
        ratio_real = round(random.uniform(0.1, 0.9), 2)
        n_real = int(n * ratio_real)
        n_fake = n - n_real

        real_part = fraud_df.sample(n=n_real).to_dict(orient="records")
        fake_part = [generate_transaction().dict() for _ in range(n_fake)]

        combined = real_part + fake_part
        random.shuffle(combined)

        print(f"[MEDIUM] Returning {n_real} real & {n_fake} fake (ratio_real={ratio_real})")

        # Optionally return ratio in header
        response = JSONResponse(content=combined)
        response.headers["X-Ratio-Real"] = str(ratio_real)
        return response
