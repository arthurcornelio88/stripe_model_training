from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
from faker import Faker
from datetime import datetime
import random

app = FastAPI()
fake = Faker()

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
def get_transactions(n: int = Query(10, ge=1, le=1000, description="Number of transactions to generate (1-1000)")):
    return [generate_transaction() for _ in range(n)]
