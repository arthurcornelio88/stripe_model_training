import pandas as pd
from fastapi import HTTPException
from typing import Literal
import os

def read_csv_flexible(path: str, env: Literal["DEV", "PROD"] = "DEV") -> pd.DataFrame:
    if env == "DEV":
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"File not found: {path}")
        return pd.read_csv(path)
    
    elif env == "PROD":
        # 🚧 Préparé pour future intégration GCS
        from google.cloud import storage
        import io

        bucket_name = os.getenv("GCS_BUCKET")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(path)

        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"GCS file not found: {path}")

        content = blob.download_as_bytes()
        return pd.read_csv(io.BytesIO(content))
    
    else:
        raise HTTPException(status_code=500, detail="Invalid ENV value")
