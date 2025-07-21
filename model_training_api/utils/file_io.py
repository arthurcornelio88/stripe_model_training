import pandas as pd
from fastapi import HTTPException
from typing import Literal
import os
import gcsfs, time

def read_csv_flexible(path: str, env: Literal["DEV", "PROD"] = "DEV") -> pd.DataFrame:
    if env == "DEV":
        if not os.path.isabs(path):
            path = os.path.join("/app/shared_data", path)
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"File not found: {path}")
        return pd.read_csv(path)

    elif env == "PROD":
        from google.cloud import storage
        import io

        bucket_name = os.getenv("GCS_BUCKET")
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # 🧼 Corriger le chemin si c'est un chemin gs:// complet
        if path.startswith(f"gs://{bucket_name}/"):
            path = path.replace(f"gs://{bucket_name}/", "")
        elif path.startswith("gs://"):
            raise HTTPException(status_code=400, detail=f"Bucket mismatch or malformed path: {path}")

        blob = bucket.blob(path)

        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"GCS file not found: {path}")

        content = blob.download_as_bytes()
        return pd.read_csv(io.BytesIO(content))

    else:
        raise HTTPException(status_code=500, detail="Invalid ENV value")


def wait_for_gcs(path: str, timeout: int = 30):
    bucket_name = os.getenv("GCS_BUCKET")
    if path.startswith(f"gs://{bucket_name}/"):
        relative_path = path.replace(f"gs://{bucket_name}/", "")
    elif path.startswith("gs://"):
        raise ValueError(f"❌ Bucket mismatch or malformed GCS path: {path}")
    else:
        raise ValueError(f"❌ Not a valid GCS path: {path}")

    fs = gcsfs.GCSFileSystem(skip_instance_cache=True, cache_timeout=0)
    full_path = f"{bucket_name}/{relative_path}"  # gcsfs format is 'bucket_name/path'

    for i in range(timeout):
        if fs.exists(full_path):
            print(f"✅ GCS file detected: gs://{full_path}")
            return
        print(f"⏳ Waiting GCS ({i+1}/{timeout}) for: gs://{full_path}")
        time.sleep(1)

    raise FileNotFoundError(f"❌ GCS file still not found after {timeout}s: gs://{full_path}")
