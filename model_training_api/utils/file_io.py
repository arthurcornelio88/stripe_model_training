import pandas as pd
from fastapi import HTTPException
from typing import Literal
import os
import gcsfs, time
import hashlib
from google.cloud import storage

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

def download_model_from_gcs(gcs_uri, cache_dir="/tmp/model_cache"):
    """
    Télécharge un modèle depuis GCS si non déjà téléchargé.
    Utilise un cache local dans /tmp/model_cache.

    Returns:
        str: chemin local du modèle
    """
    assert gcs_uri.startswith("gs://"), f"❌ Invalid GCS URI: {gcs_uri}"

    # Créer un identifiant unique pour ce chemin GCS
    gcs_hash = hashlib.md5(gcs_uri.encode()).hexdigest()
    filename = os.path.basename(gcs_uri)
    cached_model_path = os.path.join(cache_dir, f"{gcs_hash}_{filename}")

    # Si déjà téléchargé, réutiliser
    if os.path.exists(cached_model_path):
        print(f"📦 Using cached model at: {cached_model_path}")
        return cached_model_path

    # Sinon, télécharger
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📥 Downloading model from GCS: {gcs_uri}")
    bucket_name, *blob_parts = gcs_uri[5:].split("/")
    blob_path = "/".join(blob_parts)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(cached_model_path)

    print(f"✅ Model downloaded and cached at: {cached_model_path}")
    return cached_model_path
