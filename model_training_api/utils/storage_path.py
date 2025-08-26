import os

def get_storage_path(subdir: str, filename: str) -> str:
    """
    Returns the correct storage path for DEV (local) or PROD (GCS) environments.
    - DEV: /app/{subdir}/{filename}
    - PROD: gs://{GCS_BUCKET}/{subdir}/{filename}
    """
    ENV = os.getenv("ENV", "DEV")
    GCS_BUCKET = os.getenv("GCS_BUCKET", "fraud-detection-jedha2024")
    if ENV == "PROD":
        return f"gs://{GCS_BUCKET}/{subdir}/{filename}"
    else:
        return f"/app/{subdir}/{filename}"
