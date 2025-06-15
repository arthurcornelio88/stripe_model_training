import shutil
import os

MODEL_DIR = "mlruns/models"

if not os.path.exists(MODEL_DIR):
    print("❌ Pas de dossier de modèles trouvé.")
else:
    subdirs = [d for d in os.listdir(MODEL_DIR) if d.startswith("m-")]
    for subdir in subdirs:
        full_path = os.path.join(MODEL_DIR, subdir)
        print(f"🧹 Suppression : {full_path}")
        shutil.rmtree(full_path)

    print("✅ Modèles locaux cassés supprimés.")
