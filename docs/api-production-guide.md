Voici une **refonte (refactor)** plus claire et concise de ton guide "API Production & Local Dev Guide", pour éviter les doublons et simplifier la navigation.
J'ai conservé toutes les commandes essentielles, distingué net **local** et **cloud**, et recentré chaque bloc autour d’un workflow logique (prétraitement, entraînement, prédiction, etc).
La structure « Table des matières / Workflows / Endpoints / Dépannage » est conservée, mais tout ce qui est en double est regroupé ou déplacé en ressources annexes.

---

# 🚀 API Guide — Local & Production

**Mise à jour : 17 juillet 2025**

---

## 📋 Table des matières

1. [Lancer les services](#lancer-les-services)
2. [Workflows courants](#workflows-courants)
3. [Endpoints & Exemples](#endpoints--exemples)
4. [Monitoring et Debug](#monitoring-et-debug)
5. [Dépannage](#dépannage)
6. [Ressources utiles](#ressources-utiles)

---

## 1. Lancer les services

### 🔹 En développement local

```bash
docker compose up --build
```

* **model-api**: [http://localhost:8000](http://localhost:8000)
* **mlflow**: [http://localhost:5000](http://localhost:5000)
* **mock-api**: [http://localhost:8001](http://localhost:8001)

Accès rapide :

* Swagger (API docs) : [http://localhost:8000/docs](http://localhost:8000/docs)
* MLflow UI : [http://localhost:5000](http://localhost:5000)

---

### 🔹 En production (Cloud Run)

Les commandes curl restent identiques, il suffit de remplacer l’URL (voir tableau ci-dessous).

| Service      | Local                                          | Production Cloud Run                                                                                     |
| ------------ | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Training API | [http://localhost:8000](http://localhost:8000) | [https://mlops-training-api-bxzifydblq-ew.a.run.app](https://mlops-training-api-bxzifydblq-ew.a.run.app) |
| Mock API     | [http://localhost:8001](http://localhost:8001) | [https://mlops-mock-api-bxzifydblq-ew.a.run.app](https://mlops-mock-api-bxzifydblq-ew.a.run.app)         |
| MLflow UI    | [http://localhost:5000](http://localhost:5000) | [https://mlops-mlflow-bxzifydblq-ew.a.run.app](https://mlops-mlflow-bxzifydblq-ew.a.run.app)             |

---

## 2. Workflows courants

### 🔄 **Pipeline standard (dev & prod) :**

1. **Vérifier la santé de l’API**
2. **Prétraiter les données**
3. **Entraîner un modèle**
4. **Valider le modèle**
5. **Faire des prédictions**
6. **Monitorer le drift**

*(Remplacer les chemins locaux par des chemins GCS en prod)*

---

## 3. Endpoints & Exemples

Tous les endpoints sont les mêmes pour local ou prod, seul l’URL change.

### 1. 🏥 Health Check

```bash
curl http://localhost:8000/ping
# ou en prod
curl https://mlops-training-api-bxzifydblq-ew.a.run.app/ping
# -> {"status": "alive"}
```

---

### 2. 🔄 Prétraitement

```bash
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "data/raw/fraudTest.csv",
    "output_dir": "data/processed",
    "log_amt": true
  }'
```

* En prod, remplacer les chemins par des chemins GCS (`gs://...`).

---

### 3. 🤖 Entraînement

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250715_195232",
    "learning_rate": 0.1,
    "epochs": 50
  }'
```

* Réponse : Métriques + chemin du modèle.

---

### 4. 🔍 Validation

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "catboost_model_20250715_195232.cbm",
    "timestamp": "20250715_195232"
  }'
```

---

### 5. 🔮 Prédiction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "data/processed/X_pred_20250715_195232.csv",
    "model_name": "catboost_model_20250715_195232.cbm",
    "output_path": "data/predictions.csv"
  }'
```

---

### 6. 📊 Monitoring (Data Drift)

```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "data/processed/X_test_20250715_195232.csv",
    "current_path": "data/processed/X_pred_20250715_195232.csv",
    "output_html": "reports/data_drift.html"
  }'
```

* Réponse : résumé du drift, chemin du rapport HTML.

---

## 4. Monitoring et Debug

* **Swagger** :

  * Local : [http://localhost:8000/docs](http://localhost:8000/docs)
  * Prod : [https://mlops-training-api-bxzifydblq-ew.a.run.app/docs](https://mlops-training-api-bxzifydblq-ew.a.run.app/docs)
* **MLflow UI** :

  * Local : [http://localhost:5000](http://localhost:5000)
  * Prod : [https://mlops-mlflow-bxzifydblq-ew.a.run.app](https://mlops-mlflow-bxzifydblq-ew.a.run.app)

**Logs Cloud Run** :

```bash
gcloud run services logs tail mlops-training-api --region=europe-west1
```

---

## 5. Dépannage

### Problèmes fréquents

* **Erreur 500** : vérifier les logs avec `gcloud run services logs read ...`
* **Timeout training** : augmenter `--max-time` ou réduire `epochs`
* **MLflow KO** : pinguer `/health` ou vérifier secrets GCP
* **GCS inaccessible** : vérifier IAM du service account

### Diagnostic rapide

```bash
curl -I https://mlops-training-api-bxzifydblq-ew.a.run.app/ping
```

---

## 6. Ressources utiles

* [FastAPI](https://fastapi.tiangolo.com/)
* [MLflow](https://mlflow.org/docs/latest/index.html)
* [Google Cloud Run](https://cloud.google.com/run/docs)
* [CatBoost](https://catboost.ai/docs/)

---

### **Notes pratiques**

* Les chemins :

  * Local : `/app/shared_data/...` ou `data/processed/...`
  * Prod : `gs://...`
* Variables d’environnement : voir `.env` local et secrets GCP pour la prod
* Authentification :

  * Dev : fichiers de credentials à placer localement
  * Prod : Cloud Run déployé en `--allow-unauthenticated` (à renforcer pour vrai prod)