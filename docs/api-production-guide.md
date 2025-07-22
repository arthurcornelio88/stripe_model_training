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

**Local:**
```bash
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/shared_data/fraudTest.csv",
    "output_dir": "/shared_data/preprocessed",
    "log_amt": true
  }'
```
**Production :**

- Preprocessed dataset for predictions:

```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/fraudTest.csv",
    "output_dir": "gs://fraud-detection-jedha2024/preprocessed",
    "log_amt": true
  }'
```

- Preprocessed dataset for full model training: 

```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/fraudTest.csv",
    "output_dir": "gs://fraud-detection-jedha2024/preprocessed",
    "log_amt": true,
    "for_prediction": false
  }'

```

---

### 3. 🤖 Entraînement

**Local:**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250721_153507",
    "learning_rate": 0.1,
    "epochs": 50
  }'
```
**Production:**

- Full training: 

```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250722_100739",
    "learning_rate": 0.1,
    "epochs": 50
  }'
```
> "test": Defaut is `false`. If `true`, `X_raw` sample size - 5000 rows
> "fast": Defaut is `false`. If `true`, training params configured to a fast training (bad metrics).

- Fine tuning:

```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250722_100739",
    "learning_rate": 0.1,
    "epochs": 50,
    "mode": "fine_tune"
  }'
```

> Takes the last model and fine-tunes it with your selected data.

* Réponse : Métriques + chemin du modèle.

---

### 4. 🔍 Validation

**Local:**
```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "catboost_model_20250715_210730.cbm",
    "timestamp": "20250715_195232"
  }'
```
**Production:**
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "catboost_model_20250715_210730.cbm",
    "timestamp": "20250721_153507"
  }'
```

---

### 5. 🔮 Prediction

**Local:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/shared_data/preprocessed/X_pred_20250715_195232.csv",
    "model_name": "catboost_model_20250715_210730.cbm",
    "output_path": "/shared_data/predictions/predictions_20250715.csv"
  }'
```
**Production:**
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250721_153507.csv",
    "model_name": "catboost_model_20250715_210730.cbm",
    "output_path": "gs://fraud-detection-jedha2024/shared_data/predictions/predictions_20250721.csv"
  }'
```

---

### 6. 📊 Monitoring (Data Drift)

**Local:**
```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "data/processed/X_test_20250715_195232.csv",
    "current_path": "data/processed/X_pred_20250715_195232.csv",
    "output_html": "reports/data_drift.html"
  }'
```
**Production:**
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "fraudTest.csv",
    "current_path": "shared_data/tmp/raw_sample_20250721.csv",
    "output_html": "reports/data_drift.html"
  }'
```
 \

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