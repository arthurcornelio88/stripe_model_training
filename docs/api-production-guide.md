# 🚀 MLOps Fraud Detection API - Guide Complet

## 📋 Table des Matières

1. [Configuration et Déploiement](#configuration-et-déploiement)
2. [URLs des Services](#urls-des-services)
3. [Authentification](#authentification)
4. [Endpoints API](#endpoints-api)
5. [Exemples d'Usage](#exemples-dusage)
6. [Monitoring et Debug](#monitoring-et-debug)
7. [Troubleshooting](#troubleshooting)

---

## 📦 Configuration et Déploiement

### Mode Développement (Local)
```bash
# Démarrer tous les services localement
cd model_training/
docker compose up

# Services disponibles:
# - Training API: http://localhost:8000
# - MLflow UI: http://localhost:5000
```

### Mode Production (Cloud Run)
```bash
# Déployer tous les services
cd model_training/deployment/
./deploy_all_services.sh

# Ou déployer seulement l'API de training
./deploy_training_only.sh
```

---

## 🌐 URLs des Services

### 🔧 Développement (Local)
```bash
# API de Training
TRAINING_API_URL="http://localhost:8000"

# Interface MLflow
MLFLOW_UI_URL="http://localhost:5000"
```

### 🚀 Production (Cloud Run)
```bash
# API de Training
TRAINING_API_URL="https://mlops-training-api-bxzifydblq-ew.a.run.app"

# API Mock (génération de données de test)
MOCK_API_URL="https://mlops-mock-api-bxzifydblq-ew.a.run.app"

# Interface MLflow
MLFLOW_UI_URL="https://mlops-mlflow-bxzifydblq-ew.a.run.app"
```

---

## 🔐 Authentification

### Local
Aucune authentification requise.

### Production
Les services Cloud Run sont configurés avec `--allow-unauthenticated` pour simplifier l'accès. En production réelle, vous devriez configurer l'authentification IAM.

---

## 🔌 Endpoints API

### 1. 🏥 Health Check

**Endpoint**: `GET /health` ou `GET /ping`
**But**: Vérifier que l'API est opérationnelle

#### 💻 Local
```bash
curl http://localhost:8000/health
```

#### ☁️ Production
```bash
curl https://mlops-training-api-bxzifydblq-ew.a.run.app/health
```

**Réponse**:
```json
{"status": "alive"}
```

---

### 2. 🔄 Preprocessing

**Endpoint**: `POST /preprocess`
**But**: Préprocesser les données brutes pour l'entraînement

#### 📝 Corps de la requête
```json
{
  "input_path": "data/raw/fraudTest.csv",
  "output_dir": "data/processed",
  "log_amt": true
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "data/raw/fraudTest.csv",
    "output_dir": "data/processed",
    "log_amt": true
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/fraudTest.csv",
    "output_dir": "gs://fraud-detection-jedha2024/shared_data/processed",
    "log_amt": true
  }' \
  --max-time 300
```

---

### 3. 🤖 Training

**Endpoint**: `POST /train`
**But**: Entraîner un modèle de détection de fraude

#### 📝 Corps de la requête
```json
{
  "timestamp": "20250715_195232",
  "learning_rate": 0.1,
  "epochs": 100,
  "test": false
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250715_195232",
    "learning_rate": 0.1,
    "epochs": 100,
    "test": false
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250715_195232",
    "learning_rate": 0.1,
    "epochs": 100,
    "test": false
  }' \
  --max-time 600
```

**Réponse**:
```json
{
  "status": "training complete",
  "mode": "full_train",
  "model_path": "gs://fraud-detection-jedha2024/models/catboost_model_20250715_195232.cbm",
  "metrics": {
    "auc": 0.9234,
    "f1_score": 0.8567,
    "precision": 0.8901,
    "recall": 0.8245
  }
}
```

---

### 4. 🔍 Validation

**Endpoint**: `POST /validate`
**But**: Évaluer les performances d'un modèle

#### 📝 Corps de la requête
```json
{
  "model_name": "catboost_model_20250715_195232.cbm",
  "timestamp": "20250715_195232"
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "catboost_model_20250715_195232.cbm",
    "timestamp": "20250715_195232"
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "catboost_model_20250715_195232.cbm",
    "timestamp": "20250715_195232"
  }' \
  --max-time 300
```

---

### 5. 🔮 Prediction

**Endpoint**: `POST /predict`
**But**: Faire des prédictions sur de nouvelles données

#### 📝 Corps de la requête
```json
{
  "input_path": "data/processed/X_pred_20250715_195232.csv",
  "model_name": "catboost_model_20250715_195232.cbm",
  "output_path": "data/predictions.csv"
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "data/processed/X_pred_20250715_195232.csv",
    "model_name": "catboost_model_20250715_195232.cbm",
    "output_path": "data/predictions.csv"
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250715_195232.csv",
    "model_name": "catboost_model_20250715_195232.cbm",
    "output_path": "gs://fraud-detection-jedha2024/shared_data/predictions.csv"
  }' \
  --max-time 300
```

---

### 6. 📊 Monitoring (Data Drift)

**Endpoint**: `POST /monitor`
**But**: Détecter la dérive des données entre deux jeux de données

#### 📝 Corps de la requête
```json
{
  "reference_path": "data/processed/X_test_20250715_195232.csv",
  "current_path": "data/processed/X_pred_20250715_195232.csv",
  "output_html": "reports/data_drift.html"
}
```

#### 💻 Local
```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "data/processed/X_test_20250715_195232.csv",
    "current_path": "data/processed/X_pred_20250715_195232.csv",
    "output_html": "reports/data_drift.html"
  }'
```

#### ☁️ Production
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "gs://fraud-detection-jedha2024/shared_data/processed/X_test_20250715_195232.csv",
    "current_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250715_195232.csv",
    "output_html": "gs://fraud-detection-jedha2024/shared_data/reports/data_drift.html"
  }' \
  --max-time 300
```

**Réponse**:
```json
{
  "drift_summary": {
    "drift_detected": false,
    "drift_score": 0.1234,
    "features_drifted": []
  },
  "report_path": "gs://fraud-detection-jedha2024/shared_data/reports/data_drift.html"
}
```

---

## 🎯 Exemples d'Usage

### Workflow Complet en Production

```bash
# 1. Vérifier la santé de l'API
curl https://mlops-training-api-bxzifydblq-ew.a.run.app/health

# 2. Préprocesser les données
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/fraudTest.csv",
    "output_dir": "gs://fraud-detection-jedha2024/shared_data/processed",
    "log_amt": true
  }' \
  --max-time 300

# 3. Entraîner le modèle
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250715_195232",
    "learning_rate": 0.1,
    "epochs": 50
  }' \
  --max-time 600

# 4. Faire des prédictions
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250715_195232.csv",
    "model_name": "catboost_model_20250715_195232.cbm",
    "output_path": "gs://fraud-detection-jedha2024/shared_data/predictions.csv"
  }' \
  --max-time 300

# 5. Surveiller la dérive des données
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "gs://fraud-detection-jedha2024/shared_data/processed/X_test_20250715_195232.csv",
    "current_path": "gs://fraud-detection-jedha2024/shared_data/preprocessed/X_pred_20250715_195232.csv",
    "output_html": "gs://fraud-detection-jedha2024/shared_data/reports/data_drift.html"
  }' \
  --max-time 300
```

### Script de Test Automatisé

```bash
#!/bin/bash
# test_production_api.sh

API_URL="https://mlops-training-api-bxzifydblq-ew.a.run.app"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 Testing Production API: $API_URL"
echo "📅 Timestamp: $TIMESTAMP"

# Test Health Check
echo "🏥 Testing health check..."
curl -s "$API_URL/health" | jq .

# Test Training
echo "🤖 Testing training..."
curl -X POST "$API_URL/train" \
  -H "Content-Type: application/json" \
  -d "{
    \"timestamp\": \"$TIMESTAMP\",
    \"learning_rate\": 0.1,
    \"epochs\": 10
  }" \
  --max-time 600 | jq .

echo "✅ Tests completed!"
```

---

## 📊 Monitoring et Debug

### Interface Swagger
- **Local**: http://localhost:8000/docs
- **Production**: https://mlops-training-api-bxzifydblq-ew.a.run.app/docs

### MLflow UI
- **Local**: http://localhost:5000
- **Production**: https://mlops-mlflow-bxzifydblq-ew.a.run.app

### Vérification des Logs
```bash
# Logs Cloud Run
gcloud run services logs read mlops-training-api --region=europe-west1

# Logs en temps réel
gcloud run services logs tail mlops-training-api --region=europe-west1
```

### Variables d'Environnement de Debug
```bash
# Activer le debug des variables d'environnement
gcloud run services update mlops-training-api \
  --set-env-vars DEBUG_ENV=true \
  --region=europe-west1
```

---

## 🔧 Troubleshooting

### Problèmes Courants

#### 1. Erreur 500 - Internal Server Error
```bash
# Vérifier les logs
gcloud run services logs read mlops-training-api --region=europe-west1 --limit=50

# Vérifier les variables d'environnement
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/debug/env
```

#### 2. Timeout lors de l'entraînement
```bash
# Augmenter le timeout de la requête
curl ... --max-time 900  # 15 minutes

# Réduire le nombre d'epochs pour tester
{
  "epochs": 5,
  "learning_rate": 0.1
}
```

#### 3. Problème de connexion MLflow
```bash
# Vérifier l'état du service MLflow
curl https://mlops-mlflow-bxzifydblq-ew.a.run.app/health

# Vérifier les secrets
gcloud secrets versions access latest --secret="mlflow-tracking-uri"
```

#### 4. Problème d'accès aux données GCS
```bash
# Vérifier les permissions du service account
gcloud projects get-iam-policy jedha2024 \
  --flatten="bindings[].members" \
  --format="value(bindings.role)" \
  --filter="bindings.members:mlops-service-account@jedha2024.iam.gserviceaccount.com"
```

### Commandes de Diagnostic

```bash
# Redémarrer le service
gcloud run services update mlops-training-api \
  --region=europe-west1 \
  --set-env-vars RESTART=$(date +%s)

# Vérifier les métriques
gcloud run services describe mlops-training-api \
  --region=europe-west1 \
  --format="value(status.url,status.latestReadyRevisionName)"

# Tester la connectivité
curl -I https://mlops-training-api-bxzifydblq-ew.a.run.app/health
```

---

## 📚 Ressources Supplémentaires

- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [CatBoost Documentation](https://catboost.ai/docs/)

---

## 🆕 Nouveautés

### Version 2.0 - Janvier 2025
- ✅ **Gestion centralisée des variables d'environnement**
- ✅ **Connexion MLflow en production résolue**
- ✅ **Validation automatique des variables requises**
- ✅ **Scripts de déploiement optimisés**
- ✅ **Debug intégré avec `DEBUG_ENV=true`**
- ✅ **Fallbacks robustes pour les variables manquantes**

### Migration depuis v1.0
Les appels API restent identiques, mais :
- Les URLs de production ont changé
- Les variables d'environnement sont maintenant auto-validées
- MLflow fonctionne maintenant correctement en production
- Les timeouts sont mieux gérés

---

*Dernière mise à jour : 15 juillet 2025*
