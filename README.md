# 🥷 Automatic Fraud Detection — MLOps Project

Ce projet vise à détecter les fraudes sur des transactions de carte bancaire via un pipeline **modulaire et déployable en production**. Il inclut preprocessing, entraînement, prédiction, monitoring de dérive, orchestration Airflow et intégration GCP.

---

## 📁 Structure du projet

```
model_training/
├── src/
│   ├── preprocessing.py         # Feature engineering + encodage
│   ├── train.py                 # Entraînement CatBoost + MLflow
│   ├── check_preproc_data.py   # Sanity check après preprocessing
│   └── ...                     # Autres modules (à venir: predict.py, validate.py)
├── model_training_api/         # FastAPI endpoints (model-api)
├── mock_realtime_api/          # Générateur de données factices (mock-api)
├── docker-compose.yml          # Local Dev avec 3 services
└── deployment/                 # Scripts de déploiement Cloud Run
```

---

## 🚀 Fonctionnalités

* Prétraitement avec features temporelles, distance haversine, log(amount), encodage target
* Modèle CatBoost avec suivi MLflow
* API FastAPI unifiée pour prédiction, validation, drift, etc.
* Déploiement Cloud Run + GCS + Secret Manager
* Monitoring via data drift et logs GCP

---

## 🛠️ Démarrer

### En local

```bash
docker compose up --build
```

Accès rapides :

* API: [http://localhost:8000/docs](http://localhost:8000/docs)
* MLflow: [http://localhost:5000](http://localhost:5000)

### En production

Déployer via `deployment/deploy_all_services.sh`
Prévoir secrets GCP et `.env` configuré.

---

## 📦 Pipeline

1. Prétraitement : `/preprocess`
2. Entraînement : `/train`
3. Validation : `/validate`
4. Prédiction : `/predict`
5. Monitoring dérive : `/monitor`
6. Données mock : `/transactions` (mock-api)

---

## 📚 Documentation

* [EDA](docs/eda.md)
* [Preprocessing](docs/preprocessing.md)
* [Training](docs/train.md)
* \[API Guide]\(docs/api.md ou ce README)

---

## 🔧 Configuration

Voir `.env` local + Secrets Manager côté GCP
Exemples dans [`gcp_commands.md`](docs/gcp_commands.md)