# 🚀 Quick Reference - Production API

## Production URLs
```bash
# Training API
https://mlops-training-api-bxzifydblq-ew.a.run.app

# MLflow UI
https://mlops-mlflow-bxzifydblq-ew.a.run.app

# Mock API
https://mlops-mock-api-bxzifydblq-ew.a.run.app
```

## Essential Commands

### Health Check
```bash
curl https://mlops-training-api-bxzifydblq-ew.a.run.app/ping
```

> 📸 **Screenshot needed**: Terminal output showing health check response

### Quick Training
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "20250715_195232", "learning_rate": 0.1, "epochs": 10}' \
  --max-time 600
```

> 📸 **Screenshot needed**: Training output with metrics

### Production Preprocessing
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

> 📸 **Screenshot needed**: Preprocessing completion message

### Production Prediction
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

> 📸 **Screenshot needed**: Prediction results summary

## Deployment

### Deploy Training API Only
```bash
cd deployment/
./deploy_training_only.sh
```

### View Logs
```bash
gcloud run services logs read mlops-training-api --region=europe-west1
```

## Key Changes v2.0

✅ **MLflow Connection Fixed**: Now uses Cloud Run URL instead of sqlite://  
✅ **Environment Variables**: Centralized configuration system  
✅ **Production Ready**: All endpoints tested and working  
✅ **Comprehensive Documentation**: Local + Production examples  
✅ **Debug Mode**: Enable with `DEBUG_ENV=true`  

## For Complete Documentation

See: `docs/api-production-guide.md`
