# 📸 Screenshots Needed for API Documentation

This document lists all the screenshots that need to be captured to complete the API documentation.

## 🔹 Health Check Endpoint

**File**: `api-production-guide.md` - Section 1
**Command**: 
```bash
curl https://mlops-training-api-bxzifydblq-ew.a.run.app/ping
```
**Expected Output**: `{"status": "alive"}`
**Screenshot**: Terminal showing successful health check response

---

## 🔹 Preprocessing Endpoint

**File**: `api-production-guide.md` - Section 2
**Command**: 
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
**Expected Output**: Processing progress messages, completion status
**Screenshot**: Terminal showing preprocessing progress and completion

---

## 🔹 Training Endpoint

**File**: `api-production-guide.md` - Section 3
**Command**: 
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/train \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "20250715_195232",
    "learning_rate": 0.1,
    "epochs": 10,
    "test": false
  }' \
  --max-time 600
```
**Expected Output**: Training progress, final metrics (AUC, F1, precision, recall)
**Screenshot**: Terminal showing training progress and final metrics

---

## 🔹 Validation Endpoint

**File**: `api-production-guide.md` - Section 4
**Command**: 
```bash
curl -X POST https://mlops-training-api-bxzifydblq-ew.a.run.app/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "catboost_model_20250715_195232.cbm",
    "timestamp": "20250715_195232"
  }' \
  --max-time 300
```
**Expected Output**: Validation metrics and model performance
**Screenshot**: Terminal showing validation metrics and results

---

## 🔹 Prediction Endpoint

**File**: `api-production-guide.md` - Section 5
**Command**: 
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
**Expected Output**: Prediction process status and completion
**Screenshot**: Terminal showing prediction process and completion

---

## 🔹 Monitoring (Data Drift) Endpoint

**File**: `api-production-guide.md` - Section 6
**Command**: 
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
**Expected Output**: Drift detection analysis and results
**Screenshot**: Terminal showing drift detection analysis and results

---

## 🔹 Complete Workflow

**File**: `api-production-guide.md` - Usage Examples section
**Commands**: Full sequence from health check to drift monitoring
**Expected Output**: Complete workflow execution
**Screenshot**: Terminal showing complete workflow execution from health check to drift monitoring

---

## 🔹 Swagger Interface

**File**: `api-production-guide.md` - Monitoring section
**URL**: https://mlops-training-api-bxzifydblq-ew.a.run.app/docs
**Expected Output**: Interactive API documentation
**Screenshot**: Swagger UI showing all available endpoints

---

## 🔹 MLflow UI

**File**: `api-production-guide.md` - Monitoring section
**URL**: https://mlops-mlflow-bxzifydblq-ew.a.run.app
**Expected Output**: Experiment tracking interface
**Screenshot**: MLflow UI showing experiment tracking and model metrics

---

## 🔹 Quick Reference Examples

**File**: `quick-reference.md` - Multiple sections
**Commands**: All essential commands from quick reference
**Expected Output**: Various API responses
**Screenshots needed**:
- Health check response
- Training output with metrics
- Preprocessing completion message
- Prediction results summary

---

## 📋 Screenshot Capture Instructions

### For Terminal Screenshots:
1. Use a clean terminal with good contrast
2. Ensure full command and output are visible
3. Capture both the command and the complete response
4. Use a reasonable font size for readability
5. Include timestamp if relevant

### For Web Interface Screenshots:
1. Use full-screen or maximized browser window
2. Ensure all relevant information is visible
3. Use good lighting/contrast settings
4. Capture the full interface with navigation elements

### Recommended Tools:
- **Linux**: `gnome-screenshot`, `scrot`, or `flameshot`
- **macOS**: Built-in screenshot tools (Cmd+Shift+4)
- **Windows**: Snipping Tool or PowerToys Screen Ruler

### File Naming Convention:
- `screenshot_health_check.png`
- `screenshot_preprocessing.png`
- `screenshot_training.png`
- `screenshot_validation.png`
- `screenshot_prediction.png`
- `screenshot_monitoring.png`
- `screenshot_workflow_complete.png`
- `screenshot_swagger_ui.png`
- `screenshot_mlflow_ui.png`

---

## 📁 Storage Location

Save all screenshots in:
```
model_training/docs/screenshots/
```

And reference them in markdown files like:
```markdown
![Health Check Response](screenshots/screenshot_health_check.png)
```

---

*Last updated: July 16, 2025*
