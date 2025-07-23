# 📚 API Documentation

This directory contains comprehensive documentation for the MLOps Fraud Detection API.

## 📋 Documentation Structure

### 🚀 Main Guides
- **[`api-production-guide.md`](api-production-guide.md)** - Complete API guide with local and production examples

## 🎯 For Developers

### Quick Start
1. **Health Check**: `curl https://mlops-training-api-bxzifydblq-ew.a.run.app/ping`
2. **Training**: See [`quick-reference.md`](quick-reference.md) for essential commands
3. **Complete Workflow**: Follow [`api-production-guide.md`](api-production-guide.md)

### Documentation Guidelines
- All endpoints include both local and production examples
- Screenshots are marked with `📸 **Screenshot needed**:` comments
- Commands include proper timeouts for production use
- Error handling and troubleshooting sections included

## 🔧 API Endpoints

| Endpoint | Method | Purpose | Documentation |
|----------|---------|---------|---------------|
| `/ping` | GET | Health check | All guides |
| `/preprocess` | POST | Data preprocessing | All guides |
| `/train` | POST | Model training | All guides |
| `/validate` | POST | Model validation | All guides |
| `/predict` | POST | Batch prediction | All guides |
| `/monitor` | POST | Data drift detection | All guides |

## 🌐 Service URLs

### Production
- **Training API**: https://mlops-training-api-bxzifydblq-ew.a.run.app
- **MLflow UI**: https://mlops-mlflow-bxzifydblq-ew.a.run.app
- **Mock API**: https://mlops-mock-api-bxzifydblq-ew.a.run.app

### Local Development
- **Training API**: http://localhost:8000
- **MLflow UI**: http://localhost:5000

## 📸 Screenshots Status

To complete the documentation, screenshots are needed for:
- [ ] Health check response
- [ ] Preprocessing output
- [ ] Training progress and metrics
- [ ] Validation results
- [ ] Prediction completion
- [ ] Data drift monitoring
- [ ] Complete workflow execution
- [ ] Swagger UI interface
- [ ] MLflow UI interface

See [`screenshots-needed.md`](screenshots-needed.md) for detailed capture instructions.

## 🔄 Updates

### Latest Changes (v2.0)
- ✅ Translated to English
- ✅ Added screenshot placeholders
- ✅ Enhanced troubleshooting sections
- ✅ Updated production URLs
- ✅ Added comprehensive examples

### Migration Notes
- All French documentation has been translated to English
- API endpoints remain the same
- Production URLs are now fully integrated
- Screenshot locations are prepared

---

*Last updated: July 16, 2025*
