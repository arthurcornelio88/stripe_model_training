FROM python:3.10-slim

# Install MLflow and required extras
RUN pip install --no-cache-dir mlflow==2.11.3

# Create folders for artifacts
RUN mkdir -p /mlflow/mlruns

# Expose port
EXPOSE 5000

# Set working dir and command
WORKDIR /mlflow
CMD ["mlflow", "server", "--backend-store-uri", "/mlflow/mlruns", "--default-artifact-root", "/mlflow/mlruns", "--host", "0.0.0.0", "--port", "5000", "--serve-artifacts"]
