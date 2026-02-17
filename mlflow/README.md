# MLflow Tracking Server

MLflow server dengan Basic Auth, deployed ke Cloud Run dengan MinIO sebagai artifact storage.

## Architecture

```
Cloud Run (8080)
    │
    ▼
Nginx (Basic Auth)
    │
    ▼
MLflow Server (5000)
    │
    ├── Backend Store: SQLite (synced to MinIO)
    └── Artifact Store: MinIO (s3://mlflow-artifacts/)
```

## Prerequisites

1. **MinIO Bucket**: Buat bucket `mlflow-artifacts` di MinIO
2. **Secret Manager**: Setup secrets di GCP Secret Manager
3. **Basic Auth**: Generate `.htpasswd` file

## Setup

### 1. Buat MinIO Bucket

```bash
# Menggunakan MinIO Client (mc)
mc mb myminio/mlflow-artifacts
```

### 2. Generate .htpasswd

```bash
# Install apache2-utils jika belum ada
# Linux/Mac:
htpasswd -c mlflow/.htpasswd <username>

# Windows (menggunakan Docker):
docker run --rm httpd:alpine htpasswd -nb <username> <password> > mlflow/.htpasswd
```

### 3. Setup GCP Secrets

```bash
# Create secrets
echo -n "https://your-minio-endpoint.com" | gcloud secrets create mlflow-s3-endpoint --data-file=-
echo -n "your-access-key" | gcloud secrets create mlflow-s3-access-key --data-file=-
echo -n "your-secret-key" | gcloud secrets create mlflow-s3-secret-key --data-file=-
echo -n "mlflow-artifacts" | gcloud secrets create mlflow-bucket-name --data-file=-
```

### 4. Deploy ke Cloud Run

```bash
# Manual deploy
gcloud builds submit --config=mlflow/cloudbuild.yaml

# Atau manual Docker build + deploy
docker build -t gcr.io/PROJECT_ID/mlflow-server ./mlflow
docker push gcr.io/PROJECT_ID/mlflow-server
gcloud run deploy mlflow-server --image gcr.io/PROJECT_ID/mlflow-server --region asia-southeast1
```

## Local Development

```bash
# Set environment variables
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_BUCKET_NAME=mlflow-artifacts

# Run with Docker
docker build -t mlflow-server ./mlflow
docker run -p 8080:8080 \
    -e MLFLOW_S3_ENDPOINT_URL \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e MLFLOW_BUCKET_NAME \
    mlflow-server

# Access UI: http://localhost:8080 (with basic auth)
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MLFLOW_S3_ENDPOINT_URL` | MinIO endpoint URL | Yes |
| `AWS_ACCESS_KEY_ID` | MinIO access key | Yes |
| `AWS_SECRET_ACCESS_KEY` | MinIO secret key | Yes |
| `MLFLOW_BUCKET_NAME` | Bucket name for artifacts | Yes |
| `MLFLOW_PORT` | MLflow server port (default: 5000) | No |
| `MLFLOW_WORKERS` | Gunicorn workers (default: 2) | No |

## Usage

### Log Experiments from Bot

```python
import mlflow

mlflow.set_tracking_uri("https://mlflow-server-xxx.run.app")
mlflow.set_experiment("ticket-classifier")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 300)
    mlflow.log_metric("f1_score", 0.92)
    mlflow.lightgbm.log_model(model, "model")
```

### Load Model from Registry

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/ticket-classifier/Production")
predictions = model.predict(data)
```
