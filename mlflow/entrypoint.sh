#!/bin/bash
set -e

echo "=== MLflow Server Startup ==="

# Validate required environment variables
required_vars=("MLFLOW_S3_ENDPOINT_URL" "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "MLFLOW_BUCKET_NAME")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# Set AWS region (required even for MinIO)
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}

# Set default values
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_HOST=${MLFLOW_HOST:-0.0.0.0}
MLFLOW_WORKERS=${MLFLOW_WORKERS:-1}

# Ensure directories exist (volume mount replaces Dockerfile dirs)
mkdir -p /mlflow/db

# Artifact root in MinIO
ARTIFACT_ROOT="s3://${MLFLOW_BUCKET_NAME}"

# Backend store (SQLite, compatible with previous stock image path)
BACKEND_STORE="/mlflow/mlflow.db"

echo "Configuration:"
echo "  - S3 Endpoint: ${MLFLOW_S3_ENDPOINT_URL}"
echo "  - Artifact Root: ${ARTIFACT_ROOT}"
echo "  - Backend Store: sqlite:///${BACKEND_STORE}"
echo "  - MLflow Port: ${MLFLOW_PORT}"
echo "  - Workers: ${MLFLOW_WORKERS}"

# Download existing SQLite DB from MinIO if exists
echo "Checking for existing database in MinIO..."
if aws s3 cp "s3://${MLFLOW_BUCKET_NAME}/mlflow.db" "${BACKEND_STORE}" \
    --endpoint-url "${MLFLOW_S3_ENDPOINT_URL}" 2>/dev/null; then
    echo "  - Downloaded existing database from MinIO"
else
    echo "  - No existing database found, starting fresh"
    touch "${BACKEND_STORE}"
fi

# Function to backup SQLite to MinIO
backup_db() {
    while true; do
        sleep 300  # Backup every 5 minutes
        echo "Backing up database to MinIO..."
        aws s3 cp "${BACKEND_STORE}" "s3://${MLFLOW_BUCKET_NAME}/mlflow.db" \
            --endpoint-url "${MLFLOW_S3_ENDPOINT_URL}" 2>/dev/null || true
    done
}

# Start background backup process
backup_db &
BACKUP_PID=$!

# Cleanup on exit
cleanup() {
    echo "Shutting down..."
    # Final backup before exit
    aws s3 cp "${BACKEND_STORE}" "s3://${MLFLOW_BUCKET_NAME}/mlflow.db" \
        --endpoint-url "${MLFLOW_S3_ENDPOINT_URL}" 2>/dev/null || true
    kill $BACKUP_PID 2>/dev/null || true
    nginx -s quit 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start Nginx in background
echo "Starting Nginx..."
nginx

# Test MinIO connectivity
echo "Testing MinIO connectivity..."
aws s3 ls "s3://${MLFLOW_BUCKET_NAME}/" --endpoint-url "${MLFLOW_S3_ENDPOINT_URL}" || echo "Warning: Could not list MinIO bucket"

# Start MLflow server with verbose output
echo "Starting MLflow server..."
echo "Command: mlflow server --backend-store-uri sqlite:///${BACKEND_STORE} --default-artifact-root ${ARTIFACT_ROOT} --host ${MLFLOW_HOST} --port ${MLFLOW_PORT} --workers ${MLFLOW_WORKERS}"

# Run MLflow without exec first to see any error
mlflow server \
    --backend-store-uri "sqlite:///${BACKEND_STORE}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --host "${MLFLOW_HOST}" \
    --port "${MLFLOW_PORT}" \
    --workers "${MLFLOW_WORKERS}" 2>&1

# If we reach here, something went wrong
echo "ERROR: MLflow server exited unexpectedly"
sleep 60
