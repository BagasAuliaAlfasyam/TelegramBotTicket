#!/bin/bash
# MLflow Server Startup Script with Basic Authentication

set -e

echo "Starting MLflow Server with Basic Auth..."

# Create auth config with custom admin password
cat > /mlflow/basic_auth.ini << EOF
[mlflow]
default_permission = READ
database_uri = sqlite:///db/basic_auth.db
admin_username = ${MLFLOW_ADMIN_USER:-mytechops}
admin_password = ${MLFLOW_ADMIN_PASSWORD:-telkomjuara}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF

export MLFLOW_AUTH_CONFIG_PATH=/mlflow/basic_auth.ini

echo "Auth config created with user: ${MLFLOW_ADMIN_USER:-mytechops}"

# Start MLflow server with basic-auth
exec mlflow server \
    --app-name basic-auth \
    --backend-store-uri sqlite:///db/mlruns.db \
    --default-artifact-root s3://${MLFLOW_BUCKET_NAME}/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1
