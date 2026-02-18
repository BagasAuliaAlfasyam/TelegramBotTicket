# MLflow Tracking Server with Nginx Basic Auth
# Deployed to Cloud Run with MinIO as artifact storage

FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    apache2-utils \
    curl \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    mlflow==2.10.0 \
    boto3>=1.34.0 \
    psycopg2-binary>=2.9.0

# Create directories
RUN mkdir -p /mlflow/artifacts /mlflow/db /var/log/nginx

# Copy configuration files
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /entrypoint.sh

# Generate htpasswd file with proper permissions
# Username: mytechops, Password: telkomjuara
RUN htpasswd -cb /etc/nginx/.htpasswd mytechops telkomjuara && \
    chmod 644 /etc/nginx/.htpasswd

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# Expose ports:
#   8080 — nginx reverse proxy with basic auth (external/public access)
#   5000 — raw MLflow server (internal Docker network, no auth)
EXPOSE 8080 5000

# Health check (direct to MLflow, bypasses nginx auth)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start services
ENTRYPOINT ["/entrypoint.sh"]
