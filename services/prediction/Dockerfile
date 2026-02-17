FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared library first (for caching)
COPY services/shared/ /app/services/shared/

# Copy service requirements and install
COPY services/prediction/requirements.txt /app/services/prediction/
RUN pip install --no-cache-dir -r /app/services/prediction/requirements.txt

# Copy service code
COPY services/prediction/ /app/services/prediction/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "services.prediction.main"]
