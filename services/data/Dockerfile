FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/shared/ /app/services/shared/
COPY services/data/requirements.txt /app/services/data/
RUN pip install --no-cache-dir -r /app/services/data/requirements.txt

COPY services/data/ /app/services/data/

# Service account JSON is mounted via volume in docker-compose
# COPY service_account.json /app/service_account.json

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "-m", "services.data.main"]
