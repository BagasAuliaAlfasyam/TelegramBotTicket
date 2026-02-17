FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/shared/ /app/services/shared/

COPY services/training/requirements.txt /app/services/training/
RUN pip install --no-cache-dir -r /app/services/training/requirements.txt

COPY services/training/ /app/services/training/

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

CMD ["python", "-m", "services.training.main"]
