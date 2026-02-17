FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/shared/ /app/services/shared/
COPY services/collector/requirements.txt /app/services/collector/
RUN pip install --no-cache-dir -r /app/services/collector/requirements.txt

COPY services/collector/ /app/services/collector/

# Persistent state volume
VOLUME /app/state

CMD ["python", "-m", "services.collector.main"]
