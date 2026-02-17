FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/shared/ /app/services/shared/
COPY services/admin/requirements.txt /app/services/admin/
RUN pip install --no-cache-dir -r /app/services/admin/requirements.txt

COPY services/admin/ /app/services/admin/

CMD ["python", "-m", "services.admin.main"]
