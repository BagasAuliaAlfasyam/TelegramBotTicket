# Microservices Architecture — TelegramBotMyTech v2.0

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    Nginx Gateway (:80)                         │
│   /api/predict → Prediction API                               │
│   /api/data/   → Data API                                     │
│   /api/training → Training API                                │
│   /mlflow/     → MLflow UI                                    │
└────────────┬────────────┬──────────────┬──────────────────────┘
             │            │              │
   ┌─────────▼─┐  ┌──────▼────┐  ┌──────▼──────┐
   │ Prediction │  │  Data API │  │  Training   │
   │ API :8001  │  │   :8002   │  │  API :8005  │
   │            │  │           │  │             │
   │ LightGBM   │  │ Sheets    │  │ Retrain     │
   │ + Gemini   │  │ S3/MinIO  │  │ Pipeline    │
   │ Cascade    │  │ Tracking  │  │             │
   └──────┬─────┘  └─────┬─────┘  └──────┬──────┘
          │               │               │
   ┌──────▼───────────────▼───────────────▼──────┐
   │              MLflow :5000 + MinIO :9000      │
   └─────────────────────────────────────────────┘
             │                    │
   ┌─────────▼─────┐   ┌────────▼────────┐
   │ Collector Bot  │   │   Admin Bot     │
   │ (Telegram)     │   │   (Telegram)    │
   │ calls APIs     │   │   calls APIs    │
   └────────────────┘   └─────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| **prediction-api** | 8001 | ML Prediction — LightGBM + Gemini cascade (Knowledge Distillation) |
| **data-api** | 8002 | Centralized data — Google Sheets CRUD, S3 upload, ML Tracking |
| **collector-bot** | — | Telegram bot: collects ops replies, calls Prediction + Data API |
| **admin-bot** | — | Telegram bot: admin commands, calls all APIs |
| **training-api** | 8005 | Retraining pipeline — fetches data from Data API, trains to MLflow |
| **mlflow** | 5000 | MLflow tracking server |
| **minio** | 9000/9001 | S3-compatible object storage |
| **nginx** | 80 | API gateway |

## Gemini Knowledge Distillation (Cascade Pattern)

```
Input Text
    │
    ▼
┌──────────────┐
│  LightGBM    │  ← Fast, free, 15ms
│  Predict     │
└──────┬───────┘
       │
       ├── confidence ≥ 80% → Use LightGBM result (FREE)
       │
       └── confidence < 80%
              │
              ▼
       ┌──────────────┐
       │  Gemini API   │  ← Slower, paid, ~500ms
       │  Classify     │
       └──────┬────────┘
              │
              ├── Both agree → HYBRID (boosted confidence)
              ├── Gemini wins → Use Gemini label
              └── Disagree → Keep LightGBM
              
Over time: Gemini-labeled data → retrain LightGBM → fewer Gemini calls
```

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.microservices.template .env.local
# Edit .env.local with your credentials

# 2. Start all services
docker compose -f docker-compose.microservices.yml --env-file .env.local up -d

# 3. Check health
curl http://localhost:8001/health  # Prediction API
curl http://localhost:8002/health  # Data API
curl http://localhost:8005/health  # Training API
curl http://localhost:5000/health  # MLflow

# 4. Test prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"tech_raw_text": "tidak bisa login ke MIT", "solving": "Reset password"}'
```

## Development

```bash
# Run individual service locally
cd TelegramBotMyTech
pip install -r services/prediction/requirements.txt
python -m services.prediction.main

# Or with docker compose for specific service
docker compose -f docker-compose.microservices.yml up prediction-api data-api
```

## Migration from Monolith

The original monolith (`scripts/run_all.py`) is preserved. To switch:

| Monolith | Microservice |
|----------|-------------|
| `python scripts/run_all.py` | `docker compose -f docker-compose.microservices.yml up` |
| Direct Sheets access | Data API (`http://data-api:8002`) |
| Direct MLClassifier | Prediction API (`http://prediction-api:8001`) |
| `gcloud run jobs execute` | Training API (`http://training-api:8005/train`) |
| TF-IDF only | LightGBM + Gemini cascade |

## File Structure

```
services/
├── shared/                 # Shared library (all services import this)
│   ├── __init__.py
│   ├── config.py           # Service-specific configs from env vars
│   ├── models.py           # Pydantic models for API contracts
│   └── preprocessing.py    # Canonical ITSupportTextPreprocessor
├── prediction/             # ML Prediction API (FastAPI)
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── classifier.py       # LightGBM classifier
│       ├── gemini_classifier.py # Gemini zero-shot classifier
│       ├── hybrid.py           # Cascade/hybrid logic
│       └── mlflow_utils.py     # MLflow model loading
├── data/                   # Data API (FastAPI)
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── sheets.py           # Google Sheets CRUD
│       ├── tracking.py         # ML Tracking sheet
│       └── storage.py          # S3/MinIO upload
├── collector/              # Collecting Bot (python-telegram-bot)
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── bot.py              # OpsCollector (calls APIs via HTTP)
│       ├── parsers.py          # Message parsing
│       └── sla.py              # SLA calculation
├── admin/                  # Admin Bot (python-telegram-bot)
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       └── bot.py              # AdminCommandHandler (calls APIs via HTTP)
├── training/               # Training Pipeline (FastAPI)
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       └── retrain.py          # Retraining pipeline
└── gateway/
    └── nginx.conf          # API gateway configuration
```
