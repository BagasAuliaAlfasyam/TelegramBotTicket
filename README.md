# TelegramBotMyTech

Automated IT Support Ticket Classification System for MyTech — built with LightGBM + Gemini Knowledge Distillation cascade, deployed as containerized services on GCP.

## Architecture

```text
┌────────────────────────────────────────────────────────────────┐
│                     Nginx Gateway (:80)                        │
│   /api/predict  → Prediction API                               │
│   /api/data/    → Data API                                     │
│   /api/training → Training API                                 │
│   /mlflow/      → MLflow UI                                    │
└──────────┬──────────────┬──────────────┬───────────────────────┘
           │              │              │
  ┌────────▼────┐  ┌──────▼────┐  ┌──────▼──────┐
  │ Prediction  │  │  Data API │  │  Training   │
  │ API :8001   │  │   :8002   │  │  API :8005  │
  │             │  │           │  │             │
  │ LightGBM +  │  │ Sheets    │  │ Retrain     │
  │ Gemini      │  │ S3/MinIO  │  │ Pipeline    │
  │ Cascade     │  │ Tracking  │  │             │
  └──────┬──────┘  └─────┬─────┘  └──────┬──────┘
         │               │               │
  ┌──────▼───────────────▼───────────────▼───────┐
  │             MLflow :5000 + MinIO :9000        │
  └──────────────────────────────────────────────┘
           │                    │
  ┌────────▼──────┐   ┌────────▼────────┐
  │ Collector Bot │   │   Admin Bot     │
  │ (Telegram)    │   │   (Telegram)    │
  └───────────────┘   └─────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| **prediction-api** | 8001 | ML Prediction — LightGBM + Gemini cascade |
| **data-api** | 8002 | Google Sheets CRUD, S3 upload, ML Tracking |
| **training-api** | 8005 | Retraining pipeline with MLflow tracking |
| **collector-bot** | — | Telegram bot: collects ops replies, auto-classifies |
| **admin-bot** | — | Telegram bot: admin commands, reports, monitoring |
| **mlflow** | 5000 | MLflow Tracking Server |
| **minio** | 9000 | S3-compatible object storage |

## Project Structure

```
TelegramBotMyTech/
├── services/
│   ├── prediction/              # ML prediction (LightGBM + Gemini cascade)
│   ├── data/                    # Google Sheets, S3, ML tracking
│   ├── training/                # Retrain pipeline
│   ├── collector/               # Telegram collector bot
│   ├── admin/                   # Telegram admin bot
│   └── shared/                  # Shared models & config
├── mlflow/                      # MLflow server config
├── scripts/                     # Deploy & test scripts
├── docker-compose.yml           # Base compose (dev, with build:)
├── docker-compose.override.yml  # Prod override (prebuilt images)
├── cloudbuild.yaml              # CI/CD pipeline
├── pyproject.toml               # Ruff linter config
├── .env.template                # Environment template
└── README.md
```

## Quick Start

### 1. Configure Environment

```bash
cp .env.template .env
# Edit .env with your values
```

### 2. Run Locally (Dev)

```bash
docker compose up -d --build
```

### 3. Run with Pre-built Images (Prod)

```bash
export IMAGE_REGISTRY=asia-southeast2-docker.pkg.dev/mytech-480618/microservices/
export IMAGE_TAG=latest
docker compose -f docker-compose.yml -f docker-compose.override.yml \
  up -d --no-build
```

## ML Classification

### 2-Tier Prediction System

The system uses LightGBM with Gemini Knowledge Distillation cascade:

| Status | Confidence | Action |
|--------|------------|--------|
| **AUTO** | ≥ 75% | Auto-applied (LightGBM + Gemini cascade) |
| **REVIEW** | < 75% | Requires manual review |

### Flow

1. Message received → LightGBM predicts symptom + confidence
2. If confidence < 75% and Gemini enabled → Gemini cascade validates
3. If Gemini agrees with LightGBM → boost confidence, use prediction
4. If Gemini disagrees → use Gemini's prediction with adjusted confidence
5. Final status: AUTO (≥ 75%) or REVIEW (< 75%)

## Admin Bot Commands

| Command | Description |
|---------|-------------|
| `/stats` | Today's prediction statistics |
| `/report` | Performance report (weekly/monthly) |
| `/tiketreport` | Ticket report with SLA analysis |
| `/modelstatus` | Current model info & thresholds |
| `/mlflowstatus` | MLflow model versions |
| `/mlflowpromote` | Promote model version to Production |
| `/pendingreview` | Items pending manual review |
| `/retrainstatus` | Training readiness check |
| `/retrain` | Trigger model retraining |
| `/reloadmodel` | Hot-reload model (no restart) |
| `/updatestats` | Refresh monitoring statistics |
| `/trendbulan` | Monthly trend analysis |
| `/trendmingguan` | Weekly trend analysis |
| `/helpml` | Show all commands |

## Google Sheets

| Sheet | Purpose |
|-------|---------|
| **Logs** (20 cols) | All ticket data — message, classification, SLA |
| **ML_Tracking** (8 cols) | Prediction audit trail for review & retrain |
| **Monitoring** (9 cols) | Daily aggregated statistics |

## CI/CD

Cloud Build pipeline triggered by GitHub push:

```text
GitHub Push → Cloud Build → Lint → Validate → Build → Push → Deploy → Health Check
```

| Branch | Action |
|--------|--------|
| `main` | Full CI/CD: lint + build + deploy to VM |
| `develop` | CI only: lint + build (no deploy) |

### Manual Deploy (on VM)

```bash
./scripts/deploy.sh              # Pull from Artifact Registry
./scripts/deploy.sh --local      # Build locally
./scripts/deploy.sh --tag abc123 # Deploy specific tag
```

### Rollback

```bash
# On VM — deploy previous version
export IMAGE_TAG=<previous-commit-sha>
export IMAGE_REGISTRY=asia-southeast2-docker.pkg.dev/mytech-480618/microservices/
docker compose -f docker-compose.yml -f docker-compose.override.yml pull
docker compose -f docker-compose.yml -f docker-compose.override.yml \
  up -d --no-build
```

## CI/CD Setup (One-time)

### Prerequisites

```bash
# Enable GCP APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  --project=mytech-480618

# Configure Artifact Registry auth on VM
gcloud auth configure-docker asia-southeast2-docker.pkg.dev --quiet
```

### Cloud Build Triggers

Create two triggers in Cloud Console → Cloud Build → Triggers:

1. **CI (develop)**: Branch `^develop$`, substitution `_DEPLOY_ENABLED=false`
2. **CI/CD (main)**: Branch `^main$`, substitution `_DEPLOY_ENABLED=true`

### Troubleshooting

```bash
# Check ruff locally
ruff check services/

# Test Docker build locally
docker build -f services/prediction/Dockerfile -t test .

# Check service logs on VM
docker compose logs prediction-api --tail=50

# Restart a service
docker compose restart prediction-api
```

## License

MIT
