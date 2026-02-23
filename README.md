# TelegramBotMyTech

Automated IT Support Ticket Classification System — built with **LightGBM + Gemini 2.0 Flash Knowledge Distillation cascade**, deployed as containerized microservices on GCP Compute Engine.

Every message sent by the ops team to the technical group is captured, classified, recorded to Google Sheets, and acknowledged — automatically. Admins receive hourly performance reports and real-time alerts when model quality degrades.

---

## Architecture

```text
                        Telegram API
                            │
             ┌──────────────┴──────────────┐
             │                             │
    ┌────────▼────────┐          ┌─────────▼────────┐
    │  Collector Bot  │          │    Admin Bot      │
    │  (ops group)    │          │  (admin/reporting)│
    └────────┬────────┘          └─────────┬────────┘
             │ HTTP                        │ HTTP
     ┌───────▼──────────────────────────────▼──────┐
     │              Internal Docker Network         │
     │                 microservices                │
     │                                              │
     │  ┌─────────────┐   ┌───────────────────┐    │
     │  │ prediction- │   │     data-api      │    │
     │  │    api      │   │      :8002        │    │
     │  │   :8001     │   │                   │    │
     │  │             │   │  Google Sheets    │    │
     │  │ LightGBM +  │   │  MinIO S3         │    │
     │  │ Gemini 2.0  │   │  ML Tracking      │    │
     │  │  Cascade    │   └─────────┬─────────┘    │
     │  └──────┬──────┘             │              │
     │         │              ┌─────▼──────┐       │
     │         └──────────────►  training- │       │
     │                        │    api     │       │
     │                        │   :8005    │       │
     │                        └─────┬──────┘       │
     │                              │              │
     │   ┌──────────────────────────▼───────────┐  │
     │   │       MLflow :5000   MinIO :9000      │  │
     │   │  (Nginx + Basic Auth)  (S3 storage)  │  │
     │   └──────────────────────────────────────┘  │
     │                                              │
     │   ┌─────────────────────────────────────┐   │
     │   │  Prometheus :9090  Loki :3100        │   │
     │   │  Promtail          Grafana :3000     │   │
     │   └─────────────────────────────────────┘   │
     └──────────────────────────────────────────────┘
```

---

## Services

| Container | Port(s) | Role |
|-----------|---------|------|
| `prediction-api` | 8001 | ML inference — LightGBM + Gemini 2.0 Flash cascade |
| `data-api` | 8002 | Google Sheets CRUD, S3 uploads, ML Tracking |
| `training-api` | 8005 | Retraining pipeline with MLflow + Optuna |
| `collector-bot` | — | Telegram bot: captures ops replies, triggers classification |
| `admin-bot` | — | Telegram bot: admin commands, hourly auto-report, alerts |
| `mlflow` | 5000 | MLflow Tracking Server (Nginx + Basic Auth) |
| `minio` | 9000 / 9001 | S3-compatible object storage + web console |
| `minio-init` | — | One-shot: auto-creates MLflow + media buckets on startup |
| `prometheus` | 9090 | Metrics scraping (FastAPI `/metrics` endpoints) |
| `loki` | 3100 | Log aggregation |
| `promtail` | — | Docker log shipper → Loki |
| `grafana` | 3000 | Dashboards (Prometheus + Loki datasources) |

---

## Project Structure

```
TelegramBotMyTech/
├── services/
│   ├── prediction/          # ML prediction (LightGBM + Gemini cascade)
│   ├── data/                # Google Sheets, S3, ML tracking
│   ├── training/            # Retrain pipeline (LightGBM + Optuna + MLflow)
│   ├── collector/           # Telegram collector bot
│   ├── admin/               # Telegram admin bot
│   └── shared/              # Shared Pydantic models & config dataclasses
├── mlflow/                  # MLflow server (Nginx reverse proxy + htpasswd)
├── monitoring/              # Prometheus, Loki, Promtail, Grafana configs
├── scripts/                 # deploy.sh, stress tests
├── docker-compose.yml       # Full stack compose (build + run)
├── docker-compose.override.yml  # Prod override (prebuilt images from registry)
├── cloudbuild.yaml          # GCP Cloud Build CI/CD pipeline
├── pyproject.toml           # Ruff linter config
├── .env.template            # Environment variable template
└── service_account.json     # GCP Service Account key (not in git, mounted at runtime)
```

---

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose v2
- Google Service Account JSON with **Google Sheets API** and **Google Drive API** enabled
- Two Telegram bots (collecting + reporting) with their tokens

### 2. Configure Environment

```bash
cp .env.template .env
# Fill in all required values — see Environment Variables section below
```

> `service_account.json` must be placed in the project root. It is mounted read-only into `data-api` and `training-api` at `/app/service_account.json`.

### 3. Run Locally (Dev — local build)

```bash
docker compose up -d --build
```

### 4. Run in Production (prebuilt images from Artifact Registry)

```bash
export IMAGE_REGISTRY=asia-southeast2-docker.pkg.dev/mytech-480618/microservices/
export IMAGE_TAG=latest
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --no-build
```

---

## Environment Variables

All variables are documented in [.env.template](.env.template). Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN_COLLECTING` | ✅ | Token for the collector bot (monitors ops group) |
| `TELEGRAM_BOT_TOKEN_REPORTING` | ✅ | Token for the admin/reporting bot |
| `TARGET_GROUP_COLLECTING` | ✅ | Telegram chat ID of the ops group |
| `TARGET_GROUP_REPORTING` | ✅ | Telegram chat ID for hourly auto-reports |
| `ADMIN_USER_IDS` | ✅ | Comma-separated Telegram user IDs for admin access |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | ✅ | Path to GCP SA key (default: `service_account.json`) |
| `GOOGLE_SPREADSHEET_NAME` | ✅ | Name of the Google Spreadsheet |
| `GEMINI_API_KEY` | ✅ | Google Gemini API key for KD cascade |
| `GEMINI_ENABLED` | — | Enable Gemini cascade (default: `true`) |
| `ML_THRESHOLD_AUTO` | — | Min confidence for AUTO status (default: `0.75`) |
| `MLFLOW_TRACKING_URI` | — | MLflow server URI (default: `http://mlflow:5000`) |
| `MLFLOW_TRACKING_USERNAME` | ✅ | MLflow Basic Auth username |
| `MLFLOW_TRACKING_PASSWORD` | ✅ | MLflow Basic Auth password |
| `AWS_ACCESS_KEY_ID` | ✅ | MinIO root user / S3 access key |
| `AWS_SECRET_ACCESS_KEY` | ✅ | MinIO root password / S3 secret key |
| `AWS_S3_BUCKET` | ✅ | MinIO bucket for Telegram media uploads |

---

## ML Classification

### 2-Tier Prediction System

| Status | Confidence | Meaning |
|--------|------------|---------|
| `AUTO` | ≥ 75% | Prediction applied automatically |
| `REVIEW` | < 75% | Queued for manual review by admin |

### Cascade Flow

```
Ops message arrives
        │
        ▼
LightGBM predicts symptom + confidence
        │
        ├─ confidence ≥ 0.75 ──── Status: AUTO  (source: lightgbm)
        │
        └─ confidence < 0.75 ──── Gemini 2.0 Flash cascade triggered
                    │
                    ├─ Gemini agrees with LightGBM ──── confidence boosted
                    │                                        │
                    │                              ≥ 0.75 → AUTO  (source: hybrid)
                    │                              < 0.75 → REVIEW (source: hybrid)
                    │
                    └─ Gemini disagrees ──── Gemini label used
                                                 │
                                       ≥ 0.75 → AUTO  (source: gemini)
                                       < 0.75 → REVIEW (source: gemini)
```

Prediction source (`lightgbm`, `gemini`, `hybrid`) is recorded in `ML_Tracking` for audit and retraining.

---

## API Endpoints

### Prediction API (`:8001`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Classify a single ticket |
| `POST` | `/predict/batch` | Classify up to 100 tickets |
| `GET` | `/model/info` | Loaded model version, classes, thresholds |
| `POST` | `/model/reload` | Hot-reload model from MLflow (no container restart needed) |
| `GET` | `/mlflow/status` | List registered model versions in MLflow registry |
| `POST` | `/mlflow/promote` | Promote a model version to Production stage |
| `GET` | `/health` | Health check |

### Data API (`:8002`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/logs/append` | Append a new row to the Logs sheet |
| `PUT` | `/logs/{row_index}` | Update an existing row by sheet row index |
| `GET` | `/logs/find/{tech_message_id}` | Find row index by `tech_message_id` (searches column D only) |
| `GET` | `/logs/row/{row_index}` | Read a specific row by index |
| `GET` | `/logs/all` | Read all rows (headers + data) |
| `POST` | `/tracking/log` | Append prediction record to ML_Tracking sheet |
| `GET` | `/stats/realtime` | Real-time stats from ML_Tracking |
| `GET` | `/stats/weekly` | Weekly aggregated stats |
| `GET` | `/stats/monthly` | Monthly aggregated stats |
| `POST` | `/stats/hourly` | Calculate hourly stats and write to Monitoring sheet |
| `GET` | `/training/data` | Pull labeled data (Logs + ML_Tracking) for retraining |
| `POST` | `/training/mark` | Mark reviewed rows as `TRAINED` |
| `POST` | `/media/upload` | Upload media file to MinIO S3 |
| `GET` | `/health` | Health check |

### Training API (`:8005`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/train` | Trigger a training job (supports `tune`, `tune100`, `force` modes) |
| `GET` | `/status` | Current training job status |
| `GET` | `/health` | Health check |

---

## Google Sheets Structure

The spreadsheet (configured by `GOOGLE_SPREADSHEET_NAME`) must contain the following worksheets:

### `Logs` — 20 columns (A–T)

| Col | Field | Description |
|-----|-------|-------------|
| A | `group_label` | Group identifier |
| B | `ticket_date` | Ticket creation date |
| C | `response_at` | Ops response timestamp |
| D | `tech_message_id` | Tech group message ID — **primary lookup key** |
| E | `tech_message_date` | Tech message date |
| F | `tech_message_time` | Tech message time |
| G | `tech_raw_text` | Raw text from technician |
| H | `media_type` | Media type (`photo` / `video` / `none`) |
| I | `media_url` | MinIO S3 URL of uploaded media |
| J | `ops_message_id` | Ops reply message ID |
| K | `ops_text` | Ops reply text |
| L | `solving` | Extracted solver tag (e.g. `-bg`, `-dm`) |
| M | `solve_timestamp` | Solve timestamp |
| N | `app_code` | Application code |
| O | `solver_name` | Resolved solver name (e.g. `Bagas`, `Damas`) |
| P | `is_oncek` | Whether ticket was *oncek* (pre-check flag) |
| Q | `sla_response_min` | SLA response time in minutes |
| R | `sla_status` | `ON_TIME` or `LATE` |
| S | `sla_remaining_min` | Remaining SLA time in minutes |
| T | `symtomps` | Predicted symptom label |

### `ML_Tracking` — 7 columns

Prediction audit trail. One row per new classified ticket, written by `POST /tracking/log`.

Fields: `tech_message_id`, `tech_raw_text`, `solving`, `predicted_symtomps`, `ml_confidence`, `prediction_status`, `source`.

> Edited messages do **not** write a new ML_Tracking row — only new message events are tracked, preventing dirty training data.

### `Monitoring` — Hourly aggregation

Written by `POST /stats/hourly`, called automatically every hour by `TrendAlertService` in `admin-bot`.

---

## Collector Bot

Listens to the designated ops Telegram group. For each ops reply:

1. Identifies the referenced tech message (`reply_to_message`)
2. Calls `POST /predict` → LightGBM + Gemini cascade
3. Calls `GET /logs/find/{tech_message_id}` → resolves existing sheet row (column D lookup)
4. Calls `PUT /logs/{row_index}` → updates solver, SLA, and symptom label
5. Calls `POST /tracking/log` → records prediction in ML_Tracking (new messages only)
6. Sends notification to the ops group

**New message** → notification: `"Laporan dicatat ✅"`  
**Edited message** → sheet row updated again, notification: `"Laporan diperbarui ✏️"`, no new ML_Tracking row

Bot state (row index cache) is persisted to `/app/state/state_cache.json` on Docker named volume `collector-state`, so the cache survives container restarts and redeployments.

---

## Admin Bot

### Commands

| Command | Description |
|---------|-------------|
| `/stats` | Real-time ML dashboard (predictions, confidence, auto/review split) |
| `/report` | Weekly/monthly performance report |
| `/tiketreport` | Ticket report with SLA analysis |
| `/modelstatus` | Current model version, Gemini status, configured thresholds |
| `/mlflowstatus` | Registered model versions in MLflow registry |
| `/mlflowpromote` | Promote a model version to Production stage |
| `/pendingreview` | List tickets pending manual review |
| `/retrainstatus` | Check data readiness for retraining |
| `/retrain` | Trigger model retraining |
| `/reloadmodel` | Hot-reload latest Production model (no container restart) |
| `/trendbulan` | Top symptoms by application — monthly breakdown |
| `/trendmingguan` | Top symptoms by application — weekly breakdown |
| `/help` | Show all available commands |

### `/retrain` Modes

```
/retrain             Default — train with current hyperparameters
/retrain tune        + Optuna hyperparameter search (20 trials)
/retrain tune100     + Optuna hyperparameter search (100 trials)
/retrain force       Force train even if labeled data < 50 rows
```

---

## Hourly Auto-Report (TrendAlertService)

`admin-bot` runs a background `TrendAlertService` loop every **3600 seconds**. On each tick:

1. `GET /model/info` → fetch current model version
2. `POST /stats/hourly` → calculate hourly stats and write to Monitoring sheet
3. Send formatted hourly report to `TARGET_GROUP_REPORTING`
4. Evaluate alert thresholds — send alert DMs to all `ADMIN_USER_IDS` if any threshold is breached

### Alert Thresholds

| Metric | Threshold | Alert |
|--------|-----------|-------|
| Automation rate | < 70% | ⚠️ Low auto-classification rate — consider retraining |
| Review rate | > 30% | 🔴 High manual review queue |
| Avg confidence | < 80% | ⚠️ Model confidence dropping — monitor closely |
| Pending queue | > 50 items | 📋 Large unreviewed backlog |

> There is no `/updatestats` command. Hourly stats are written to the Monitoring sheet and broadcast to the reporting group automatically.

---

## Observability (PLG Stack)

| Tool | Port | Purpose |
|------|------|---------|
| **Prometheus** | 9090 | Scrapes `/metrics` from `prediction-api` and `data-api` |
| **Loki** | 3100 | Receives structured JSON logs from all containers via Promtail |
| **Promtail** | — | Tails Docker container logs and ships them to Loki |
| **Grafana** | 3000 | Dashboards — Prometheus (metrics) + Loki (logs) datasources pre-provisioned |

All FastAPI services expose `/metrics` automatically via `prometheus-fastapi-instrumentator`.

---

## CI/CD

Cloud Build pipeline triggered by GitHub push:

```
GitHub Push → Cloud Build → Lint (ruff) → Build Images → Push to Artifact Registry → SSH Deploy to VM → Health Check
```

| Branch | Action |
|--------|--------|
| `main` | Full CI/CD: lint + build + push + deploy |
| `develop` | CI only: lint + build (no push, no deploy) |

### Manual Deploy (on VM)

```bash
./scripts/deploy.sh              # Pull latest from Artifact Registry and deploy
./scripts/deploy.sh --local      # Build locally (no registry pull)
./scripts/deploy.sh --tag abc123 # Deploy a specific image tag
```

### Rollback

```bash
export IMAGE_TAG=<previous-commit-sha>
export IMAGE_REGISTRY=asia-southeast2-docker.pkg.dev/mytech-480618/microservices/
docker compose -f docker-compose.yml -f docker-compose.override.yml pull
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --no-build
```

---

## One-time Setup

### Enable GCP APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  drive.googleapis.com \
  sheets.googleapis.com \
  --project=mytech-480618
```

### Configure Artifact Registry Auth on VM

```bash
gcloud auth configure-docker asia-southeast2-docker.pkg.dev --quiet
```

### Cloud Build Triggers

Create two triggers in Cloud Console → Cloud Build → Triggers:

1. **CI (develop)**: Branch `^develop$`, substitution `_DEPLOY_ENABLED=false`
2. **CI/CD (main)**: Branch `^main$`, substitution `_DEPLOY_ENABLED=true`

---

## Development

```bash
# Lint before pushing
ruff check services/

# Test a single Docker build locally
docker build -f services/prediction/Dockerfile -t test-prediction .

# Tail logs for a service
docker compose logs -f collector-bot

# Check all container health statuses
docker ps --format 'table {{.Names}}\t{{.Status}}'

# Restart a single service
docker compose restart prediction-api
```
