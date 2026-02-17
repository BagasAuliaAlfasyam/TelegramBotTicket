# TelegramBotMyTech

Automated IT Support Ticket Classification Bot for MyTech team.

## Features

- 🤖 **ML Classification**: Automatic symptom classification using LightGBM
- 📊 **Google Sheets Integration**: Auto-logging tickets to spreadsheet
- 📁 **S3 Media Upload**: Store ticket attachments in S3
- 📈 **Admin Dashboard**: Telegram commands for monitoring
- 🔄 **Auto-retrain**: Script for model retraining with new data

## Project Structure

```
TelegramBotMyTech/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── core/                     # Core configuration
│   │   ├── __init__.py
│   │   └── config.py
│   ├── ml/                       # ML components
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # Domain-aware text preprocessing
│   │   ├── classifier.py         # ML model wrapper
│   │   └── tracking.py           # Audit trail & monitoring
│   ├── bots/                     # Telegram bot handlers
│   │   ├── __init__.py
│   │   ├── collector.py          # Ops reply collector
│   │   ├── admin.py              # Admin command handlers
│   │   ├── parsers.py            # Message parsing utils
│   │   └── sla.py                # SLA calculation
│   └── services/                 # External integrations
│       ├── __init__.py
│       ├── sheets.py             # Google Sheets client
│       └── storage.py            # S3 uploader
├── scripts/                      # Entry points
│   ├── run_all.py                # 🌟 Unified: Run both bots in one process
│   ├── run_collecting.py         # Start collecting bot only
│   └── run_reporting.py          # Start reporting bot only
├── models/                       # ML model artifacts (versioned)
│   ├── v1/
│   │   ├── lgb_model.bin
│   │   ├── tfidf_vectorizer.pkl
│   │   └── label_encoder.pkl
│   ├── current_version.txt      # Active version pointer
│   └── versions.json            # Version history
├── .env                          # Environment variables
├── .env.local                    # Local overrides (gitignored)
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
# Telegram Bot Tokens
TELEGRAM_BOT_TOKEN=your_collecting_bot_token
TELEGRAM_BOT_TOKEN_REPORTING=your_reporting_bot_token

# Chat IDs
OPS_CHAT_ID=123456789
TECH_CHAT_ID=123456789
ADMIN_CHAT_ID=123456789

# Google Sheets
GOOGLE_SERVICE_ACCOUNT_JSON=service_account.json
GOOGLE_SPREADSHEET_NAME=Log_Tiket_MyTech
GOOGLE_WORKSHEET_NAME=Logs

# AWS S3
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET_NAME=your_bucket
S3_REGION=ap-southeast-1

# ML Settings
MODEL_VERSION=auto  # auto = read from current_version.txt
ML_THRESHOLD_AUTO=0.90
ML_THRESHOLD_HIGH=0.85
ML_THRESHOLD_MEDIUM=0.70

# Optional
ADMIN_USER_IDS=123456789,987654321
TIMEZONE=Asia/Jakarta
DEBUG=false
```

### 3. Setup Google Sheets

1. Create a Google Cloud project
2. Enable Google Sheets API
3. Create a service account
4. Download JSON credentials as `service_account.json`
5. Share the spreadsheet with service account email

## Running

### 🌟 Recommended: Run Both Bots Together

```bash
python scripts/run_all.py
```

This runs **both Collecting and Reporting bots** in a single process.

### Run Individual Bots (for debugging)

Collecting Bot only:
```bash
python scripts/run_collecting.py
```

Reporting Bot only:
```bash
python scripts/run_reporting.py
```

### Retrain Model (Manual Trigger)

Training bisa dilakukan via:
- **Notebook** (untuk full Optuna tuning) - folder `Analyst/`
- **Script** (untuk quick retrain) - `scripts/retrain.py`
- **Telegram** (hot reload) - `/retrain` command via admin bot

Lihat section "Retraining Model" di bawah.

## CI/CD Pipeline

**Cloud Build** is used for all CI/CD automation:

- **Lint** — Python code quality checks with ruff
- **Validate** — Dockerfile & docker-compose syntax validation
- **Build** — Parallel Docker image builds for all 5 microservices
- **Deploy** — Automatic deployment to GCP VM (main branch only)
- **Health Check** — Verify all services are running after deploy

**For detailed setup instructions, see [CICD_SETUP.md](CICD_SETUP.md)**

## ML Admin Commands

| Command | Description |
|---------|-------------|
| `/stats` | Today's prediction statistics |
| `/report weekly` | Weekly performance report |
| `/report monthly` | Monthly performance report |
| `/modelstatus` | Current model information |
| `/pendingreview` | Items pending manual review |
| `/retrainstatus` | Check retrain readiness |
| `/retrain` | **🔥 Retrain + auto-reload (all in Telegram!)** |
| `/retrain force` | Force retrain tanpa check threshold |
| `/reloadmodel [v3]` | Hot reload model manual |
| `/helpml` | Show help message |

## ML Classification Thresholds

| Status | Confidence | Action |
|--------|------------|--------|
| AUTO | ≥90% | Auto-applied, trusted |
| HIGH_REVIEW | 85-90% | High confidence, review recommended |
| MEDIUM_REVIEW | 70-85% | Medium confidence, review needed |
| MANUAL | <70% | Manual classification required |

## Retraining Model

Ada **2 cara** untuk retrain model:

### 📓 Option A: Via Notebooks (Recommended for Major Updates)

Full pipeline dengan Optuna hyperparameter tuning.

```
Analyst/
├── 01_DataExploration.ipynb    # Data loading & exploration
├── 02_Preprocessing.ipynb      # Text cleaning & TF-IDF
├── 03_Training.ipynb           # Model training & optimization
├── 04_SemiSupervised.ipynb     # Semi-supervised labeling
├── 05_RelabelKendalaLogin.ipynb # Specific relabeling tasks
└── 06_UpdateMasterData.ipynb   # Update master dataset
```

**Workflow:**

1. Jalankan notebook berurutan: `01 → 02 → 03`
2. Copy artifacts: `Analyst/artifacts/` → `TelegramBotMyTech/models/`
3. Update `.env`: `MODEL_VERSION=v3`
4. Restart: `python scripts/run_all.py`

### 🚀 Option B: Via Script (Quick Retrain)

Script untuk quick retrain yang **match pipeline notebook** (word+char TF-IDF, LightGBM, Calibration):

```bash
# Manual retrain (selalu jalan)
python scripts/retrain.py

# Check threshold dulu (hanya retrain jika reviewed ≥ 100)
python scripts/retrain.py --check-threshold 100

# Force retrain (skip check)
python scripts/retrain.py --force

# Dengan custom master data
python scripts/retrain.py --master-data ../Analyst/artifacts/training_data.csv
```

**What retrain.py does:**
1. Load Master data + Reviewed data dari ML_Tracking sheet
2. Preprocess dengan ITSupportTextPreprocessor (match notebook)
3. TF-IDF: Word (1-3 ngram) + Char (3-5 ngram)
4. Train LightGBM dengan params optimal (from Optuna)
5. Probability calibration
6. Save artifacts ke `models/`

**After retrain (pilih salah satu):**
- 🔥 **Hot Reload** (no restart): `/reloadmodel v3` via Telegram
- 🔄 **Restart**: Update `.env` → `MODEL_VERSION=v3` → `python scripts/run_all.py`

### Kapan Pakai Apa?

| Scenario | Use |
|----------|-----|
| Major update, banyak class baru | Notebook (with Optuna) |
| Quick retrain, data corrections | Script |
| Production auto-retrain | Script with `--check-threshold` |

### Kenapa Notebook Tetap Diperlukan?

- ✅ **Optuna tuning**: Script pakai params yang sudah di-tune, notebook bisa re-tune
- ✅ **Interactive**: Bisa lihat metrics, confusion matrix, per-class F1
- ✅ **Char n-grams**: Handle typos lebih baik
- ✅ **Probability calibration**: Lebih akurat confidence scores
- ✅ **Iterative**: Bisa experiment dan debug

## Architecture

### Text Preprocessing

The preprocessing pipeline is consistent between training and inference:

1. URL removal
2. Email removal
3. Phone number removal
4. WO/SC/Ticket ID normalization
5. Abbreviation expansion (moban → mohon bantuan, etc.)
6. IT terms preservation (OTP, TOTP, WO, SC, etc.)
7. Lowercase conversion
8. Punctuation removal
9. Whitespace normalization
10. [SEP] token merging (Tech text [SEP] Solving text)

### Google Sheets Structure

**Logs (A-T):**
- Ticket info, technician message, ops response, SLA, Symtomps

**ML_Tracking:**
- Prediction audit trail for review and retrain

**Monitoring:**
- Daily aggregated statistics

## Legacy Files

✅ **All legacy files have been removed!** The project now uses clean architecture exclusively.

If you need to reference old code, check `legacy_backup/` folder (not tracked in git).

### Migration Summary

| Old File | New Location |
|----------|-------------|
| `main_collecting.py` | `scripts/run_collecting.py` |
| `main_reporting.py` | `scripts/run_reporting.py` |
| `collecting_bot.py` | `src/bots/collector.py` |
| `admin_commands.py` | `src/bots/admin.py` |
| `ml_classifier.py` | `src/ml/classifier.py` |
| `ml_tracking.py` | `src/ml/tracking.py` |
| `google_sheets_client.py` | `src/services/sheets.py` |
| `s3_uploader.py` | `src/services/storage.py` |
| `config.py` | `src/core/config.py` |
| `ops_parser.py` | `src/bots/parsers.py` |
| `retrain_model.py` | `scripts/retrain.py` |

## License

MIT
