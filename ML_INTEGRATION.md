# ML Integration for MyTech Telegram Bot

## Overview

Model ML terintegrasi untuk klasifikasi otomatis tiket berdasarkan `tech_raw_text` dan `solving`.

## Project Structure

### Source Code
- `src/ml/classifier.py` - ML model wrapper dengan predict method
- `src/ml/tracking.py` - Audit trail & monitoring ke Google Sheets
- `src/ml/preprocessing.py` - Domain-aware text preprocessing
- `src/bots/admin.py` - Admin command handlers (/stats, /report, etc)

### Model Artifacts (Versioned)
```
models/
├── v1/
│   ├── lgb_model.bin         # LightGBM model (binary)
│   ├── tfidf_vectorizer.pkl  # Word + Char TF-IDF
│   ├── label_encoder.pkl     # Label encoder
│   ├── preprocessor.pkl      # Text preprocessor
│   └── metadata.json         # Model metadata
├── current_version.txt       # Active version pointer
└── versions.json             # Version history
```

### Scripts
- `scripts/retrain.py` - Retrain model dengan auto-versioning
- `scripts/sync_training_data.py` - Sync Logs → ML_Tracking sheet

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Telegram Message                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    collecting_bot.py                            │
│                    (handle_ops_reply)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ ML Predict  │  │  Log Row    │  │ ML Tracking │
     │ (col T)     │  │  to Logs    │  │ Audit Trail │
     └─────────────┘  └─────────────┘  └─────────────┘
```

## ML Model Specs

| Property | Value |
|----------|-------|
| Model | LightGBM (versioned) |
| Classes | 35 symptom categories |
| Features | Word TF-IDF (1-3 ngram) + Char TF-IDF (3-5 ngram) |
| Training Source | ML_Tracking sheet (single source of truth) |
| Hot Reload | ✅ Yes, via /reloadmodel command |

## Confidence Thresholds

| Status | Confidence | Action |
|--------|------------|--------|
| AUTO | ≥ 90% | Langsung pakai, tidak perlu review |
| HIGH_REVIEW | 85-90% | Prioritas tinggi untuk review |
| MEDIUM_REVIEW | 70-85% | Review jika sempat |
| MANUAL | < 70% | Wajib manual classification |

## Google Sheets Structure

### Logs Sheet (Production)
- Columns A-S: Existing columns
- Column T: `Symtomps` (predicted symptom label)

### ML_Tracking Sheet (Audit Trail)
| Column | Description |
|--------|-------------|
| tech_message_id | ID pesan teknisi |
| timestamp | Waktu prediksi |
| tech_raw_text | Text dari teknisi |
| solving | Text solving dari ops |
| predicted_symtomps | Hasil prediksi ML |
| ml_confidence | Confidence score (0-1) |
| prediction_status | AUTO/HIGH/MEDIUM/MANUAL |
| reviewed_symtomps | Hasil review manual |
| review_status | pending/reviewed |
| inference_time_ms | Waktu inference |

### Monitoring Sheet (Daily Stats)
| Column | Description |
|--------|-------------|
| date | Tanggal |
| total_predictions | Total prediksi |
| avg_confidence | Rata-rata confidence |
| auto_count | Jumlah AUTO |
| high_review_count | Jumlah HIGH_REVIEW |
| medium_review_count | Jumlah MEDIUM_REVIEW |
| manual_count | Jumlah MANUAL |
| reviewed_count | Jumlah sudah direview |
| accuracy | Akurasi dari reviewed |
| model_version | Versi model |

## Admin Commands

| Command | Description |
|---------|-------------|
| `/stats` | Statistik prediksi hari ini |
| `/report weekly` | Report 7 hari terakhir |
| `/report monthly` | Report 30 hari terakhir |
| `/modelstatus` | Info model saat ini |
| `/pendingreview` | Items yang perlu direview |
| `/retrainstatus` | Cek data tersedia untuk retrain |
| `/retrain` | Retrain model + auto-reload |
| `/retrain force` | Paksa retrain tanpa cek threshold |
| `/reloadmodel` | Hot reload model |
| `/helpml` | Help admin commands |

## Sample Output

```
/stats

📊 Today's ML Stats (05 Jun 2025)

📈 Total Predictions: 127
🎯 Avg Confidence: 89.2%

Distribution:
  ✅ AUTO (≥90%): 98 (77.2%)
  🔶 HIGH REVIEW: 15
  🟡 MEDIUM REVIEW: 10
  🔴 MANUAL: 4

Review Status:
  📋 Pending Review: 29
  ✅ Reviewed Today: 12

🤖 Model: v2.0
```

## Running the Bot

```bash
# Install dependencies
pip install -r requirements.txt

# Run both bots
python scripts/run_all.py
```

## Logs

Bot akan log:
```
2025-06-05 10:30:15 | main_collecting | INFO | ✅ ML Classifier initialized - Model v2.0, 39 classes
2025-06-05 10:30:20 | collecting_bot | DEBUG | ML Prediction: USER AUTH FAILED (92.5%) - AUTO [3.2ms]
```

## Retraining Model

### Via Telegram (Recommended)
```
/retrain force
```
Model akan auto-reload setelah selesai. Tidak perlu restart bot!

### Via Terminal
```bash
# Sync data dulu (Logs → ML_Tracking)
python scripts/sync_training_data.py

# Retrain
python scripts/retrain.py --force
```

Model baru akan tersimpan di folder versioned (`models/v2/`, `models/v3/`, dst) dan otomatis aktif.
