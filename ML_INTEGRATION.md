# ML Integration for MyTech Telegram Bot

## Overview

Model ML v2.0 terintegrasi untuk klasifikasi otomatis tiket berdasarkan `tech_raw_text` dan `solving`.

## Files Created/Modified

### New Files
- `ml_classifier.py` - ML model wrapper dengan predict method
- `ml_tracking.py` - Audit trail & monitoring ke Google Sheets
- `admin_commands.py` - Admin command handlers (/stats, /report, etc)
- `models/` - Folder berisi model artifacts

### Modified Files
- `main_collecting.py` - Initialize ML components
- `collecting_bot.py` - Integrate ML prediction di message handler
- `google_sheets_client.py` - Support column T (Symtomps)
- `requirements.txt` - Added ML dependencies

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
| Model | LightGBM v2.0 |
| Classes | 39 symptom categories |
| Training Accuracy | 83.13% |
| Accuracy at AUTO (≥90%) | 95.85% |
| Features | TF-IDF (tech_raw_text + solving) |

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
| `/model_status` | Info model saat ini |
| `/pending_review` | Items yang perlu direview |
| `/help_ml` | Help admin commands |

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

# Run bot
python main_collecting.py
```

## Logs

Bot akan log:
```
2025-06-05 10:30:15 | main_collecting | INFO | ✅ ML Classifier initialized - Model v2.0, 39 classes
2025-06-05 10:30:20 | collecting_bot | DEBUG | ML Prediction: USER AUTH FAILED (92.5%) - AUTO [3.2ms]
```

## Retraining Model

Untuk retrain model dengan data baru:

1. Export data dari `ML_Tracking` sheet (reviewed items only)
2. Jalankan training notebook di `Analyst/Cleaning.ipynb`
3. Copy model baru ke `TelegramBotMyTech/models/`
4. Restart bot
