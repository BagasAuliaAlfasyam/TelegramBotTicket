#!/usr/bin/env python
"""
Batch Predict Script
====================

Prediksi semua data yang Symtomps-nya kosong dan update ke Google Sheets.

Logika:
    - Confidence >= 90% (AUTO): Insert langsung ke Logs.Symtomps
    - Confidence < 90%: Tambah ke ML_Tracking untuk review,
      Logs.Symtomps tetap kosong

Usage:
    python scripts/batch_predict.py                  # Gunakan .env config
    python scripts/batch_predict.py --spreadsheet "Log_Tiket"  # Override spreadsheet
    python scripts/batch_predict.py --dry-run        # Preview tanpa update
    python scripts/batch_predict.py --model-version v2  # Gunakan versi spesifik

Author: Bagas Aulia Alfasyam
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import joblib
import lightgbm as lgb
from dotenv import load_dotenv

# Configuration defaults (overridable via args or config)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / 'models'

# Thresholds
AUTO_THRESHOLD = 0.90  # >= 90% confidence = AUTO (insert to Symtomps)


def get_current_version(model_dir: Path) -> str:
    """
    Dapatkan versi model saat ini dari current_version.txt.
    
    Jika file tidak ada, cari folder versi tertinggi.
    """
    version_file = model_dir / "current_version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    # Fallback: find highest version
    versions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
    if versions:
        return sorted(versions, key=lambda x: int(x[1:]))[-1]
    return "v1"


def load_model(model_dir: Path, version: str = None):
    """
    Load model dan vectorizer dari direktori berversi.
    
    Args:
        model_dir: Path ke direktori models/
        version: Versi spesifik (None = gunakan current)
    
    Returns:
        tuple: (tfidf, model, label_encoder, version)
    """
    if version is None:
        version = get_current_version(model_dir)
    
    version_dir = model_dir / version
    print(f"Loading model from: {version_dir}")
    
    if not version_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {version_dir}")
    
    # Load TF-IDF vectorizer
    tfidf_path = version_dir / 'tfidf_vectorizer.pkl'
    if not tfidf_path.exists():
        # Fallback to old naming
        tfidf_path = version_dir / f'tfidf_vectorizer_{version}.pkl'
    tfidf = joblib.load(tfidf_path)
    
    # Load LightGBM model (try .bin first, then .txt)
    model_path = version_dir / 'lgb_model.bin'
    if not model_path.exists():
        model_path = version_dir / 'lgb_model.txt'
    if not model_path.exists():
        model_path = version_dir / f'lgb_model_{version}.txt'
    model = lgb.Booster(model_file=str(model_path))
    
    # Load label encoder
    le_path = version_dir / 'label_encoder.pkl'
    if not le_path.exists():
        le_path = version_dir / f'label_encoder_{version}.pkl'
    le = joblib.load(le_path)
    
    print(f"  Version: {version}")
    print(f"  Classes: {len(le.classes_)}")
    
    return tfidf, model, le, version


def preprocess_text(text: str) -> str:
    """
    Preprocessing teks sederhana.
    
    Lowercase dan normalize whitespace.
    """
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower().strip()
    # Remove multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    return text


def predict_batch(texts: list, tfidf, model, le):
    """
    Prediksi batch teks sekaligus.
    
    Args:
        texts: List teks yang akan diprediksi
        tfidf: TF-IDF vectorizer (bisa dict word+char atau single)
        model: LightGBM model
        le: Label encoder
    
    Returns:
        list[dict]: List prediksi dengan keys: label, confidence, status
    """
    # Preprocess
    processed = [preprocess_text(t) for t in texts]
    
    # Check if tfidf is a dict (word + char)
    if isinstance(tfidf, dict):
        # Handle different key naming conventions
        word_key = 'word_tfidf' if 'word_tfidf' in tfidf else 'word'
        char_key = 'char_tfidf' if 'char_tfidf' in tfidf else 'char'
        X_word = tfidf[word_key].transform(processed)
        X_char = tfidf[char_key].transform(processed)
        from scipy.sparse import hstack
        X = hstack([X_word, X_char])
    else:
        X = tfidf.transform(processed)
    
    # Predict probabilities
    probs = model.predict(X)
    
    # Get predictions
    predictions = []
    for i, prob in enumerate(probs):
        max_idx = np.argmax(prob)
        confidence = prob[max_idx]
        label = le.classes_[max_idx]
        predictions.append({
            'label': label,
            'confidence': float(confidence),
            'status': 'AUTO' if confidence >= AUTO_THRESHOLD else (
                'HIGH_REVIEW' if confidence >= 0.70 else (
                    'MEDIUM_REVIEW' if confidence >= 0.50 else 'MANUAL'
                )
            )
        })
    
    return predictions


def parse_args():
    """
    Parse command line arguments.
    
    Mendukung override spreadsheet, dry-run mode, dan model version.
    """
    parser = argparse.ArgumentParser(description='Batch predict Symtomps for empty rows')
    parser.add_argument('--spreadsheet', '-s', type=str, help='Override spreadsheet name from config')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Preview without updating sheets')
    parser.add_argument('--model-version', '-m', type=str, help='Specific model version to use (e.g., v1, v2)')
    parser.add_argument('--env-file', '-e', type=str, help='Path to .env file (default: project .env)')
    parser.add_argument('--credentials', '-c', type=str, help='Path to Google service account JSON')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load .env from custom path or try common locations
    if args.env_file:
        env_path = Path(args.env_file)
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"Loaded env from: {env_path}")
        else:
            print(f"⚠️ Warning: env file not found: {env_path}")
    else:
        # Try common locations
        env_paths = [
            PROJECT_ROOT / '.env.local',
            PROJECT_ROOT / '.env',
            Path.home() / '.telegram-bot.env',
            Path('/etc/telegram-bot/.env'),
        ]
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path, override=True)
                print(f"Loaded env from: {env_path}")
                break
    
    # Determine spreadsheet name (priority: arg > env var > default)
    spreadsheet_name = args.spreadsheet or os.getenv('GOOGLE_SPREADSHEET_NAME', '')
    if not spreadsheet_name:
        print("❌ Error: No spreadsheet name. Set GOOGLE_SPREADSHEET_NAME or use --spreadsheet")
        sys.exit(1)
    
    # Determine credentials file (priority: arg > env var > default)
    if args.credentials:
        cred_file = Path(args.credentials)
    else:
        cred_env = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', '')
        if cred_env:
            cred_file = Path(cred_env)
        else:
            cred_file = PROJECT_ROOT / 'white-set-293710-9cca41a1afd6.json'
    
    if not cred_file.exists():
        print(f"❌ Error: Credentials file not found: {cred_file}")
        sys.exit(1)
    
    # Determine model dir
    model_dir_env = os.getenv('MODEL_DIR', '')
    if model_dir_env:
        model_dir = Path(model_dir_env)
    else:
        model_dir = DEFAULT_MODEL_DIR
    
    print("=" * 70)
    print("BATCH PREDICTION - Update Logs & ML_Tracking")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Spreadsheet: {spreadsheet_name}")
    print(f"Credentials: {cred_file}")
    print(f"Model dir: {model_dir}")
    print(f"AUTO threshold: >= {AUTO_THRESHOLD * 100:.0f}%")
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made")
    
    # === 1. Connect to Google Sheets ===
    print("\n1. Connecting to Google Sheets...")
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    credentials = Credentials.from_service_account_file(str(cred_file), scopes=scopes)
    client = gspread.authorize(credentials)
    
    spreadsheet = client.open(spreadsheet_name)
    logs_ws = spreadsheet.worksheet('Logs')
    
    # Get or create ML_Tracking
    # Headers aligned with sync_training_data.py + extra columns for review
    ml_tracking_headers = [
        'tech_message_id', 'timestamp', 'tech_raw_text', 'solving',
        'Symtomps', 'sync_date', 'source',
        'ml_confidence', 'prediction_status', 'review_status', 'logs_row_number'
    ]
    
    try:
        tracking_ws = spreadsheet.worksheet('ML_Tracking')
        print("  ML_Tracking sheet exists")
        
        # Check if headers need updating (add new columns if missing)
        existing_headers = tracking_ws.row_values(1)
        if len(existing_headers) < len(ml_tracking_headers):
            # Add missing columns to header
            for i, h in enumerate(ml_tracking_headers):
                if i >= len(existing_headers):
                    tracking_ws.update_cell(1, i + 1, h)
            print(f"  Updated ML_Tracking headers (added {len(ml_tracking_headers) - len(existing_headers)} columns)")
            
    except gspread.WorksheetNotFound:
        tracking_ws = spreadsheet.add_worksheet(title='ML_Tracking', rows=1000, cols=15)
        tracking_ws.update(values=[ml_tracking_headers], range_name='A1')
        print("  Created ML_Tracking sheet")
    
    # === 2. Fetch Logs data ===
    print("\n2. Fetching Logs data...")
    data = logs_ws.get_all_values()
    headers = data[0]
    df = pd.DataFrame(data[1:], columns=headers)
    print(f"  Total rows: {len(df)}")
    
    # Find Symtomps column index (1-based for Sheets)
    symtomps_col_idx = headers.index('Symtomps') + 1
    print(f"  Symtomps column: {symtomps_col_idx} ({headers[symtomps_col_idx-1]})")
    
    # === 3. Filter rows that need prediction ===
    # Clean Symtomps column
    df['Symtomps_clean'] = df['Symtomps'].astype(str).str.strip()
    df['Symtomps_clean'] = df['Symtomps_clean'].replace(['nan', 'None', 'NaN', ''], '')
    
    # Find empty rows
    empty_mask = df['Symtomps_clean'] == ''
    df_empty = df[empty_mask].copy()
    print(f"  Rows needing prediction: {len(df_empty)}")
    
    if len(df_empty) == 0:
        print("\n✅ No rows need prediction!")
        return
    
    # === 4. Load model and predict ===
    print("\n3. Loading model...")
    tfidf, model, le, model_version = load_model(model_dir, args.model_version)
    
    print("\n4. Predicting...")
    # Combine tech raw text and solving for prediction
    texts = []
    for _, row in df_empty.iterrows():
        tech_text = str(row.get('tech raw text', ''))
        solving = str(row.get('solving', ''))
        combined = f"{tech_text} {solving}".strip()
        texts.append(combined)
    
    predictions = predict_batch(texts, tfidf, model, le)
    
    # === 5. Categorize results ===
    auto_count = sum(1 for p in predictions if p['status'] == 'AUTO')
    review_count = len(predictions) - auto_count
    print(f"  AUTO (>= 90%): {auto_count}")
    print(f"  Need Review (< 90%): {review_count}")
    
    # === 6. Prepare updates ===
    print("\n5. Preparing updates...")
    
    # For Logs: Update Symtomps for AUTO only
    logs_updates = []  # [(row_num, value)]
    
    # For ML_Tracking: Add all non-AUTO predictions
    tracking_rows = []
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for i, (idx, row) in enumerate(df_empty.iterrows()):
        pred = predictions[i]
        row_num = idx + 2  # 1-based, plus header
        
        if pred['status'] == 'AUTO':
            # Insert to Logs.Symtomps directly
            logs_updates.append((row_num, pred['label']))
        else:
            # Add to ML_Tracking for review
            # Structure matches sync_training_data.py + extra columns for review
            tracking_row = [
                str(row.get('tech message id', '')),  # tech_message_id
                timestamp,  # timestamp
                str(row.get('tech raw text', ''))[:500],  # tech_raw_text
                str(row.get('solving', ''))[:200],  # solving
                pred['label'],  # Symtomps (predicted)
                timestamp,  # sync_date
                'BATCH_PREDICT',  # source
                f"{pred['confidence']:.4f}",  # ml_confidence
                pred['status'],  # prediction_status
                'pending',  # review_status
                str(row_num),  # logs_row_number
            ]
            tracking_rows.append(tracking_row)
    
    # === 7. Update Logs (AUTO predictions) ===
    if logs_updates:
        print(f"\n6. Updating Logs ({len(logs_updates)} rows)...")
        if args.dry_run:
            print("    [DRY RUN] Would update the following rows:")
            for row_num, value in logs_updates[:10]:
                print(f"      Row {row_num}: {value}")
            if len(logs_updates) > 10:
                print(f"      ... and {len(logs_updates) - 10} more")
        else:
            # Batch update using cell notation
            # Group by chunks to avoid quota
            chunk_size = 100
            for i in range(0, len(logs_updates), chunk_size):
                chunk = logs_updates[i:i+chunk_size]
                cells = []
                for row_num, value in chunk:
                    cells.append({
                        'range': f'{gspread.utils.rowcol_to_a1(row_num, symtomps_col_idx)}',
                        'values': [[value]]
                    })
                logs_ws.batch_update(cells)
                print(f"    Updated rows {i+1} to {min(i+chunk_size, len(logs_updates))}")
    
    # === 8. Update ML_Tracking (non-AUTO for review) ===
    if tracking_rows:
        print(f"\n7. Adding to ML_Tracking ({len(tracking_rows)} rows)...")
        if args.dry_run:
            print("    [DRY RUN] Would add the following to ML_Tracking:")
            for row in tracking_rows[:5]:
                print(f"      {row[4]} ({row[5]}) - {row[6]}")
            if len(tracking_rows) > 5:
                print(f"      ... and {len(tracking_rows) - 5} more")
        else:
            # Get current row count
            existing = tracking_ws.get_all_values()
            next_row = len(existing) + 1
            
            # Expand sheet if needed
            rows_needed = next_row + len(tracking_rows)
            if rows_needed > tracking_ws.row_count:
                tracking_ws.add_rows(rows_needed - tracking_ws.row_count + 100)
                print(f"    Expanded sheet to {tracking_ws.row_count} rows")
            
            # Append in chunks
            chunk_size = 100
            for i in range(0, len(tracking_rows), chunk_size):
                chunk = tracking_rows[i:i+chunk_size]
                tracking_ws.update(values=chunk, range_name=f'A{next_row + i}')
                print(f"    Added rows {i+1} to {min(i+chunk_size, len(tracking_rows))}")
    
    # === 9. Summary ===
    print("\n" + "=" * 70)
    if args.dry_run:
        print("🔍 DRY RUN COMPLETE - No changes made")
    else:
        print("✅ BATCH PREDICTION COMPLETE!")
    print("=" * 70)
    print(f"""
📊 Summary:
   Total predictions: {len(predictions)}
   Model version: {model_version}
   
   AUTO (>= 90% confidence):
     → {auto_count} rows {'would be' if args.dry_run else ''} inserted to Logs.Symtomps
   
   Need Review (< 90% confidence):
     → {review_count} rows {'would be' if args.dry_run else ''} added to ML_Tracking
     
📋 Status breakdown:
""")
    
    status_counts = {}
    for p in predictions:
        status_counts[p['status']] = status_counts.get(p['status'], 0) + 1
    
    for status, count in sorted(status_counts.items()):
        pct = count / len(predictions) * 100
        print(f"   {status}: {count} ({pct:.1f}%)")
    
    if args.dry_run:
        print(f"""
➡️ To apply changes, run without --dry-run:
   python scripts/batch_predict.py --spreadsheet "{spreadsheet_name}"
""")
    else:
        print(f"""
➡️ Next Steps:
   1. Open Google Sheets: {spreadsheet_name}
   2. Go to ML_Tracking sheet
   3. Review predictions with status != AUTO
   4. Set review_status to APPROVED or CORRECTED
   5. Run batch_predict.py again for remaining empty rows
""")


if __name__ == '__main__':
    main()
