#!/usr/bin/env python
"""
Sync Training Data - Logs → ML_Tracking
========================================
Script untuk sync data training dari Logs sheet ke ML_Tracking sheet.

ML_Tracking akan menjadi single source of truth untuk training data.

Usage:
    python scripts/sync_training_data.py
    
Flow:
    1. Load semua data dari Logs sheet (yang punya Symtomps)
    2. Format ke struktur ML_Tracking
    3. Replace/Update ML_Tracking sheet
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

# Setup path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
_LOGGER = logging.getLogger(__name__)


def sync_logs_to_ml_tracking(config: Config) -> dict:
    """
    Sync data dari Logs sheet ke ML_Tracking sheet.
    
    Returns:
        dict with sync stats
    """
    if not HAS_GSPREAD:
        raise ImportError("gspread is required for syncing")
    
    _LOGGER.info("="*60)
    _LOGGER.info("SYNC: Logs → ML_Tracking")
    _LOGGER.info("="*60)
    
    # Connect to Google Sheets
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    # Try different credential paths
    cred_paths = [
        config.google_service_account_json,
        PROJECT_ROOT / 'white-set-293710-9cca41a1afd6.json',
        PROJECT_ROOT / 'service_account.json',
    ]
    
    cred_file = None
    for path in cred_paths:
        if path.exists():
            cred_file = path
            break
    
    if cred_file is None:
        raise FileNotFoundError(f"No credentials file found. Tried: {cred_paths}")
    
    _LOGGER.info(f"Using credentials: {cred_file}")
    
    credentials = Credentials.from_service_account_file(
        str(cred_file), 
        scopes=scopes
    )
    client = gspread.authorize(credentials)
    
    # Use config spreadsheet name or fallback to default
    spreadsheet_name = config.google_spreadsheet_name
    if not spreadsheet_name:
        spreadsheet_name = 'Log_Tiket_MyTech_ML_Test'
    
    spreadsheet = client.open(spreadsheet_name)
    
    # 1. Load Logs sheet
    _LOGGER.info("Loading Logs sheet...")
    logs_ws = spreadsheet.worksheet("Logs")
    logs_data = logs_ws.get_all_values()
    
    if len(logs_data) <= 1:
        raise ValueError("Logs sheet is empty!")
    
    df_logs = pd.DataFrame(logs_data[1:], columns=logs_data[0])
    _LOGGER.info(f"  Total rows in Logs: {len(df_logs)}")
    
    # 2. Filter rows with Symtomps filled
    # Handle gspread empty strings
    df_logs['Symtomps'] = df_logs['Symtomps'].astype(str).str.strip()
    df_valid = df_logs[df_logs['Symtomps'] != ''].copy()
    _LOGGER.info(f"  Rows with Symtomps: {len(df_valid)}")
    
    if len(df_valid) == 0:
        raise ValueError("No rows with Symtomps found in Logs!")
    
    # 3. Format to ML_Tracking structure
    _LOGGER.info("Formatting to ML_Tracking structure...")
    
    # ML_Tracking columns:
    # tech_message_id, timestamp, tech_raw_text, solving, Symtomps
    ml_tracking_data = []
    
    for _, row in df_valid.iterrows():
        # Get tech message id (handle column name variations)
        tech_id = row.get('tech message id', row.get('tech_message_id', ''))
        
        # Get timestamp (combine date + time or use Response At)
        timestamp = row.get('Response At', '')
        if not timestamp:
            date = row.get('tech message date ', row.get('tech message date', ''))
            time = row.get('tech message time ', row.get('tech message time', ''))
            timestamp = f"{date} {time}".strip()
        
        # Get text fields
        tech_raw_text = row.get('tech raw text', '')
        solving = row.get('solving', '')
        symtomps = row['Symtomps']
        
        ml_tracking_data.append({
            'tech_message_id': str(tech_id),
            'timestamp': str(timestamp),
            'tech_raw_text': str(tech_raw_text),
            'solving': str(solving),
            'Symtomps': str(symtomps),
            'sync_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'LOGS'
        })
    
    df_ml = pd.DataFrame(ml_tracking_data)
    
    # Remove duplicates by tech_message_id (keep last)
    before_dedup = len(df_ml)
    df_ml = df_ml.drop_duplicates(subset=['tech_message_id'], keep='last')
    after_dedup = len(df_ml)
    if before_dedup - after_dedup > 0:
        _LOGGER.info(f"  Removed {before_dedup - after_dedup} duplicates")
    
    _LOGGER.info(f"  Final training rows: {len(df_ml)}")
    _LOGGER.info(f"  Unique classes: {df_ml['Symtomps'].nunique()}")
    
    # 4. Update ML_Tracking sheet
    _LOGGER.info("Updating ML_Tracking sheet...")
    
    try:
        ml_ws = spreadsheet.worksheet("ML_Tracking")
    except gspread.WorksheetNotFound:
        _LOGGER.info("  Creating ML_Tracking worksheet...")
        ml_ws = spreadsheet.add_worksheet(title="ML_Tracking", rows=1000, cols=10)
    
    # Clear existing data
    ml_ws.clear()
    
    # Prepare data for upload
    headers = ['tech_message_id', 'timestamp', 'tech_raw_text', 'solving', 'Symtomps', 'sync_date', 'source']
    values = [headers] + df_ml[headers].values.tolist()
    
    # Upload in batches (gspread limit)
    _LOGGER.info(f"  Uploading {len(values)-1} rows...")
    ml_ws.update(range_name='A1', values=values)
    
    _LOGGER.info("✅ Sync complete!")
    
    # Stats
    stats = {
        'logs_total': len(df_logs),
        'logs_with_symtomps': len(df_valid),
        'ml_tracking_rows': len(df_ml),
        'unique_classes': df_ml['Symtomps'].nunique(),
        'sync_date': datetime.now().isoformat()
    }
    
    # Print class distribution
    _LOGGER.info("\n📊 Class Distribution (Top 10):")
    for label, count in df_ml['Symtomps'].value_counts().head(10).items():
        _LOGGER.info(f"  {count:4d} - {label}")
    
    return stats


def main():
    """Main entry point."""
    _LOGGER.info("Starting sync...")
    
    try:
        config = Config()
        stats = sync_logs_to_ml_tracking(config)
        
        print("\n" + "="*60)
        print("🎉 SYNC COMPLETE!")
        print("="*60)
        print(f"""
📊 Summary:
   Logs total rows: {stats['logs_total']}
   Rows with Symtomps: {stats['logs_with_symtomps']}
   ML_Tracking rows: {stats['ml_tracking_rows']}
   Unique classes: {stats['unique_classes']}
   Sync date: {stats['sync_date']}

➡️ Next: Run retrain
   python scripts/retrain.py --force
        """)
        
    except Exception as e:
        _LOGGER.error(f"Sync failed: {e}")
        raise


if __name__ == "__main__":
    main()
