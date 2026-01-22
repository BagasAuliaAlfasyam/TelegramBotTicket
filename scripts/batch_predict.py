#!/usr/bin/env python
"""
Batch Predict Script
====================
Predict semua data yang Symtomps-nya kosong dan update ke Google Sheets.

Logic:
- Confidence >= 90% (AUTO): Insert langsung ke Logs.Symtomps
- Confidence < 90%: Tambah ke ML_Tracking untuk review, Logs.Symtomps tetap kosong
"""

import sys
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

# Configuration
CREDENTIALS_FILE = Path(__file__).parent.parent / 'white-set-293710-9cca41a1afd6.json'
SPREADSHEET_NAME = 'Log_Tiket_MyTech_ML_Test'
MODEL_DIR = Path(__file__).parent.parent / 'models'

# Thresholds
AUTO_THRESHOLD = 0.90  # >= 90% confidence = AUTO (insert to Symtomps)


def load_model():
    """Load model and vectorizer"""
    print("Loading model...")
    
    # Load TF-IDF vectorizer
    tfidf = joblib.load(MODEL_DIR / 'tfidf_vectorizer_v2.pkl')
    
    # Load LightGBM model
    model = lgb.Booster(model_file=str(MODEL_DIR / 'lgb_model_v2.txt'))
    
    # Load label encoder
    le = joblib.load(MODEL_DIR / 'label_encoder_v2.pkl')
    
    print(f"  Classes: {len(le.classes_)}")
    
    return tfidf, model, le


def preprocess_text(text: str) -> str:
    """Simple text preprocessing"""
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower().strip()
    # Remove multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    return text


def predict_batch(texts: list, tfidf, model, le):
    """Predict batch of texts"""
    # Preprocess
    processed = [preprocess_text(t) for t in texts]
    
    # Check if tfidf is a dict (word + char)
    if isinstance(tfidf, dict):
        X_word = tfidf['word'].transform(processed)
        X_char = tfidf['char'].transform(processed)
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


def main():
    print("=" * 70)
    print("BATCH PREDICTION - Update Logs & ML_Tracking")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"AUTO threshold: >= {AUTO_THRESHOLD * 100:.0f}%")
    
    # === 1. Connect to Google Sheets ===
    print("\n1. Connecting to Google Sheets...")
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    credentials = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
    client = gspread.authorize(credentials)
    
    spreadsheet = client.open(SPREADSHEET_NAME)
    logs_ws = spreadsheet.worksheet('Logs')
    
    # Get or create ML_Tracking
    try:
        tracking_ws = spreadsheet.worksheet('ML_Tracking')
        print("  ML_Tracking sheet exists")
    except gspread.WorksheetNotFound:
        tracking_ws = spreadsheet.add_worksheet(title='ML_Tracking', rows=1000, cols=15)
        # Add headers
        headers = [
            'timestamp', 'tech_message_id', 'tech_raw_text', 'solving',
            'predicted_symtomps', 'ml_confidence', 'prediction_status',
            'reviewed_symtomps', 'review_status', 'reviewer', 'review_timestamp',
            'logs_row_number'
        ]
        tracking_ws.update('A1', [headers])
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
    tfidf, model, le = load_model()
    
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
            tracking_row = [
                timestamp,
                str(row.get('tech message id', '')),
                str(row.get('tech raw text', ''))[:500],  # Limit length
                str(row.get('solving', ''))[:200],
                pred['label'],
                f"{pred['confidence']:.4f}",
                pred['status'],
                '',  # reviewed_symtomps
                'pending',  # review_status
                '',  # reviewer
                '',  # review_timestamp
                str(row_num)  # logs_row_number for reference
            ]
            tracking_rows.append(tracking_row)
    
    # === 7. Update Logs (AUTO predictions) ===
    if logs_updates:
        print(f"\n6. Updating Logs ({len(logs_updates)} rows)...")
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
        # Get current row count
        existing = tracking_ws.get_all_values()
        next_row = len(existing) + 1
        
        # Append in chunks
        chunk_size = 100
        for i in range(0, len(tracking_rows), chunk_size):
            chunk = tracking_rows[i:i+chunk_size]
            tracking_ws.update(f'A{next_row + i}', chunk)
            print(f"    Added rows {i+1} to {min(i+chunk_size, len(tracking_rows))}")
    
    # === 9. Summary ===
    print("\n" + "=" * 70)
    print("✅ BATCH PREDICTION COMPLETE!")
    print("=" * 70)
    print(f"""
📊 Summary:
   Total predictions: {len(predictions)}
   
   AUTO (>= 90% confidence):
     → {auto_count} rows inserted to Logs.Symtomps
   
   Need Review (< 90% confidence):
     → {review_count} rows added to ML_Tracking
     
📋 Status breakdown:
""")
    
    status_counts = {}
    for p in predictions:
        status_counts[p['status']] = status_counts.get(p['status'], 0) + 1
    
    for status, count in sorted(status_counts.items()):
        pct = count / len(predictions) * 100
        print(f"   {status}: {count} ({pct:.1f}%)")
    
    print(f"""
➡️ Next Steps:
   1. Open Google Sheets: {SPREADSHEET_NAME}
   2. Go to ML_Tracking sheet
   3. Review predictions with status != AUTO
   4. Set review_status to APPROVED or CORRECTED
   5. Run batch_predict.py again for remaining empty rows
""")


if __name__ == '__main__':
    main()
