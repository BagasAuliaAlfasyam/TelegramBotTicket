"""
Fix ML_Tracking Sheet Columns
==============================
Script untuk menambahkan kolom reviewed_symtomps ke sheet ML_Tracking yang sudah ada.
Jalankan sekali saja untuk migrate struktur sheet.

Struktur LAMA (11 kolom):
tech_message_id, timestamp, tech_raw_text, solving, Symtomps, sync_date, source, ml_confidence, prediction_status, review_status, logs_row_number

Struktur BARU (12 kolom):
tech_message_id, timestamp, tech_raw_text, solving, Symtomps, reviewed_symtomps, sync_date, source, ml_confidence, prediction_status, review_status, logs_row_number
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gspread
from google.oauth2.service_account import Credentials

def main():
    print("=" * 60)
    print("Fix ML_Tracking Sheet Columns")
    print("=" * 60)
    
    # Find credentials
    cred_paths = [
        PROJECT_ROOT / 'white-set-293710-9cca41a1afd6.json',
        PROJECT_ROOT / 'service_account.json',
    ]
    
    cred_file = None
    for path in cred_paths:
        if path.exists():
            cred_file = path
            break
    
    if cred_file is None:
        print("❌ No credentials file found!")
        return
    
    print(f"📄 Using credentials: {cred_file.name}")
    
    # Connect to Google Sheets
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    credentials = Credentials.from_service_account_file(str(cred_file), scopes=scopes)
    client = gspread.authorize(credentials)
    
    # Open spreadsheet
    spreadsheet_name = 'Log_Tiket_MyTech'
    print(f"📊 Opening spreadsheet: {spreadsheet_name}")
    
    try:
        spreadsheet = client.open(spreadsheet_name)
    except Exception as e:
        print(f"❌ Failed to open spreadsheet: {e}")
        return
    
    # Get ML_Tracking sheet
    try:
        worksheet = spreadsheet.worksheet("ML_Tracking")
        print("✅ Found ML_Tracking sheet")
    except gspread.WorksheetNotFound:
        print("❌ ML_Tracking sheet not found!")
        return
    
    # Get current headers
    headers = worksheet.row_values(1)
    print(f"\n📋 Current headers ({len(headers)} columns):")
    for i, h in enumerate(headers):
        print(f"   {i}: {h}")
    
    # Check if reviewed_symtomps already exists
    if 'reviewed_symtomps' in headers:
        print("\n✅ reviewed_symtomps column already exists!")
        print("   No changes needed.")
        return
    
    # Find position to insert (after Symtomps)
    if 'Symtomps' not in headers:
        print("\n❌ Symtomps column not found!")
        return
    
    symtomps_idx = headers.index('Symtomps')
    insert_col = symtomps_idx + 2  # +1 for 0-index, +1 for after
    
    print(f"\n📍 Will insert 'reviewed_symtomps' at column {insert_col} (after Symtomps)")
    
    # Confirm
    confirm = input("\n⚠️  This will modify the sheet. Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        return
    
    # Insert column
    print("\n⏳ Inserting column...")
    
    # Use gspread to insert column
    # insert_cols(col, number_of_cols, values=None, inherit_from_before=False)
    worksheet.insert_cols([[""]] * worksheet.row_count, col=insert_col)
    
    # Update header
    worksheet.update_cell(1, insert_col, "reviewed_symtomps")
    
    print("✅ Column inserted!")
    
    # Verify
    new_headers = worksheet.row_values(1)
    print(f"\n📋 New headers ({len(new_headers)} columns):")
    for i, h in enumerate(new_headers):
        print(f"   {i}: {h}")
    
    print("\n🎉 Done! ML_Tracking sheet has been updated.")
    print("   Now you can restart the bot on server.")


if __name__ == "__main__":
    main()
