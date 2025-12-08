# Telegram Ops Collector Bot

Bot Telegram yang merekap respon tim Ops untuk pesan teknisi, menyimpan log ke Google Sheets, dan mengunggah lampiran ke S3 kompatibel (MinIO/Wasabi/AWS). Bot ini memakai dua token: satu untuk membaca pesan grup teknisi dan satu lagi untuk mengirim notifikasi/reporting.

## Fitur utama
- Mencatat respon Ops dengan format `solusi, APP -xx` (contoh: `Restart service, MIT -bg`) hanya untuk kode app yang diizinkan (`MIT`, `MIS`).
- Menandai respon awal/ack jika pesan mengandung kata `oncek`.
- Menyimpan catatan ke Google Sheets (tanggal, waktu, teks tiket, media type, solver, SLA, dsb).
- Mengunggah lampiran pesan teknisi ke S3 dan menyimpan URL publik di sheet.
- Bot reporting mengirim balasan/notifikasi ke grup target.

## Arsitektur singkat
- `main_collecting.py` mem-boot Application Telegram, memuat config, Google Sheets client, dan uploader S3.
- `collecting_bot.py` meng-handle pesan, parsing format, hitung SLA, upload lampiran, dan tulis ke sheet.
- `google_sheets_client.py` membungkus gspread untuk append/update baris log.
- `s3_uploader.py` membungkus boto3 untuk unggah bytes ke bucket publik.
- `ops_parser.py` validasi format pesan Ops.
- `debug_chat_id_bot.py` bot kecil untuk mendapatkan `chat_id` grup.

## Prasyarat
- Python 3.10+.
- Kredensial service account Google Sheets (file JSON).
- Bucket S3 yang dapat diakses publik (atau endpoint MinIO/Wasabi sejenis).
- Token Bot Telegram (2 buah: collecting & reporting).

## Instalasi cepat
```bash
python -m venv .venv
.venv\Scripts\activate  # PowerShell: .venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install python-telegram-bot gspread google-auth boto3 python-dotenv
```
Sesuaikan versi paket jika diperlukan oleh environment Anda.

## Konfigurasi `.env`
Buat file `.env` di root proyek:
```
TELEGRAM_BOT_TOKEN_COLLECTING=...
TELEGRAM_BOT_TOKEN_REPORTING=...

GOOGLE_SERVICE_ACCOUNT_JSON=white-set-293710-9cca41a1afd6.json
GOOGLE_SPREADSHEET_NAME=NamaSpreadsheet
GOOGLE_WORKSHEET_NAME=NamaWorksheet

TARGET_GROUP_COLLECTING=123456789      # optional, chat_id grup teknisi; kosongkan untuk semua
TARGET_GROUP_REPORTING=123456789       # optional, kemana notifikasi dikirim

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=...
AWS_S3_ENDPOINT=https://your-s3-endpoint
AWS_S3_PUBLIC_BASE_URL=                # optional, default dibangun dari endpoint + bucket
AWS_S3_REGION=                         # optional
AWS_S3_MEDIA_PREFIX=tech-media/        # optional, suffix "/" otomatis ditambahkan
AWS_S3_SIGNATURE_VERSION=s3            # atau s3v4
AWS_S3_ADDRESSING_STYLE=virtual        # atau path
```
Pastikan path `GOOGLE_SERVICE_ACCOUNT_JSON` mengarah ke file JSON kredensial yang sudah ada.

## Cara menjalankan
```bash
.venv\Scripts\activate
python main_collecting.py
```
Bot akan polling dan memproses pesan text di grup yang dituju.

## Format pesan yang didukung
- **Ack awal**: balas pesan teknisi dengan teks yang mengandung kata `oncek`. Bot menandai `isOncek=true` dan mencatat waktu respon.
- **Solusi**: balas pesan teknisi dengan pola `solusi, APP -xx`
  - `solusi` : teks solusi bebas
  - `APP` : `MIT` atau `MIS`
  - `-xx`  : inisial solver (dipetakan ke nama di `collecting_bot.py`)

Media lampiran pada pesan teknisi akan di-upload ke S3 dengan nama file otomatis.

## Mendapatkan chat_id grup
Jika belum tahu `chat_id`, jalankan:
```bash
.venv\Scripts\activate
python debug_chat_id_bot.py
```
Invite bot ke grup lalu kirim `/chatid` di grup; bot akan membalas detail `chat_id`.

## Catatan lain
- Untuk SLA, waktu respon dihitung dari timestamp pesan teknisi ke ack/solusi; batas default 15 menit.
- Jika ingin menambahkan app baru atau peta inisial->nama solver, edit konstanta di `collecting_bot.py`.
