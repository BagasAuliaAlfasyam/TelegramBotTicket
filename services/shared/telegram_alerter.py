"""
Telegram Alerter (Shared Utility)
===================================
Kirim alert system ke TELEGRAM_BOT_TOKEN_REPORTING tanpa dependency tambahan.
Menggunakan stdlib urllib saja (sudah ada di Python).

Fitur:
- Cooldown per alert_key → anti-spam (default 30 menit per jenis error)
- Pesan spesifik per tipe error Gemini (429, quota, auth, dll)
- Fire-and-forget di background thread agar tidak blok prediction
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

_LOGGER = logging.getLogger(__name__)


class TelegramAlerter:
    """
    Kirim alert ke Telegram reporting bot.

    Contoh penggunaan:
        alerter = TelegramAlerter(bot_token="...", chat_ids=[294278923, ...])
        alerter.alert_gemini_rate_limit(model="gemini-2.0-flash")
    """

    # Cooldown default per alert_key (detik)
    DEFAULT_COOLDOWN = 30 * 60  # 30 menit

    TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        bot_token: str,
        chat_ids: list[int],
        cooldown_seconds: int = DEFAULT_COOLDOWN,
    ) -> None:
        self._token = bot_token
        self._chat_ids = chat_ids
        self._cooldown = cooldown_seconds
        # { alert_key: last_sent_timestamp }
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return bool(self._token) and bool(self._chat_ids)

    # ------------------------------------------------------------------
    # Public alert methods (spesifik per tipe error)
    # ------------------------------------------------------------------

    def alert_gemini_rate_limit(self, model: str = "gemini-2.0-flash") -> None:
        """429 Resource Exhausted — kena rate limit / quota."""
        msg = (
            "⚠️ *[Prediction API] Gemini Rate Limit (429)*\n\n"
            f"Model `{model}` mengembalikan error *429 Resource Exhausted*.\n\n"
            "📋 *Kemungkinan penyebab:*\n"
            "• RPM (Request per Minute) atau TPM (Token per Minute) sudah habis\n"
            "• Terlalu banyak tiket confidence rendah dalam waktu singkat\n\n"
            "✅ *Tidak perlu tindakan segera* — sistem sudah retry otomatis.\n"
            "Jika terjadi terus-menerus:\n"
            "1️⃣ Cek quota di https://aistudio.google.com/apikey\n"
            "2️⃣ Pertimbangkan upgrade ke Gemini API paid tier\n"
            "3️⃣ Atau turunkan `GEMINI_CASCADE_THRESHOLD` di `.env` (misal 0.65) "
            "untuk mengurangi frekuensi call Gemini"
        )
        self._send_with_cooldown("gemini_rate_limit", msg)

    def alert_gemini_quota_exceeded(self, model: str = "gemini-2.0-flash") -> None:
        """Daily quota habis — perlu upgrade atau tunggu reset."""
        msg = (
            "🚫 *[Prediction API] Gemini Quota Habis*\n\n"
            f"Model `{model}` tidak bisa digunakan — daily quota sudah habis.\n\n"
            "🔧 *Tindakan yang diperlukan:*\n"
            "1️⃣ *Opsi A — Gratis:* Tunggu reset quota otomatis (biasanya tengah malam UTC)\n"
            "2️⃣ *Opsi B — Bayar:* Aktifkan billing di Google Cloud untuk quota lebih besar\n"
            "   → https://console.cloud.google.com/billing\n"
            "3️⃣ *Opsi C — Ganti Key:* Buat Gemini API key baru di project berbeda\n"
            "   → https://aistudio.google.com/apikey\n\n"
            "⚠️ Selama quota habis, prediksi tetap jalan via LightGBM saja (tanpa Gemini fallback)"
        )
        self._send_with_cooldown("gemini_quota_exceeded", msg)

    def alert_gemini_auth_error(self, model: str = "gemini-2.0-flash") -> None:
        """API key invalid / expired."""
        msg = (
            "🔑 *[Prediction API] Gemini API Key Invalid*\n\n"
            f"Model `{model}` mengembalikan error autentikasi.\n\n"
            "🔧 *Tindakan yang diperlukan:*\n"
            "1️⃣ Buat / perbarui `GEMINI_API_KEY` di file `.env.production`\n"
            "   → https://aistudio.google.com/apikey\n"
            "2️⃣ Restart service prediction-api setelah update:\n"
            "   `docker compose restart prediction-api`\n\n"
            "⚠️ Gemini fallback *dinonaktifkan sementara* sampai key valid"
        )
        self._send_with_cooldown("gemini_auth_error", msg)

    def alert_gemini_timeout(self, model: str = "gemini-2.0-flash", timeout_s: float = 10.0) -> None:
        """Request timeout — Gemini lambat/tidak merespons."""
        msg = (
            "⏱️ *[Prediction API] Gemini Timeout*\n\n"
            f"Request ke `{model}` timeout setelah `{timeout_s}s`.\n\n"
            "Kemungkinan Gemini API sedang lambat atau gangguan sementara.\n"
            "Sistem tetap menggunakan LightGBM sebagai fallback.\n\n"
            "Jika sering terjadi, pertimbangkan naikkan `GEMINI_TIMEOUT` di `.env`"
        )
        self._send_with_cooldown("gemini_timeout", msg)

    def alert_gemini_generic_error(self, error: str, model: str = "gemini-2.0-flash") -> None:
        """Error tidak dikenal dari Gemini."""
        # Pendekkan pesan error supaya tidak terlalu panjang
        short_err = error[:200] if len(error) > 200 else error
        msg = (
            "❌ *[Prediction API] Gemini Error*\n\n"
            f"Model `{model}` mengalami error tidak dikenal:\n"
            f"`{short_err}`\n\n"
            "Sistem tetap menggunakan LightGBM sebagai fallback.\n"
            "Mohon cek log prediction-api untuk detail lebih lanjut."
        )
        self._send_with_cooldown("gemini_generic_error", msg)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send_with_cooldown(self, alert_key: str, message: str) -> None:
        """Kirim alert hanya jika belum dikirim dalam cooldown period."""
        now = time.time()
        with self._lock:
            last = self._last_sent.get(alert_key, 0.0)
            if now - last < self._cooldown:
                remaining = int(self._cooldown - (now - last))
                _LOGGER.debug(
                    "Alert '%s' skipped (cooldown %ds remaining)", alert_key, remaining
                )
                return
            self._last_sent[alert_key] = now

        # Kirim di background thread agar tidak blok prediction
        threading.Thread(
            target=self._send_to_all,
            args=(message,),
            daemon=True,
            name=f"tg-alert-{alert_key}",
        ).start()

    def _send_to_all(self, message: str) -> None:
        """Kirim message ke semua chat_ids."""
        for chat_id in self._chat_ids:
            try:
                self._send_single(chat_id, message)
            except Exception as e:
                _LOGGER.warning("Failed to send Telegram alert to %s: %s", chat_id, e)

    def _send_single(self, chat_id: int, message: str) -> None:
        """HTTP POST ke Telegram Bot API (stdlib urllib, no extra deps)."""
        url = self.TELEGRAM_API.format(token=self._token)
        payload = json.dumps({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                _LOGGER.warning("Telegram API returned status %d", resp.status)
            else:
                _LOGGER.info("Telegram alert sent to chat_id=%s", chat_id)
