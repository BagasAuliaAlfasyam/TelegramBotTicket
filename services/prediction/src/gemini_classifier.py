"""
Gemini Zero-Shot Classifier
=============================
Uses Google Gemini API for ticket classification when LightGBM
confidence is below threshold (cascade/fallback).

Role in the system:
- "Teacher" in Knowledge Distillation pattern
- Only called when LightGBM confidence < cascade_threshold
- Results stored for future LightGBM retraining
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Optional

from services.shared.telegram_alerter import TelegramAlerter

_LOGGER = logging.getLogger(__name__)

# Lazy import
_genai = None


def _import_genai():
    global _genai
    if _genai is None:
        import google.generativeai as genai  # type: ignore[import-unresolved]
        _genai = genai
    return _genai


class GeminiClassifier:
    """
    Zero-shot classifier menggunakan Google Gemini API.

    Tidak perlu training — cukup kirim prompt + daftar label,
    Gemini akan classify berdasarkan semantic understanding.
    """

    SYSTEM_PROMPT = """Kamu adalah sistem klasifikasi tiket IT support untuk Telkom Indonesia.
Tugasmu adalah mengklasifikasikan teks tiket ke dalam SATU kategori yang paling sesuai.

ATURAN PENTING:
1. Pilih HANYA SATU kategori dari daftar yang diberikan
2. Berikan confidence score 0.0 - 1.0 (seberapa yakin)
3. Jika tidak ada kategori yang cocok, pilih yang paling mendekati dengan confidence rendah
4. Jawab HANYA dalam format JSON yang diminta, tanpa penjelasan tambahan

KONTEKS DOMAIN:
- Tiket dari teknisi lapangan Telkom untuk support IT
- Bahasa Indonesia (kadang campur Inggris untuk istilah IT)
- Singkatan umum: moban=mohon bantuan, wo=work order, sc=id order awalan sc
- MIT = aplikasi MyIndiHome, MIS = Aplikasi MyStaff"""

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0  # seconds

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        timeout: float = 10.0,
        alerter: TelegramAlerter | None = None,
    ):
        self._api_key = api_key
        self._model_name = model_name
        self._timeout = timeout
        self._alerter = alerter
        self._model = None
        self._is_ready = False
        self._labels: list[str] = []

        self._init_model()

    def _init_model(self) -> None:
        """Initialize Gemini model."""
        if not self._api_key:
            _LOGGER.warning("Gemini API key not configured, classifier disabled")
            return

        try:
            genai = _import_genai()
            genai.configure(api_key=self._api_key)

            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low temperature for deterministic classification
                    top_p=0.8,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
                system_instruction=self.SYSTEM_PROMPT,
            )
            self._is_ready = True
            _LOGGER.info("Gemini classifier initialized: model=%s", self._model_name)

        except Exception as e:
            _LOGGER.error("Failed to initialize Gemini: %s", e)
            self._is_ready = False

    def set_labels(self, labels: list[str]) -> None:
        """
        Set daftar label yang valid dari LightGBM model.

        Dipanggil setelah LightGBM model loaded agar Gemini
        tahu label mana yang boleh digunakan.
        """
        self._labels = labels
        _LOGGER.info("Gemini labels updated: %d categories", len(labels))

    @property
    def is_ready(self) -> bool:
        return self._is_ready and bool(self._labels)

    def predict(
        self,
        tech_raw_text: str,
        solving: str = "",
    ) -> dict | None:
        """
        Classify ticket text using Gemini.

        Args:
            tech_raw_text: Raw text from technician
            solving: Solving text from ops

        Returns:
            dict with keys: label, confidence, inference_time_ms
            None if prediction fails
        """
        if not self.is_ready:
            _LOGGER.warning("Gemini not ready (no API key or no labels)")
            return None

        start_time = time.time()

        # Build prompt with label list (shared across retries)
        labels_str = "\n".join(f"- {label}" for label in self._labels)
        user_prompt = f"""Klasifikasikan tiket IT support berikut:

TEKS TEKNISI: {tech_raw_text}
SOLVING OPS: {solving if solving else "(tidak ada)"}

KATEGORI YANG TERSEDIA:
{labels_str}

Jawab dalam format JSON:
{{"label": "NAMA_KATEGORI", "confidence": 0.85, "reasoning": "alasan singkat"}}"""

        last_exception: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self._model.generate_content(
                    user_prompt,
                    request_options={"timeout": self._timeout},
                )

                # --- process response (moved to inner try) ---
                response_text = response.text
                if response_text is None:
                    _LOGGER.warning("Gemini returned empty response (possibly blocked by safety filters)")
                    return None
                response_text = response_text.strip()

                # Handle potential markdown code blocks
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()

                result = json.loads(response_text)

                label = result.get("label", "")
                confidence = float(result.get("confidence", 0.0))

                # Validate label exists in our list
                if label not in self._labels:
                    # Try case-insensitive match
                    label_lower_map = {lbl.lower(): lbl for lbl in self._labels}
                    matched = label_lower_map.get(label.lower())
                    if matched:
                        label = matched
                    else:
                        _LOGGER.warning("Gemini returned unknown label: %s (not in %d known labels)", label, len(self._labels))
                        return None  # Reject unknown labels entirely

                inference_time = (time.time() - start_time) * 1000

                if attempt > 1:
                    _LOGGER.info("Gemini succeeded on attempt %d", attempt)

                _LOGGER.info(
                    "Gemini prediction: %s (%.1f%%) [%.0fms] - %s",
                    label, confidence * 100, inference_time,
                    result.get("reasoning", "")[:80]
                )

                return {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "inference_time_ms": round(inference_time, 2),
                    "reasoning": result.get("reasoning", ""),
                }

            except json.JSONDecodeError as e:
                _LOGGER.error("Failed to parse Gemini response as JSON: %s", e)
                return None  # JSON parse error → no point retrying
            except Exception as e:
                last_exception = e
                err_str = str(e)
                is_rate_limit = "429" in err_str or "resource exhausted" in err_str.lower()
                is_quota = "quota" in err_str.lower() and "daily" in err_str.lower()
                is_auth = any(k in err_str.lower() for k in ("api key", "invalid credentials", "401", "403", "permission denied"))
                is_timeout = "timeout" in err_str.lower() or "timed out" in err_str.lower()

                if is_rate_limit and attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    _LOGGER.warning(
                        "Gemini 429 rate limit (attempt %d/%d), retrying in %.1fs...",
                        attempt, self.MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                else:
                    _LOGGER.error("Gemini prediction failed (attempt %d/%d): %s", attempt, self.MAX_RETRIES, e)
                    # Kirim alert Telegram sesuai tipe error (hanya pada attempt terakhir / non-retry)
                    if self._alerter and attempt == self.MAX_RETRIES or (not is_rate_limit):
                        if is_quota:
                            self._alerter and self._alerter.alert_gemini_quota_exceeded(self._model_name)
                        elif is_auth:
                            self._alerter and self._alerter.alert_gemini_auth_error(self._model_name)
                        elif is_timeout:
                            self._alerter and self._alerter.alert_gemini_timeout(self._model_name, self._timeout)
                        elif is_rate_limit:
                            self._alerter and self._alerter.alert_gemini_rate_limit(self._model_name)
                        else:
                            self._alerter and self._alerter.alert_gemini_generic_error(err_str, self._model_name)
                    if not is_rate_limit:
                        break  # Non-retriable error → stop immediately

        _LOGGER.error("Gemini prediction failed after %d attempts: %s", self.MAX_RETRIES, last_exception)
        return None

    def get_info(self) -> dict:
        """Get Gemini classifier status info."""
        return {
            "enabled": self._is_ready,
            "model": self._model_name,
            "num_labels": len(self._labels),
            "labels": self._labels,
        }
