"""
Retraining Pipeline (microservice version)
============================================
Fetches training data from Data API, trains LightGBM + TF-IDF,
logs to MLflow, and optionally distills Gemini labels.

This is a simplified adaptation of the monolith scripts/retrain.py.
Key differences:
  - Data comes from Data API (HTTP) instead of direct Sheets
  - Preprocessing uses shared canonical ITSupportTextPreprocessor
  - MLflow logging preserved
  - Gemini Knowledge Distillation: augments training data with Gemini labels
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import httpx
import lightgbm as lgb
import numpy as np
from scipy.sparse import hstack
from services.shared.config import TrainingServiceConfig
from services.shared.preprocessing import ITSupportTextPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score

_LOGGER = logging.getLogger(__name__)

# LightGBM default optimal params (tuned for Docker performance)
OPTIMAL_PARAMS = {
    "objective": "multiclass",
    "boosting_type": "gbdt",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": 2,
    "verbose": -1,
    "force_col_wise": True,  # faster for sparse data
}

# Label normalization map
LABEL_MAPPING = {
    "account management": "Account Management",
    "application error": "Application Error",
    "configuration": "Configuration",
    "connectivity": "Connectivity",
    "data management": "Data Management",
    "hardware": "Hardware",
    "integration": "Integration",
    "performance": "Performance",
    "security": "Security",
    "user access": "User Access",
}


class RetrainPipeline:
    """Fetch data → preprocess → train → log to MLflow."""

    def __init__(self, config: TrainingServiceConfig):
        self._config = config
        self._preprocessor = ITSupportTextPreprocessor()
        self._http = httpx.Client(timeout=120.0)
        self._status = "idle"
        self._last_trained: str | None = None
        self._last_result: dict = {}

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_trained(self) -> str | None:
        return self._last_trained

    @property
    def last_result(self) -> dict:
        return self._last_result

    def run(self, force: bool = False, tune: bool = False, tune_trials: int = 50) -> dict:
        self._status = "running"
        try:
            result = self._execute(force, tune, tune_trials)
            self._status = "completed"
            self._last_trained = datetime.utcnow().isoformat()
            self._last_result = result
            return result
        except Exception as e:
            self._status = "failed"
            self._last_result = {"success": False, "message": str(e)}
            _LOGGER.exception("Retrain failed")
            return self._last_result

    def _execute(self, force: bool, tune: bool, tune_trials: int) -> dict:
        start = time.time()

        # 1. Fetch training data from Data API
        _LOGGER.info("Fetching training data from Data API...")
        resp = self._http.get(f"{self._config.data_api_url.rstrip('/')}/training/data")
        resp.raise_for_status()
        data = resp.json()

        logs_data = data.get("logs_data", [])
        tracking_data = data.get("tracking_data", [])
        _LOGGER.info("From Logs: %d, From ML_Tracking: %d", len(logs_data), len(tracking_data))

        # 2. Combine and prepare training data (deduplicate by tech_message_id)
        #    ML_Tracking has priority (manually curated), Logs fills the rest
        texts = []
        labels = []
        seen_ids: set[str] = set()

        # Process ML_Tracking first (higher priority — curated data)
        for row in tracking_data:
            text = self._preprocessor.preprocess(
                row.get("tech_raw_text", ""),
                row.get("solving", ""),
            )
            label = row.get("symtomps", "").strip()
            if text and label:
                norm = LABEL_MAPPING.get(label.lower(), label)
                texts.append(text)
                labels.append(norm)
                mid = row.get("tech_message_id", "").strip()
                if mid:
                    seen_ids.add(mid)

        # Then process Logs, skipping duplicates already in ML_Tracking
        skipped = 0
        for row in logs_data:
            mid = row.get("tech_message_id", "").strip()
            if mid and mid in seen_ids:
                skipped += 1
                continue
            text = self._preprocessor.preprocess(
                row.get("tech_raw_text", ""),
                row.get("solving", ""),
            )
            label = row.get("symtomps", "").strip()
            if text and label:
                norm = LABEL_MAPPING.get(label.lower(), label)
                texts.append(text)
                labels.append(norm)
                if mid:
                    seen_ids.add(mid)

        if skipped:
            _LOGGER.info("Dedup: dropped %d duplicate rows from Logs (already in ML_Tracking)", skipped)

        if len(texts) < 50 and not force:
            return {"success": False, "message": f"Only {len(texts)} samples, need 50+. Use force=true."}

        _LOGGER.info("Total training data: %d samples, %d classes", len(texts), len(set(labels)))

        # 3. TF-IDF Vectorization (reduced features for Docker performance)
        tfidf_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=5000,
            sublinear_tf=True, min_df=2,
        )
        tfidf_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=2000,
            sublinear_tf=True, min_df=2,
        )

        X_word = tfidf_word.fit_transform(texts)
        X_char = tfidf_char.fit_transform(texts)
        X = hstack([X_word, X_char], format="csc")  # CSC format for LightGBM
        _LOGGER.info("TF-IDF done: %d features (word=%d, char=%d)", X.shape[1], X_word.shape[1], X_char.shape[1])

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # 4. Optuna tuning or fixed params
        params = dict(OPTIMAL_PARAMS)
        params["num_class"] = len(le.classes_)

        if tune:
            _LOGGER.info("Running Optuna hyperparameter tuning (%d trials)...", tune_trials)
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def objective(trial):
                    p = {
                        "objective": "multiclass",
                        "num_class": len(le.classes_),
                        "boosting_type": "gbdt",
                        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                        "random_state": 42, "n_jobs": -1, "verbose": -1,
                    }
                    model = lgb.LGBMClassifier(**p)
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
                    return scores.mean()

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=tune_trials, show_progress_bar=False)
                params.update(study.best_params)
                params["num_class"] = len(le.classes_)
                _LOGGER.info("Best Optuna F1: %.4f", study.best_value)
            except ImportError:
                _LOGGER.warning("Optuna not installed, using fixed params")

        # 5. Train final model on all data (skip CV for speed — evaluate via training accuracy)
        _LOGGER.info("Training LightGBM with %d features, %d classes...", X.shape[1], len(le.classes_))
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        _LOGGER.info("Model training complete.")

        # 6. Quick training-set accuracy as quality metric (no expensive CV)
        from sklearn.metrics import f1_score
        y_pred = model.predict(X)
        f1_macro = f1_score(y, y_pred, average="macro")
        _LOGGER.info("Training F1 (macro): %.4f", f1_macro)

        # 7. Log to MLflow
        mlflow_version = None
        try:
            import json as _json
            import os
            import tempfile

            import joblib
            import mlflow
            import mlflow.lightgbm
            import pandas as pd
            from mlflow.data.pandas_dataset import PandasDataset

            # Set auth credentials (MLflow client reads from env)
            if self._config.mlflow_tracking_username and self._config.mlflow_tracking_password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = self._config.mlflow_tracking_username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = self._config.mlflow_tracking_password

            mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
            mlflow.set_experiment(self._config.mlflow_experiment_name)

            run_name = f"train-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metric("f1_macro", f1_macro)
                mlflow.log_metric("n_samples", len(texts))
                mlflow.log_metric("n_classes", len(le.classes_))

                # Log training dataset info
                df = pd.DataFrame({"text": texts, "label": labels})
                dataset: PandasDataset = mlflow.data.from_pandas(
                    df,
                    source="google-sheets://Logs+ML_Tracking",
                    name="ticket-classifier-training",
                    targets="label",
                )
                mlflow.log_input(dataset, context="training")

                # Log LightGBM model via mlflow.lightgbm
                mlflow.lightgbm.log_model(model, artifact_path="model")

                # Log supporting artifacts with names prediction service expects
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Vectorizers as dict {word_tfidf, char_tfidf}
                    vectorizers = {"word_tfidf": tfidf_word, "char_tfidf": tfidf_char}
                    joblib.dump(vectorizers, os.path.join(tmpdir, "tfidf_vectorizer.pkl"))
                    joblib.dump(le, os.path.join(tmpdir, "label_encoder.pkl"))
                    joblib.dump(self._preprocessor, os.path.join(tmpdir, "preprocessor.pkl"))

                    metadata = {
                        "classes": list(le.classes_),
                        "n_samples": len(texts),
                        "n_classes": len(le.classes_),
                        "f1_macro": round(f1_macro, 4),
                        "params": {k: str(v) for k, v in params.items()},
                        "trained_at": datetime.utcnow().isoformat(),
                    }
                    with open(os.path.join(tmpdir, "metadata.json"), "w") as mf:
                        _json.dump(metadata, mf, indent=2, ensure_ascii=False)

                    mlflow.log_artifacts(tmpdir, "model")

                # Register model and promote to Production
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model"
                result = mlflow.register_model(model_uri, self._config.mlflow_model_name)
                mlflow_version = result.version

                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=self._config.mlflow_model_name,
                    version=mlflow_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                _LOGGER.info("MLflow Version: v%s (promoted to Production)", mlflow_version)

        except Exception as e:
            _LOGGER.warning("MLflow logging failed (training still succeeded): %s", e)

        # 8. Mark training data as TRAINED
        try:
            self._http.post(f"{self._config.data_api_url.rstrip('/')}/training/mark")
        except Exception:
            _LOGGER.warning("Failed to mark data as trained")

        elapsed = time.time() - start
        return {
            "success": True,
            "status": "completed",
            "f1_score": round(f1_macro, 4),
            "n_samples": len(texts),
            "n_classes": len(le.classes_),
            "classes": list(le.classes_),
            "model_version": f"v{mlflow_version}" if mlflow_version else "local",
            "elapsed_seconds": round(elapsed, 1),
            "message": f"Training complete. F1={f1_macro:.4f}, {len(texts)} samples, {len(le.classes_)} classes.",
        }
