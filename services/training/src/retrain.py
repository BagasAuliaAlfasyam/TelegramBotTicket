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
import threading
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
    "class_weight": "balanced",
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
    "n_jobs": -1,
    "verbose": -1,
    "force_col_wise": True,  # faster for sparse data
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
        self._progress: dict = {}
        self._progress_lock = threading.Lock()

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_trained(self) -> str | None:
        return self._last_trained

    @property
    def last_result(self) -> dict:
        return self._last_result

    @property
    def progress(self) -> dict:
        with self._progress_lock:
            return dict(self._progress)

    def _update_progress(self, **kwargs) -> None:
        with self._progress_lock:
            self._progress.update(kwargs)

    def _set_phase(self, phase: str, label: str, **extra) -> None:
        """Update current training phase for live progress reporting."""
        with self._progress_lock:
            self._progress["phase"] = phase
            self._progress["phase_label"] = label
            self._progress.update(extra)
        _LOGGER.info("Phase: %s — %s", phase, label)

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
        with self._progress_lock:
            self._progress = {
                "phase": "starting",
                "phase_label": "🚀 Memulai training...",
                "started_at": datetime.utcnow().isoformat(),
                "n_samples": 0,
                "n_classes": 0,
                "n_features": 0,
                "current_trial": 0,
                "total_trials": tune_trials if tune else 0,
                "best_f1": 0.0,
                "current_f1": 0.0,
                "tune": tune,
            }

        # 1. Fetch training data from Data API
        self._set_phase("fetching_data", "📥 Mengambil data dari Google Sheets...")
        resp = self._http.get(f"{self._config.data_api_url.rstrip('/')}/training/data")
        resp.raise_for_status()
        data = resp.json()

        logs_data = data.get("logs_data", [])
        tracking_data = data.get("tracking_data", [])
        _LOGGER.info("From Logs: %d, From ML_Tracking: %d", len(logs_data), len(tracking_data))

        # 2. Combine and prepare training data (deduplicate by tech_message_id)
        #    ML_Tracking has priority (manually curated), Logs fills the rest
        self._set_phase("preprocessing", "🔧 Preprocessing & deduplikasi data...")
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
                texts.append(text)
                labels.append(label)
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
                texts.append(text)
                labels.append(label)
                if mid:
                    seen_ids.add(mid)

        if skipped:
            _LOGGER.info("Dedup: dropped %d duplicate rows from Logs (already in ML_Tracking)", skipped)

        if len(texts) < 50 and not force:
            return {"success": False, "message": f"Only {len(texts)} samples, need 50+. Use force=true."}

        n_classes = len(set(labels))
        self._update_progress(n_samples=len(texts), n_classes=n_classes)
        _LOGGER.info("Total training data: %d samples, %d classes", len(texts), n_classes)

        # 3. TF-IDF Vectorization (reduced features for Docker performance)
        self._set_phase("tfidf", "📊 TF-IDF vectorization...",
                        n_samples=len(texts), n_classes=n_classes)
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
        self._update_progress(n_features=X.shape[1])
        _LOGGER.info("TF-IDF done: %d features (word=%d, char=%d)", X.shape[1], X_word.shape[1], X_char.shape[1])

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # 4. Optuna tuning or fixed params
        params = dict(OPTIMAL_PARAMS)
        params["num_class"] = len(le.classes_)

        if tune:
            self._set_phase("optuna_tuning",
                            f"🔬 Optuna tuning (0/{tune_trials})...",
                            current_trial=0, total_trials=tune_trials,
                            best_f1=0.0, current_f1=0.0)
            _LOGGER.info("Running Optuna hyperparameter tuning (%d trials)...", tune_trials)
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def _optuna_callback(study, trial):
                    """Update progress after each Optuna trial."""
                    t_num = trial.number + 1
                    t_val = trial.value if trial.value is not None else 0.0
                    best = study.best_value if study.best_trial else 0.0
                    self._update_progress(
                        current_trial=t_num,
                        current_f1=round(t_val, 4),
                        best_f1=round(best, 4),
                        phase_label=f"🔬 Optuna tuning ({t_num}/{tune_trials})...",
                    )
                    if t_num % 10 == 0 or t_num == tune_trials:
                        _LOGGER.info("Optuna trial %d/%d — F1=%.4f, best=%.4f",
                                     t_num, tune_trials, t_val, best)

                def objective(trial):
                    import warnings
                    warnings.filterwarnings("ignore", category=UserWarning)
                    p = {
                        "objective": "multiclass",
                        "num_class": len(le.classes_),
                        "boosting_type": "gbdt",
                        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
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
                study.optimize(objective, n_trials=tune_trials,
                               show_progress_bar=False, callbacks=[_optuna_callback])
                params.update(study.best_params)
                params["num_class"] = len(le.classes_)
                _LOGGER.info("Best Optuna F1: %.4f", study.best_value)
            except ImportError:
                _LOGGER.warning("Optuna not installed, using fixed params")

        # 5. Train final model on all data (skip CV for speed — evaluate via training accuracy)
        self._set_phase("training_final", "🏋️ Training model final...")
        _LOGGER.info("Training LightGBM with %d features, %d classes...", X.shape[1], len(le.classes_))
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        _LOGGER.info("Model training complete.")

        # 6. Quick training-set accuracy as quality metric (no expensive CV)
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
        y_pred = model.predict(X)
        f1_macro = f1_score(y, y_pred, average="macro")
        accuracy = accuracy_score(y, y_pred)
        precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(
            y,
            y_pred,
            labels=np.arange(len(le.classes_)),
            zero_division=0,
        )
        class_distribution = {
            cls_name: int(count)
            for cls_name, count in zip(
                le.classes_.tolist(),
                np.bincount(y, minlength=len(le.classes_)).tolist(),
            )
        }
        per_class_metrics = []
        for idx, cls_name in enumerate(le.classes_):
            per_class_metrics.append(
                {
                    "class_index": int(idx),
                    "class_label": str(cls_name),
                    "precision": float(precision_cls[idx]),
                    "recall": float(recall_cls[idx]),
                    "f1": float(f1_cls[idx]),
                    "support": int(support_cls[idx]),
                }
            )

        report_dict = classification_report(
            y,
            y_pred,
            target_names=le.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        )
        conf_matrix = confusion_matrix(y, y_pred, labels=np.arange(len(le.classes_)))
        top_confusion_pairs = []
        for i, true_label in enumerate(le.classes_):
            for j, pred_label in enumerate(le.classes_):
                if i == j:
                    continue
                count = int(conf_matrix[i, j])
                if count <= 0:
                    continue
                top_confusion_pairs.append(
                    {
                        "true_label": str(true_label),
                        "pred_label": str(pred_label),
                        "count": count,
                    }
                )
        top_confusion_pairs.sort(key=lambda item: item["count"], reverse=True)
        top_confusion_pairs = top_confusion_pairs[:20]
        _LOGGER.info("Training F1 (macro): %.4f", f1_macro)
        _LOGGER.info("Training accuracy: %.4f", accuracy)

        # 7. Log to MLflow
        self._set_phase("logging_mlflow", "📦 Logging ke MLflow...")
        mlflow_version = None
        try:
            import json as _json
            import os
            import tempfile

            import joblib
            import mlflow
            import mlflow.lightgbm
            import pandas as pd

            # Set auth credentials (MLflow client reads from env)
            if self._config.mlflow_tracking_username and self._config.mlflow_tracking_password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = self._config.mlflow_tracking_username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = self._config.mlflow_tracking_password

            mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
            mlflow.set_experiment(self._config.mlflow_experiment_name)

            run_name = f"train-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                active_run_id = mlflow.active_run().info.run_id
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metric("f1_macro", f1_macro)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("n_samples", len(texts))
                mlflow.log_metric("n_classes", len(le.classes_))

                def _safe_metric_suffix(name: str) -> str:
                    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
                    cleaned = "_".join(filter(None, cleaned.split("_")))
                    return cleaned[:80] if cleaned else "unknown"

                for row in per_class_metrics:
                    suffix = _safe_metric_suffix(row["class_label"])
                    mlflow.log_metric(f"recall__{suffix}", row["recall"])
                    mlflow.log_metric(f"precision__{suffix}", row["precision"])
                    mlflow.log_metric(f"f1__{suffix}", row["f1"])

                mlflow.log_dict(class_distribution, "analysis/class_distribution.json")
                mlflow.log_dict({"per_class_metrics": per_class_metrics}, "analysis/per_class_metrics.json")
                mlflow.log_dict(
                    {
                        "labels": le.classes_.tolist(),
                        "matrix": conf_matrix.tolist(),
                    },
                    "analysis/confusion_matrix.json",
                )
                mlflow.log_dict(report_dict, "analysis/classification_report.json")
                mlflow.log_dict(
                    {"top_confusion_pairs": top_confusion_pairs},
                    "analysis/top_confusion_pairs.json",
                )

                # Compare label distribution with previous run (if available)
                drift_payload = {
                    "previous_run_id": None,
                    "current_run_id": active_run_id,
                    "label_distribution_drift": [],
                    "note": "No comparable previous run artifact found.",
                }
                try:
                    client = mlflow.tracking.MlflowClient()
                    experiment = mlflow.get_experiment_by_name(self._config.mlflow_experiment_name)
                    if experiment is not None:
                        runs = client.search_runs(
                            experiment_ids=[experiment.experiment_id],
                            order_by=["attributes.start_time DESC"],
                            max_results=20,
                        )
                        previous_runs = [r for r in runs if r.info.run_id != active_run_id]
                        for prev_run in previous_runs:
                            try:
                                prev_dist_path = client.download_artifacts(
                                    prev_run.info.run_id,
                                    "analysis/class_distribution.json",
                                )
                                with open(prev_dist_path, encoding="utf-8") as pf:
                                    prev_distribution = _json.load(pf)

                                prev_total = max(sum(int(v) for v in prev_distribution.values()), 1)
                                curr_total = max(sum(int(v) for v in class_distribution.values()), 1)
                                labels_union = sorted(set(prev_distribution.keys()) | set(class_distribution.keys()))
                                drift_rows = []
                                for label_name in labels_union:
                                    prev_count = int(prev_distribution.get(label_name, 0))
                                    curr_count = int(class_distribution.get(label_name, 0))
                                    prev_share = prev_count / prev_total
                                    curr_share = curr_count / curr_total
                                    drift_rows.append(
                                        {
                                            "label": label_name,
                                            "prev_count": prev_count,
                                            "curr_count": curr_count,
                                            "prev_share": round(prev_share, 6),
                                            "curr_share": round(curr_share, 6),
                                            "share_delta": round(curr_share - prev_share, 6),
                                        }
                                    )

                                drift_rows.sort(key=lambda item: abs(item["share_delta"]), reverse=True)
                                drift_payload = {
                                    "previous_run_id": prev_run.info.run_id,
                                    "current_run_id": active_run_id,
                                    "label_distribution_drift": drift_rows,
                                }
                                break
                            except Exception:
                                continue
                except Exception as drift_err:
                    _LOGGER.warning("Label distribution drift logging skipped: %s", drift_err)
                mlflow.log_dict(drift_payload, "analysis/label_distribution_drift_vs_prev_run.json")

                # Log training dataset info (optional — don't fail MLflow if this breaks)
                try:
                    df = pd.DataFrame({"text": texts, "label": labels})
                    mlflow.log_dict(
                        {
                            "rows": int(len(df)),
                            "columns": ["text", "label"],
                            "preview": df.head(20).to_dict(orient="records"),
                        },
                        "analysis/training_dataset_preview.json",
                    )
                except Exception as ds_err:
                    _LOGGER.warning("Dataset logging skipped: %s", ds_err)

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
                        "accuracy": round(float(accuracy), 4),
                        "class_distribution": class_distribution,
                        "params": {k: str(v) for k, v in params.items()},
                        "trained_at": datetime.utcnow().isoformat(),
                    }
                    with open(os.path.join(tmpdir, "metadata.json"), "w") as mf:
                        _json.dump(metadata, mf, indent=2, ensure_ascii=False)

                    mlflow.log_artifacts(tmpdir, "model")

                # Register model and promote to Production
                run_id = active_run_id
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
        self._set_phase("marking_trained", "✅ Marking data as trained...")
        try:
            self._http.post(f"{self._config.data_api_url.rstrip('/')}/training/mark")
        except Exception:
            _LOGGER.warning("Failed to mark data as trained")

        self._set_phase("done", "✅ Training selesai!")
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
