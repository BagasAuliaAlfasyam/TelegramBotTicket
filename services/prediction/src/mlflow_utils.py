"""
MLflow Utilities (microservice version)
========================================
Same logic as original src/ml/mlflow_utils.py, but standalone.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

_LOGGER = logging.getLogger(__name__)

_mlflow = None
_mlflow_lightgbm = None


def _import_mlflow():
    global _mlflow, _mlflow_lightgbm
    if _mlflow is None:
        import mlflow
        import mlflow.lightgbm
        _mlflow = mlflow
        _mlflow_lightgbm = mlflow.lightgbm
    return _mlflow, _mlflow_lightgbm


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    model_name: str
    s3_endpoint_url: str
    bucket_name: str
    tracking_username: str = ""
    tracking_password: str = ""


class MLflowManager:
    """MLflow integration for model loading and registry."""

    def __init__(self, config: MLflowConfig):
        self.config = config
        self._initialized = False
        self._current_run = None
        self._experiment_id = None

    @property
    def is_enabled(self) -> bool:
        return bool(self.config.tracking_uri)

    def init(self) -> bool:
        if not self.is_enabled:
            return False
        if self._initialized:
            return True

        try:
            mlflow, _ = _import_mlflow()

            if self.config.s3_endpoint_url:
                os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config.s3_endpoint_url
                os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

            if self.config.tracking_username and self.config.tracking_password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.tracking_username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.tracking_password

            mlflow.set_tracking_uri(self.config.tracking_uri)

            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    self.config.experiment_name,
                    artifact_location=f"s3://{self.config.bucket_name}/artifacts/{self.config.experiment_name}"
                )
            else:
                self._experiment_id = experiment.experiment_id

            mlflow.set_experiment(self.config.experiment_name)
            self._initialized = True
            return True

        except Exception as e:
            _LOGGER.error("Failed to initialize MLflow: %s", e)
            return False

    @contextmanager
    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None):
        if not self._initialized:
            yield None
            return

        mlflow, _ = _import_mlflow()
        try:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self._current_run = run
            yield run
        finally:
            mlflow.end_run()
            self._current_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._current_run:
            return
        mlflow, _ = _import_mlflow()
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._current_run:
            return
        mlflow, _ = _import_mlflow()
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, vectorizers, label_encoder, preprocessor, metadata, artifact_path="model"):
        if not self._current_run:
            return None

        mlflow, mlflow_lgb = _import_mlflow()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                joblib.dump(vectorizers, tmppath / "tfidf_vectorizer.pkl")
                joblib.dump(label_encoder, tmppath / "label_encoder.pkl")
                joblib.dump(preprocessor, tmppath / "preprocessor.pkl")

                with open(tmppath / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

                model_info = mlflow_lgb.log_model(model, artifact_path=artifact_path)
                mlflow.log_artifact(str(tmppath / "tfidf_vectorizer.pkl"), artifact_path)
                mlflow.log_artifact(str(tmppath / "label_encoder.pkl"), artifact_path)
                mlflow.log_artifact(str(tmppath / "preprocessor.pkl"), artifact_path)
                mlflow.log_artifact(str(tmppath / "metadata.json"), artifact_path)

                return model_info.model_uri
        except Exception as e:
            _LOGGER.error("Failed to log model: %s", e)
            return None

    def register_model(self, model_uri=None, stage="Staging", description=""):
        if not self.is_enabled:
            return None

        mlflow, _ = _import_mlflow()

        try:
            if model_uri is None and self._current_run:
                model_uri = f"runs:/{self._current_run.info.run_id}/model"
            if not model_uri:
                return None

            result = mlflow.register_model(model_uri=model_uri, name=self.config.model_name)
            version = result.version

            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=self.config.model_name, version=version, stage=stage,
                archive_existing_versions=(stage == "Production")
            )

            if description:
                client.update_model_version(name=self.config.model_name, version=version, description=description)

            return version
        except Exception as e:
            _LOGGER.error("Failed to register model: %s", e)
            return None

    def load_production_model(self):
        return self.load_model_by_stage("Production")

    def load_model_by_stage(self, stage: str = "Production"):
        if not self.is_enabled:
            return None

        mlflow, mlflow_lgb = _import_mlflow()

        try:
            client = mlflow.tracking.MlflowClient()

            # Try to find model by stage first
            version = None
            try:
                versions = client.get_latest_versions(self.config.model_name, stages=[stage])
                if versions:
                    version = versions[0]
            except Exception:
                pass

            # Fallback: get latest version regardless of stage
            if version is None:
                _LOGGER.info("No model in stage '%s', trying latest version...", stage)
                try:
                    all_versions = client.get_latest_versions(self.config.model_name, stages=["None"])
                    if all_versions:
                        version = all_versions[0]
                except Exception:
                    try:
                        from mlflow.entities.model_registry import ModelVersion
                        search = client.search_model_versions(f"name='{self.config.model_name}'")
                        if search:
                            version = sorted(search, key=lambda v: int(v.version), reverse=True)[0]
                    except Exception:
                        pass

            if version is None:
                _LOGGER.error("No model versions found for '%s'", self.config.model_name)
                return None

            run_id = version.run_id
            _LOGGER.info("Loading model version=%s, stage=%s, run_id=%s", version.version, version.current_stage, run_id)

            # Download artifacts
            artifact_path = client.download_artifacts(run_id, "model")
            artifact_dir = Path(artifact_path)

            # Load LightGBM model — try MLflow format first, fallback to joblib
            model = None
            try:
                model = mlflow_lgb.load_model(f"models:/{self.config.model_name}/{version.current_stage}")
            except Exception:
                try:
                    model = mlflow_lgb.load_model(f"runs:/{run_id}/model")
                except Exception:
                    # Try raw joblib
                    model_file = artifact_dir / "model.joblib"
                    if model_file.exists():
                        model = joblib.load(model_file)

            if model is None:
                _LOGGER.error("Failed to load LightGBM model from any source")
                return None

            # Load vectorizers
            vectorizers = None
            vec_path = artifact_dir / "tfidf_vectorizer.pkl"
            if vec_path.exists():
                vectorizers = joblib.load(vec_path)
            else:
                # Fallback: load word/char separately
                word_path = artifact_dir / "tfidf_word.joblib"
                char_path = artifact_dir / "tfidf_char.joblib"
                if word_path.exists() and char_path.exists():
                    vectorizers = {"word_tfidf": joblib.load(word_path), "char_tfidf": joblib.load(char_path)}

            # Load label encoder
            encoder = None
            for name in ["label_encoder.pkl", "label_encoder.joblib"]:
                enc_path = artifact_dir / name
                if enc_path.exists():
                    encoder = joblib.load(enc_path)
                    break

            # Load preprocessor (optional)
            preprocessor = None
            for name in ["preprocessor.pkl", "preprocessor.joblib"]:
                prep_path = artifact_dir / name
                if prep_path.exists():
                    preprocessor = joblib.load(prep_path)
                    break

            # Load metadata (optional)
            metadata = {}
            meta_path = artifact_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)

            if vectorizers is None or encoder is None:
                _LOGGER.error("Missing required artifacts (vectorizers=%s, encoder=%s)", vectorizers is not None, encoder is not None)
                return None

            return {
                "model": model, "vectorizers": vectorizers, "encoder": encoder,
                "preprocessor": preprocessor, "metadata": metadata,
                "version": version.version, "run_id": run_id, "stage": version.current_stage,
            }
        except Exception as e:
            _LOGGER.error("Failed to load model from stage %s: %s", stage, e)
            return None

    def get_status(self) -> dict[str, Any]:
        status = {
            "enabled": self.is_enabled,
            "initialized": self._initialized,
            "tracking_uri": self.config.tracking_uri if self.is_enabled else None,
            "model_name": self.config.model_name,
        }

        if self._initialized:
            try:
                mlflow, _ = _import_mlflow()
                client = mlflow.tracking.MlflowClient()
                prod_versions = client.get_latest_versions(self.config.model_name, stages=["Production"])
                if prod_versions:
                    v = prod_versions[0]
                    status["production_model"] = {
                        "version": v.version, "run_id": v.run_id,
                        "created_at": v.creation_timestamp,
                    }
            except Exception:
                pass

        return status

    def get_model_versions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get registered model versions with stage info."""
        if not self._initialized:
            return []
        try:
            mlflow, _ = _import_mlflow()
            client = mlflow.tracking.MlflowClient()
            search = client.search_model_versions(f"name='{self.config.model_name}'")
            versions = sorted(search, key=lambda v: int(v.version), reverse=True)[:limit]
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "created_at": v.creation_timestamp,
                    "description": v.description or "",
                }
                for v in versions
            ]
        except Exception as e:
            _LOGGER.error("Failed to get model versions: %s", e)
            return []

    def transition_model_stage(self, version: str, stage: str = "Production") -> dict[str, Any]:
        """Promote/transition a model version to a new stage."""
        if not self._initialized:
            return {"success": False, "message": "MLflow not initialized"}
        try:
            mlflow, _ = _import_mlflow()
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=self.config.model_name,
                version=version,
                stage=stage,
                archive_existing_versions=(stage == "Production"),
            )
            return {"success": True, "version": version, "stage": stage, "message": f"Version {version} → {stage}"}
        except Exception as e:
            _LOGGER.error("Failed to transition model %s to %s: %s", version, stage, e)
            return {"success": False, "message": str(e)}
