"""
Model loading service with lazy initialization and optimization.
Handles model caching, HuggingFace downloads, and GPU/CPU resource management.
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModelLoader:
    """Lazy-loading model manager with caching."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_name = None
        self.feature_names: Optional[List[str]] = None
        self.input_dim = None
        self.run_dir = None
        self.test_metrics: Dict[str, float] = {}
        self.is_loaded = False
        self._loading = False  # Prevent concurrent loads

    def get_models_path(self) -> str:
        """Get models directory path."""
        env_path = os.getenv("MODELS_PATH")
        if env_path and os.path.isdir(env_path):
            return env_path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(repo_root, "models")

    def _download_from_hf(self, filename: str, subfolder: Optional[str] = None) -> str:
        """Download artifact from Hugging Face Hub."""
        try:
            from huggingface_hub import hf_hub_download

            hf_repo = os.getenv("HF_MODEL_REPO", "PedroGF45/lol-draft-predictor")
            hf_rev = os.getenv("HF_MODEL_REV", "main")
            hf_token = os.getenv("HF_TOKEN")
            hf_cache = os.getenv("HF_CACHE_DIR")

            logger.info(f"Downloading {filename} from {hf_repo}@{hf_rev}")
            if not hf_token:
                logger.warning("HF_TOKEN not set - attempting to download public models")

            return hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                subfolder=subfolder,
                revision=hf_rev,
                token=hf_token,
                cache_dir=hf_cache,
            )
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}. Make sure HF_TOKEN is set for private repos")
            raise

    async def load_best_model(self, model_bucket: str = "DeepLearningClassifier"):
        """Load the best model with lazy initialization."""
        if self.is_loaded or self._loading:
            return

        self._loading = True
        try:
            use_hf = os.getenv("HF_MODEL_REPO") is not None

            if use_hf:
                await self._load_from_hf(model_bucket)
            else:
                await self._load_from_local(model_bucket)

            self.is_loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
        finally:
            self._loading = False

    async def _load_from_hf(self, model_bucket: str):
        """Load model from Hugging Face."""
        model_path = self._download_from_hf("model.pkl")
        info_path = self._download_from_hf("info.json")
        metrics_path = self._download_from_hf("metrics.json")
        prep_path = self._download_from_hf("preprocessor.pkl")

        self.model_name = model_bucket
        self.run_dir = os.path.dirname(model_path)

        # Load model - try pickle first, fallback to joblib
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            logger.warning(f"Pickle load failed, trying joblib: {e}")
            try:
                self.model = joblib.load(model_path)
            except Exception as e2:
                logger.error(f"Both pickle and joblib failed for {model_path}: {e2}")
                raise RuntimeError(f"Failed to load model file: {e2}") from e2

        # Load preprocessor - try pickle first, fallback to joblib
        try:
            with open(prep_path, "rb") as f:
                self.preprocessor = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError, FileNotFoundError) as e:
            logger.warning(f"Preprocessor pickle load failed: {e}")
            try:
                if os.path.exists(prep_path):
                    self.preprocessor = joblib.load(prep_path)
                else:
                    logger.warning("Preprocessor file not found, continuing without it")
                    self.preprocessor = None
            except Exception as e2:
                logger.warning(f"Preprocessor loading skipped: {e2}")
                self.preprocessor = None

        # Load metadata
        try:
            with open(info_path) as f:
                info = json.load(f)
                self.feature_names = info.get("feature_names")
                self.input_dim = info.get("input_dim")
        except Exception as e:
            logger.warning(f"Could not load info.json: {e}")

        # Load metrics
        try:
            with open(metrics_path) as f:
                self.test_metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metrics.json: {e}")
            self.test_metrics = {}

    async def _load_from_local(self, model_bucket: str):
        """Load model from local filesystem."""
        models_path = self.get_models_path()
        model_dir = os.path.join(models_path, model_bucket)

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        model_path = os.path.join(model_dir, "model.pkl")
        info_path = os.path.join(model_dir, "info.json")
        metrics_path = os.path.join(model_dir, "metrics.json")
        prep_path = os.path.join(model_dir, "preprocessor.pkl")

        # Load model - try joblib first, then pickle
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            logger.warning(f"Joblib load failed, trying pickle: {e}")
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model from {model_path}: {e2}") from e2

        self.model_name = model_bucket
        self.run_dir = model_dir

        # Load preprocessor if exists
        if os.path.exists(prep_path):
            try:
                self.preprocessor = joblib.load(prep_path)
            except Exception as e:
                logger.warning(f"Could not load preprocessor with joblib: {e}")
                try:
                    with open(prep_path, "rb") as f:
                        self.preprocessor = pickle.load(f)
                except Exception as e2:
                    logger.warning(f"Preprocessor load skipped: {e2}")
                    self.preprocessor = None

        # Load metadata
        if os.path.exists(info_path):
            try:
                with open(info_path) as f:
                    info = json.load(f)
                    self.feature_names = info.get("feature_names")
                    self.input_dim = info.get("input_dim")
            except Exception as e:
                logger.warning(f"Could not load info.json: {e}")

        # Load metrics
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    self.test_metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metrics.json: {e}")
                self.test_metrics = {}

    def predict(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Make prediction using loaded model."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        class_index = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        return int(class_index), probabilities

    def predict_batch(self, features_list: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        """Make batch predictions."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        results = []
        for features in features_list:
            class_index, probs = self.predict(features)
            results.append((class_index, probs))
        return results

    async def unload(self):
        """Unload model from memory."""
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        logger.info("Model unloaded")
