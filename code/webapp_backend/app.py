import os
import sys
import json
import joblib
import numpy as np
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load .env file from repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Ensure workspace 'code' folder is importable for model classes
CODE_ROOT = os.path.join(REPO_ROOT, "code")
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

# Ensure model classes are importable for pickle loading
try:
    from modeling.deep_learner import DeepLearningClassifier  # noqa: F401
except Exception:
    # If torch isn't available, DL models won't load; handled at runtime
    DeepLearningClassifier = None  # type: ignore

# Import pipeline components for match processing
from data_extraction.requester import Requester
from data_extraction.match_fetcher import MatchFetcher
from data_preparation.data_handler import DataHandler
from data_preparation.data_cleaner import DataCleaner
from feature_engineering.feature_engineer import FeatureEngineer
from helpers.parquet_handler import ParquetHandler
from helpers.master_data_registry import MasterDataRegistry
import pandas as pd
import logging

app = FastAPI(title="LoL Draft Predictor API", version="0.1.0")

class PredictRequest(BaseModel):
    # Either provide ordered list of values or feature dict
    features: Optional[List[float]] = None
    feature_map: Optional[Dict[str, float]] = None

class PredictResponse(BaseModel):
    model_name: str
    class_index: int
    class_probabilities: List[float]
    model_config = {"protected_namespaces": ()}

class ModelInfoResponse(BaseModel):
    model_name: str
    run_dir: str
    metrics: Dict[str, float]
    feature_names: Optional[List[str]]
    input_dim: int
    model_config = {"protected_namespaces": ()}

class MatchPredictRequest(BaseModel):
    match_id: str

class MatchPredictResponse(BaseModel):
    match_id: str
    team_red_win_probability: float
    team_blue_win_probability: float
    predicted_winner: str  # "red" or "blue"

# Global cached model and metadata
_MODEL = None
_PREPROCESSOR = None
_MODEL_NAME = None
_FEATURE_NAMES: Optional[List[str]] = None
_INPUT_DIM = None
_RUN_DIR = None
_TEST_METRICS: Dict[str, float] = {}

# Global pipeline components
_REQUESTER = None
_MATCH_FETCHER = None
_PARQUET_HANDLER = None
_FEATURE_ENGINEER = None
_LOGGER = None
_RANDOM_SEED = 42


def get_models_path() -> str:
    # Prefer env var if set; otherwise use repo-root/models
    env_path = os.getenv("MODELS_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    return os.path.join(repo_root, "models")


def load_best_model(model_bucket: str = "DeepLearningClassifier"):
    global _MODEL, _PREPROCESSOR, _MODEL_NAME, _FEATURE_NAMES, _INPUT_DIM, _RUN_DIR, _TEST_METRICS

    models_path = get_models_path()
    best_json = os.path.join(models_path, model_bucket, "best.json")
    if not os.path.exists(best_json):
        raise FileNotFoundError(f"best.json not found at {best_json}. Train and save a model first.")

    with open(best_json, "r", encoding="utf-8") as f:
        best = json.load(f)

    run_dir_rel = best.get("run_dir")
    run_dir = os.path.join(models_path, run_dir_rel) if run_dir_rel else None
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError("Run directory for best model not found.")

    model_rel = best.get("model_path") or "model.pkl"
    model_path = os.path.join(models_path, model_rel) if os.path.isabs(model_rel) else os.path.join(models_path, model_rel)
    if not os.path.exists(model_path):
        # fallback to run_dir/model.pkl
        model_path = os.path.join(run_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")

    # Load model - try joblib first (for DL models saved via joblib), then torch
    try:
        _MODEL = joblib.load(model_path)
        # If it's a DeepLearningModel, move to CUDA
        if hasattr(_MODEL, "to") and callable(getattr(_MODEL, "to")):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("GPU is required but torch.cuda.is_available() is False. Please install CUDA-enabled PyTorch and ensure a CUDA GPU is available.")
            _MODEL.to(torch.device("cuda"))
            if not hasattr(_MODEL, "device"):
                setattr(_MODEL, "device", torch.device("cuda"))
    except Exception as joblib_err:
        # If joblib fails, try torch.load (for raw state dicts)
        try:
            import torch
            import torch.nn as nn
            if not torch.cuda.is_available():
                raise RuntimeError("GPU is required but torch.cuda.is_available() is False. Please install CUDA-enabled PyTorch and ensure a CUDA GPU is available.")

            obj = torch.load(model_path, map_location=torch.device("cuda"), weights_only=False)
            if isinstance(obj, nn.Module):
                _MODEL = obj.to(torch.device("cuda"))
                if not hasattr(_MODEL, "device"):
                    setattr(_MODEL, "device", torch.device("cuda"))
            elif isinstance(obj, dict) and DeepLearningClassifier is not None:
                # Assume binary classification for team1_win
                num_classes = 2
                _MODEL = DeepLearningClassifier(input_dim=int(_INPUT_DIM or 0), num_classes=num_classes, require_cuda=True)  # type: ignore
                _MODEL.load_state_dict(obj)
            else:
                raise joblib_err
        except Exception:
            raise joblib_err

    # Try load info.json for feature names and input_dim
    info_path = os.path.join(run_dir, "info.json")
    _FEATURE_NAMES = None
    _INPUT_DIM = None
    _MODEL_NAME = model_bucket
    _RUN_DIR = run_dir
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        _FEATURE_NAMES = info.get("feature_names")
        _INPUT_DIM = info.get("input_dim")
        artifacts = info.get("artifacts", {})
        prep_rel = artifacts.get("preprocessor")
        if prep_rel:
            prep_path = os.path.join(models_path, prep_rel) if not os.path.isabs(prep_rel) else prep_rel
            if os.path.exists(prep_path):
                _PREPROCESSOR = joblib.load(prep_path)
        metrics_path_rel = artifacts.get("metrics")
        if metrics_path_rel:
            metrics_path = os.path.join(models_path, metrics_path_rel) if not os.path.isabs(metrics_path_rel) else metrics_path_rel
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                _TEST_METRICS = metrics.get("test_metrics", {})

    # sanity
    if _INPUT_DIM is None:
        # derive from model if possible (DL has attribute input_dim)
        _INPUT_DIM = getattr(_MODEL, "input_dim", None)
    if _INPUT_DIM is None:
        raise RuntimeError("Unable to determine model input_dim for inference.")


def load_best_overall_model():
    """Load the globally best model recorded during training (best_overall.json)."""
    models_path = get_models_path()
    overall_json = os.path.join(models_path, "best_overall.json")
    if not os.path.exists(overall_json):
        raise FileNotFoundError(
            f"best_overall.json not found at {overall_json}. Train models to generate it, or load per-bucket best.json."
        )

    with open(overall_json, "r", encoding="utf-8") as f:
        best = json.load(f)

    # Reuse load_best_model path by deriving bucket from run_dir if possible
    run_dir_rel = best.get("run_dir")
    if not run_dir_rel:
        raise RuntimeError("Invalid best_overall.json: missing run_dir")

    # Expect run_dir like "<ModelName>/<timestamp>"
    parts = run_dir_rel.replace("\\", "/").split("/")
    if not parts or len(parts) < 2:
        raise RuntimeError("Invalid run_dir format in best_overall.json")
    model_bucket = parts[0]

    # Delegate to per-bucket loader for consistent artifact handling
    return load_best_model(model_bucket)


@app.on_event("startup")
def startup_event():
    global _REQUESTER, _MATCH_FETCHER, _PARQUET_HANDLER, _FEATURE_ENGINEER, _LOGGER
    
    # Setup logger
    _LOGGER = logging.getLogger("webapp_backend")
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        _LOGGER.addHandler(handler)
        _LOGGER.setLevel(logging.INFO)
    
    # Load model at startup, but don't crash server if loading fails
    try:
        # Prefer globally best model when available
        try:
            load_best_overall_model()
        except Exception as _:
            # Fallback to DeepLearningClassifier bucket
            load_best_model("DeepLearningClassifier")
    except Exception as e:
        _LOGGER.warning(f"Model load deferred: {e}")
    
    # Initialize pipeline components
    try:
        riot_api_key = os.getenv("RIOT_API_KEY")
        if not riot_api_key:
            _LOGGER.warning("RIOT_API_KEY not set. Match prediction will not work.")
        else:
            region_v4 = "euw1"
            region_v5 = "europe"
            base_url_v4 = f'https://{region_v4}.api.riotgames.com'
            base_url_v5 = f'https://{region_v5}.api.riotgames.com'
            headers = {"X-Riot-Token": riot_api_key}
            
            _REQUESTER = Requester(base_url_v4=base_url_v4, base_url_v5=base_url_v5, headers=headers, logger=_LOGGER)
            _PARQUET_HANDLER = ParquetHandler(logger=_LOGGER, random_state=_RANDOM_SEED)
            # Resolve a valid target directory for MatchFetcher (required by its init)
            df_target = os.getenv("PREPROCESSED_DATA_PATH")
            if not df_target:
                df_target = os.path.join(REPO_ROOT, "data", "preprocessed")
            _MATCH_FETCHER = MatchFetcher(
                requester=_REQUESTER,
                logger=_LOGGER,
                parquet_handler=_PARQUET_HANDLER,
                dataframe_target_path=df_target,
                checkpoint_loading_path=None,
                load_percentage=100,
                random_state=_RANDOM_SEED,
                master_registry=None,  # No registry needed for single match
                max_workers=1
            )
            _FEATURE_ENGINEER = FeatureEngineer(logger=_LOGGER, parquet_handler=_PARQUET_HANDLER, random_state=_RANDOM_SEED)
            _LOGGER.info("Pipeline components initialized")
    except Exception as e:
        _LOGGER.error(f"Failed to initialize pipeline: {e}")

    # Mount static
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name=_MODEL_NAME or "unknown",
        run_dir=_RUN_DIR or "",
        metrics=_TEST_METRICS or {},
        feature_names=_FEATURE_NAMES,
        input_dim=int(_INPUT_DIM or 0),
    )


def _prepare_features(req: PredictRequest) -> np.ndarray:
    # Determine input vector
    if req.feature_map:
        if not _FEATURE_NAMES:
            raise HTTPException(status_code=400, detail="Model does not expose feature_names. Provide ordered 'features' list instead.")
        try:
            vec = [float(req.feature_map[name]) for name in _FEATURE_NAMES]
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Missing feature: {e.args[0]}")
        arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    elif req.features:
        arr = np.asarray(req.features, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != int(_INPUT_DIM):
            raise HTTPException(status_code=400, detail=f"Expected {_INPUT_DIM} features, got {arr.shape[1]}")
    else:
        raise HTTPException(status_code=400, detail="Provide 'features' (ordered list) or 'feature_map' (dict)")

    # Apply preprocessor if available
    if _PREPROCESSOR is not None:
        try:
            import pandas as pd
            # Prefer passing a DataFrame with feature names when available
            df = pd.DataFrame(arr, columns=_FEATURE_NAMES) if _FEATURE_NAMES else pd.DataFrame(arr)
            arr = _PREPROCESSOR.transform(df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    return arr


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    x = _prepare_features(req)

    # Log model info for this prediction
    try:
        _LOGGER.info(
            "Predict: model=%s type=%s input_dim=%s run_dir=%s",
            _MODEL_NAME or "unknown",
            type(_MODEL).__name__ if _MODEL is not None else "None",
            str(_INPUT_DIM),
            _RUN_DIR or ""
        )
    except Exception:
        pass

    # DeepLearningClassifier inference
    if hasattr(_MODEL, "forward"):
        import torch
        _MODEL.eval()
        with torch.no_grad():
            inp = torch.FloatTensor(x).to(getattr(_MODEL, "device", "cpu"))
            logits = _MODEL.forward(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            return PredictResponse(model_name=_MODEL_NAME or "", class_index=pred_idx, class_probabilities=list(map(float, probs)))

    # sklearn classifiers
    if hasattr(_MODEL, "predict_proba"):
        probs = _MODEL.predict_proba(x)[0]
        pred_idx = int(np.argmax(probs))
        return PredictResponse(model_name=_MODEL_NAME or "", class_index=pred_idx, class_probabilities=list(map(float, probs)))

    # fallback to predict only
    pred = _MODEL.predict(x)
    return PredictResponse(model_name=_MODEL_NAME or "", class_index=int(pred[0]), class_probabilities=[])


@app.post("/predict-match", response_model=MatchPredictResponse)
def predict_match(req: MatchPredictRequest):
    """Predict match outcome from match ID by running full pipeline."""
    if not _REQUESTER or not _MATCH_FETCHER or not _FEATURE_ENGINEER:
        raise HTTPException(status_code=500, detail="Pipeline not initialized. Set RIOT_API_KEY environment variable.")
    
    if _MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Process into DataFrame (use MatchFetcher helper on match_id)
        _LOGGER.info(f"Fetching and processing match {req.match_id}")
        kpi_limit = int(os.getenv("PREDICT_KPI_LIMIT", "10"))
        match_record, player_history_records = _MATCH_FETCHER._process_single_match(req.match_id, match_limit_per_player=kpi_limit)
        if not match_record:
            raise HTTPException(status_code=400, detail="Failed to process match data")
        
        # Create DataFrames from the processed records
        match_df = pd.DataFrame([match_record])
        player_history_df = pd.DataFrame(player_history_records)
        
        # Create temporary DataHandler for this match
        temp_handler = DataHandler(
            logger=_LOGGER,
            parquet_handler=_PARQUET_HANDLER,
            target_feature="team1_win",
            random_state=_RANDOM_SEED,
            master_registry=None
        )
        
        # Join match and player data (similar to DataHandler.join_match_and_player_data logic)
        # This creates aggregated player stats per match
        from data_preparation.data_handler import DataHandler as DH
        
        # Use the join method to combine match and player history data
        # We'll save to temporary paths and load them back
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            match_tmp = os.path.join(tmpdir, "match.parquet")
            player_tmp = os.path.join(tmpdir, "player.parquet")
            _PARQUET_HANDLER.write_parquet(match_df, match_tmp)
            _PARQUET_HANDLER.write_parquet(player_history_df, player_tmp)
            
            # Join the data
            temp_handler.join_match_and_player_data(
                match_parquet_path=match_tmp,
                player_parquet_path=player_tmp
            )
        
        # Set as "test" data (we're predicting, not training)
        combined_df = temp_handler.get_combined_dataframe()
        
        # Extract features that match the model's training features
        # Generate the same derived features used during training
        try:
            X = _FEATURE_ENGINEER.generate_new_features(combined_df.copy())
        except Exception:
            # Fall back to raw combined data if feature generation fails
            X = combined_df.copy()
        
        # Drop non-feature columns
        drop_cols = [col for col in ["team1_win", "match_id", "game_version"] if col in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
        
        _LOGGER.info(f"Available features after joining: {X.shape[1]} columns")
        
        # Apply preprocessor first if available (e.g., to generate PCA components)
        if _PREPROCESSOR is not None:
            try:
                _LOGGER.info(f"Applying saved preprocessor to {X.shape[1]} features")
                X_pre = _PREPROCESSOR.transform(X if isinstance(X, pd.DataFrame) else pd.DataFrame(X))
            except Exception as e:
                _LOGGER.error(f"Preprocessor transform failed: {e}")
                X_pre = X
        else:
            X_pre = X

        # If model specifies required feature names, align after preprocessing
        if _FEATURE_NAMES:
            if isinstance(X_pre, pd.DataFrame):
                available = set(X_pre.columns)
                required = set(_FEATURE_NAMES)
                missing = required - available
                extra = available - required

                # Concise summary of alignment
                try:
                    _LOGGER.info(
                        "Feature alignment: required=%d, present=%d, missing=%d (zero-filled), extra=%d (ignored)",
                        len(_FEATURE_NAMES), len(available), len(missing), len(extra)
                    )
                except Exception:
                    pass

                # Debug sample of missing names (optional)
                try:
                    import logging as _py_logging
                    if missing and _LOGGER.isEnabledFor(_py_logging.DEBUG):
                        _LOGGER.debug("Missing features sample: %s", list(missing)[:5])
                except Exception:
                    pass

                # Reindex to fill missing and drop extras in one step
                try:
                    X_pre = X_pre.reindex(columns=_FEATURE_NAMES, fill_value=0.0)
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Cannot predict: model expects {len(_FEATURE_NAMES)} features but only {len(available)} available. "
                            f"This match may be missing data or model was trained on different feature pipeline."
                        ),
                    )
                x = X_pre.values.astype(np.float32)
            else:
                # If preprocessing returned ndarray, ensure shape matches
                x = X_pre
                if x.ndim != 2 or x.shape[1] != int(_INPUT_DIM):
                    raise HTTPException(status_code=400, detail="Preprocessed features do not match model input dimension")
        else:
            # No feature list declared; use numeric matrix
            x = X_pre.values if isinstance(X_pre, pd.DataFrame) else X_pre
        
        _LOGGER.info(f"Input shape for model: {x.shape}")
        
        # Debug: Log feature vector stats before prediction
        try:
            if isinstance(x, np.ndarray):
                _LOGGER.info(
                    "Feature vector stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f, zeros=%d/%d",
                    float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x)),
                    int(np.sum(x == 0)), x.size
                )
                # Log top 5 largest/smallest values
                top_vals = np.argsort(x.flatten())[-5:][::-1]
                bottom_vals = np.argsort(x.flatten())[:5]
                _LOGGER.info(
                    "Top 5 values (idx, val): %s | Bottom 5 (idx, val): %s",
                    [(int(i), float(x.flatten()[i])) for i in top_vals],
                    [(int(i), float(x.flatten()[i])) for i in bottom_vals]
                )
        except Exception as e:
            _LOGGER.warning(f"Failed to log feature stats: {e}")
        
        # Log model info for this match prediction
        try:
            _LOGGER.info(
                "PredictMatch: model=%s type=%s input_dim=%s run_dir=%s",
                _MODEL_NAME or "unknown",
                type(_MODEL).__name__ if _MODEL is not None else "None",
                str(_INPUT_DIM),
                _RUN_DIR or ""
            )
        except Exception:
            pass
        
        # Predict
        if hasattr(_MODEL, "forward"):
            import torch
            _MODEL.eval()
            with torch.no_grad():
                inp = torch.FloatTensor(x).to(getattr(_MODEL, "device", "cpu"))
                logits = _MODEL.forward(inp)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        elif hasattr(_MODEL, "predict_proba"):
            probs = _MODEL.predict_proba(x)[0]
        else:
            pred = _MODEL.predict(x)
            probs = np.array([1.0 - pred[0], pred[0]]) if pred[0] in [0, 1] else np.array([0.5, 0.5])

        # Label mapping:
        # - 1 = team1_win (team1 is Blue side, 100)
        # - 0 = team1_loss (team2 is Red side, 200)
        # Respect model.classes_ order when available
        idx_blue = None
        idx_red = None
        try:
            if hasattr(_MODEL, "classes_"):
                classes = list(getattr(_MODEL, "classes_", []))
                _LOGGER.info("Model classes_: %s", classes)
                # Find indices where class 1 (True/team1_win) and 0 (False/team1_loss) map in probs
                if True in classes:
                    idx_blue = classes.index(True)
                elif 1 in classes:
                    idx_blue = classes.index(1)
                if False in classes:
                    idx_red = classes.index(False)
                elif 0 in classes:
                    idx_red = classes.index(0)
                _LOGGER.info("Class index mapping: idx_blue=%s (True/1), idx_red=%s (False/0)", idx_blue, idx_red)
        except Exception as e:
            _LOGGER.warning(f"Failed to resolve class indices: {e}")
        
        # Fallback if class resolution failed
        if idx_blue is None:
            idx_blue = 1
        if idx_red is None:
            idx_red = 0
        
        team_blue_prob = float(probs[idx_blue]) if idx_blue < len(probs) else 0.5
        team_red_prob = float(probs[idx_red]) if idx_red < len(probs) else 0.5
        
        _LOGGER.info(
            "Probabilities: red_prob=%.4f (idx %d), blue_prob=%.4f (idx %d)",
            team_red_prob, idx_red, team_blue_prob, idx_blue
        )
        
        predicted_winner = "red" if team_red_prob > team_blue_prob else "blue"
        
        return MatchPredictResponse(
            match_id=req.match_id,
            team_red_win_probability=team_red_prob,
            team_blue_win_probability=team_blue_prob,
            predicted_winner=predicted_winner
        )
        
    except HTTPException:
        raise
    except Exception as e:
        _LOGGER.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

