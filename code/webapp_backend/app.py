"""
Refactored FastAPI application with modular services, enhanced caching,
batch predictions, and comprehensive logging.

Key improvements:
- Separated concerns: cache, rate limiting, model loading, monitoring into services
- Enhanced Redis integration with fallback caching
- Batch prediction endpoint
- Lazy model loading
- Improved error logging and monitoring
- URL-based prediction sharing support
"""

import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env file from repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure workspace 'code' folder is importable for model classes
CODE_ROOT = os.path.join(REPO_ROOT, "code")
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

# Ensure model classes are importable for pickle loading
try:
    from modeling.deep_learner import DeepLearningClassifier  # noqa: F401
except Exception:
    DeepLearningClassifier = None  # type: ignore

import logging

from helpers.master_data_registry import MasterDataRegistry

# Import new modular services
from services.cache_service import CacheService
from services.model_loader import ModelLoader
from services.monitoring_service import MonitoringService
from services.rate_limit_service import RateLimitService

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="LoL Draft Predictor API",
    version="2.0.0",
    description="Live game detection and draft prediction with batch processing",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models
# ============================================================================


class PredictRequest(BaseModel):
    """Single prediction request."""

    features: Optional[List[float]] = None
    feature_map: Optional[Dict[str, float]] = None
    model_config = {"protected_namespaces": ()}


class PredictResponse(BaseModel):
    """Single prediction response with confidence."""

    model_name: str
    class_index: int
    class_probabilities: List[float]
    confidence: float  # Max probability
    model_config = {"protected_namespaces": ()}


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    predictions: List[PredictRequest]
    include_confidence: bool = True


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    model_name: str
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    results: List[Dict]  # Each result has class_index, probabilities, confidence
    processing_time_ms: float
    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_name: str
    run_dir: str
    metrics: Dict[str, Dict]  # Contains cv_metrics and test_metrics dicts
    feature_names: Optional[List[str]]
    input_dim: int
    model_config = {"protected_namespaces": ()}


class MatchPredictRequest(BaseModel):
    """Match prediction request."""

    match_id: str
    model_config = {"protected_namespaces": ()}


class MatchPredictResponse(BaseModel):
    """Match prediction response with confidence."""

    match_id: str
    team_red_win_probability: float
    team_blue_win_probability: float
    predicted_winner: str
    confidence: float  # Max of the two probabilities
    share_url: Optional[str] = None  # URL to share prediction
    model_config = {"protected_namespaces": ()}


class LiveGameRequest(BaseModel):
    """Live game check request."""

    game_name: str
    tag_line: str
    region: str = "euw1"
    model_config = {"protected_namespaces": ()}


class LiveGameResponse(BaseModel):
    """Live game check response."""

    has_active_game: bool
    game_id: Optional[str] = None
    team_red_win_probability: Optional[float] = None
    team_blue_win_probability: Optional[float] = None
    predicted_winner: Optional[str] = None
    confidence: Optional[float] = None
    share_url: Optional[str] = None
    error: Optional[str] = None
    model_config = {"protected_namespaces": ()}


class MetricsResponse(BaseModel):
    """Application metrics response."""

    uptime_seconds: float
    requests_total: int
    requests_errors: int
    error_rate: float
    cache_hit_rate: float
    model_predictions: int
    batch_predictions: int
    timestamp: str
    model_config = {"protected_namespaces": ()}


# ============================================================================
# Global Services
# ============================================================================

logger = logging.getLogger("webapp_backend")
monitoring = MonitoringService("lol-draft-predictor")
cache_service = CacheService()
rate_limit_service = RateLimitService()
model_loader = ModelLoader()

# ============================================================================
# Middleware
# ============================================================================


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Enhanced logging and monitoring for all requests."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    # Allow public endpoints
    if request.url.path in {"/health", "/static", "/", "/docs", "/openapi.json"}:
        return await call_next(request)

    # Rate limiting check
    endpoint_key = (
        "match"
        if request.url.path == "/predict-match"
        else "batch" if request.url.path == "/batch-predict" else "default"
    )
    if not await rate_limit_service.is_allowed(client_ip, endpoint_key):
        monitoring.log_request(request.url.path, request.method, client_ip, 429, 0, "Rate limit exceeded")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": rate_limit_service.window,
            },
        )

    # Process request
    response = await call_next(request)

    # Log request
    duration_ms = (time.time() - start_time) * 1000
    error = None if response.status_code < 400 else f"HTTP {response.status_code}"
    monitoring.log_request(request.url.path, request.method, client_ip, response.status_code, duration_ms, error)

    return response


# ============================================================================
# Event Handlers
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize services and load models."""
    # Setup monitoring
    log_file = os.getenv("LOG_FILE", os.path.join(REPO_ROOT, "logs", "app.log"))
    monitoring.setup(log_level=os.getenv("LOG_LEVEL", "INFO"), log_file=log_file)

    monitoring.info("Starting LoL Draft Predictor v2.0")

    # Initialize cache service
    await cache_service.init()

    # Initialize rate limiting
    await rate_limit_service.init()

    # Load model with lazy initialization
    try:
        await model_loader.load_best_model("DeepLearningClassifier")
        monitoring.info(f"Model loaded: {model_loader.model_name}")
    except Exception as e:
        monitoring.warning(f"Model load deferred: {e}")

    # Mount static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    monitoring.info("Startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services."""
    monitoring.info("Shutting down...")
    await cache_service.close()
    await rate_limit_service.close()
    await model_loader.unload()


# ============================================================================
# Helper Functions
# ============================================================================


def _generate_share_url(prediction_id: str) -> str:
    """Generate shareable URL for a prediction."""
    return f"/share/{prediction_id}"


def _create_prediction_id(data: dict) -> str:
    """Create unique ID for prediction sharing."""
    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _prepare_features(req: PredictRequest) -> np.ndarray:
    """Prepare feature array from request."""
    if req.feature_map:
        if not model_loader.feature_names:
            raise HTTPException(
                status_code=400,
                detail="Model does not expose feature_names. Provide ordered 'features' list instead.",
            )
        try:
            vec = [float(req.feature_map[name]) for name in model_loader.feature_names]
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Missing feature: {e.args[0]}")
        arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    elif req.features:
        arr = np.asarray(req.features, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != int(model_loader.input_dim or 0):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model_loader.input_dim} features, got {arr.shape[1]}",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide 'features' (ordered list) or 'feature_map' (dict)",
        )

    return arr


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model_loader.is_loaded}


@app.get("/")
async def root():
    """Redirect to static frontend."""
    return RedirectResponse(url="/static/index.html")


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata."""
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=model_loader.model_name or "unknown",
        run_dir=model_loader.run_dir or "",
        metrics=model_loader.test_metrics or {},
        feature_names=model_loader.feature_names,
        input_dim=int(model_loader.input_dim or 0),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Single prediction endpoint."""
    if not model_loader.is_loaded or model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        x = _prepare_features(req)
        class_index, probs = model_loader.predict(x)

        monitoring.log_prediction(
            is_batch=False,
            num_samples=1,
            duration_ms=(time.time() - start_time) * 1000,
            confidence=float(np.max(probs)),
        )

        return PredictResponse(
            model_name=model_loader.model_name or "",
            class_index=int(class_index),
            class_probabilities=list(map(float, probs)),
            confidence=float(np.max(probs)),
        )
    except HTTPException:
        raise
    except Exception as e:
        monitoring.log_prediction(
            is_batch=False,
            num_samples=1,
            duration_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(req: BatchPredictRequest):
    """Batch prediction endpoint for multiple samples."""
    if not model_loader.is_loaded or model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    results = []
    failed = 0

    monitoring.info(f"Batch prediction: {len(req.predictions)} samples")

    for i, pred_req in enumerate(req.predictions):
        try:
            x = _prepare_features(pred_req)
            class_index, probs = model_loader.predict(x)

            result = {
                "index": i,
                "class_index": int(class_index),
                "class_probabilities": list(map(float, probs)),
            }
            if req.include_confidence:
                result["confidence"] = float(np.max(probs))

            results.append(result)
        except Exception as e:
            monitoring.warning(f"Batch prediction {i} failed: {e}")
            failed += 1
            results.append(
                {
                    "index": i,
                    "error": str(e),
                }
            )

    duration_ms = (time.time() - start_time) * 1000

    monitoring.log_prediction(
        is_batch=True,
        num_samples=len(req.predictions),
        duration_ms=duration_ms,
    )

    return BatchPredictResponse(
        model_name=model_loader.model_name or "",
        total_predictions=len(req.predictions),
        successful_predictions=len(req.predictions) - failed,
        failed_predictions=failed,
        results=results,
        processing_time_ms=duration_ms,
    )


@app.post("/predict-match", response_model=MatchPredictResponse)
async def predict_match(req: MatchPredictRequest):
    """Predict match outcome from match ID."""
    if not _REQUESTER or not _MATCH_FETCHER or not _FEATURE_ENGINEER:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Set RIOT_API_KEY environment variable.",
        )

    if not model_loader.is_loaded or model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check cache first
    cache_key = f"match_pred:{req.match_id}"
    cached = await cache_service.get(cache_key)
    if cached:
        monitoring.log_cache("hit", cache_key)
        return MatchPredictResponse(**json.loads(cached))

    monitoring.log_cache("miss", cache_key)

    start_time = time.time()

    try:
        monitoring.info(f"Processing match {req.match_id}")

        # Process match using match fetcher
        kpi_limit = int(os.getenv("PREDICT_KPI_LIMIT", "10"))
        match_record, player_history_records = _MATCH_FETCHER._process_single_match(
            req.match_id, match_limit_per_player=kpi_limit
        )

        if not match_record:
            raise HTTPException(status_code=400, detail="Failed to process match data")

        # Create DataFrames
        match_df = pd.DataFrame([match_record])
        player_history_df = pd.DataFrame(player_history_records)

        # Join data
        temp_handler = DataHandler(
            logger=logger,
            parquet_handler=_PARQUET_HANDLER,
            target_feature="team1_win",
            random_state=_RANDOM_SEED,
            master_registry=None,
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            match_tmp = os.path.join(tmpdir, "match.parquet")
            player_tmp = os.path.join(tmpdir, "player.parquet")
            _PARQUET_HANDLER.write_parquet(match_df, match_tmp)
            _PARQUET_HANDLER.write_parquet(player_history_df, player_tmp)
            temp_handler.join_match_and_player_data(match_parquet_path=match_tmp, player_parquet_path=player_tmp)

        combined_df = temp_handler.get_combined_dataframe()

        # Generate features
        try:
            X = _FEATURE_ENGINEER.generate_new_features(combined_df.copy())
        except Exception:
            X = combined_df.copy()

        # Drop non-feature columns
        drop_cols = [col for col in ["team1_win", "match_id", "game_version"] if col in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        # Apply preprocessing
        if model_loader.preprocessor is not None:
            try:
                X_pre = model_loader.preprocessor.transform(X if isinstance(X, pd.DataFrame) else pd.DataFrame(X))
            except Exception as e:
                monitoring.warning(f"Preprocessing failed: {e}")
                X_pre = X
        else:
            X_pre = X

        # Align features
        if model_loader.feature_names:
            if isinstance(X_pre, pd.DataFrame):
                X_pre = X_pre.reindex(columns=model_loader.feature_names, fill_value=0.0)
                x = X_pre.values.astype(np.float32)
            else:
                x = X_pre
        else:
            x = X_pre.values if isinstance(X_pre, pd.DataFrame) else X_pre

        # Predict
        class_index, probs = model_loader.predict(x)

        # Map to probabilities
        idx_blue = 1
        idx_red = 0
        if hasattr(model_loader.model, "classes_"):
            classes = list(model_loader.model.classes_)
            if True in classes:
                idx_blue = classes.index(True)
            elif 1 in classes:
                idx_blue = classes.index(1)
            if False in classes:
                idx_red = classes.index(False)
            elif 0 in classes:
                idx_red = classes.index(0)

        team_blue_prob = float(probs[idx_blue]) if idx_blue < len(probs) else 0.5
        team_red_prob = float(probs[idx_red]) if idx_red < len(probs) else 0.5

        predicted_winner = "red" if team_red_prob > team_blue_prob else "blue"
        confidence = max(team_red_prob, team_blue_prob)

        # Create prediction ID for sharing
        pred_data = {
            "match_id": req.match_id,
            "red_prob": team_red_prob,
            "blue_prob": team_blue_prob,
        }
        pred_id = _create_prediction_id(pred_data)

        result = MatchPredictResponse(
            match_id=req.match_id,
            team_red_win_probability=team_red_prob,
            team_blue_win_probability=team_blue_prob,
            predicted_winner=predicted_winner,
            confidence=confidence,
            share_url=_generate_share_url(pred_id),
        )

        # Cache for 1 hour
        await cache_service.set(cache_key, result.model_dump_json(), ttl=3600)

        duration_ms = (time.time() - start_time) * 1000
        monitoring.log_prediction(
            is_batch=False,
            num_samples=1,
            duration_ms=duration_ms,
            confidence=confidence,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        monitoring.log_prediction(
            is_batch=False,
            num_samples=1,
            duration_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/check-live-game", response_model=LiveGameResponse)
async def check_live_game(req: LiveGameRequest):
    """Check for active game and predict outcome."""
    if not _REQUESTER or not _MATCH_FETCHER or not _FEATURE_ENGINEER:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Set RIOT_API_KEY environment variable.",
        )

    if not model_loader.is_loaded or model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        monitoring.info(f"Checking live game for {req.game_name}#{req.tag_line}")

        match_pre_features = _MATCH_FETCHER.fetch_active_game_pre_features(
            req.game_name, req.tag_line, data_miner=_DATA_MINER
        )

        if not match_pre_features:
            return LiveGameResponse(
                has_active_game=False,
                error="No active game found.",
            )

        # Process similar to completed match
        participants = match_pre_features.get("team1_participants", []) + match_pre_features.get(
            "team2_participants", []
        )

        monitoring.info(f"Live game found with {len(participants)} participants")

        # Fetch player data
        summoner_levels = _MATCH_FETCHER.fetch_summoner_level_data(participants=participants)
        champion_picks = match_pre_features.get("team1_picks", []) + match_pre_features.get("team2_picks", [])
        champion_masteries = _MATCH_FETCHER.fetch_champion_mastery_data(
            participants=participants, champion_picks=champion_picks
        )
        champion_total_mastery_scores = _MATCH_FETCHER.fetch_total_mastery_score(participants=participants)
        rank_queue_data = _MATCH_FETCHER.fetch_rank_queue_data(participants=participants)

        kpi_limit = int(os.getenv("PREDICT_KPI_LIMIT", "10"))
        kpis_data = _MATCH_FETCHER.fetch_raw_player_kpis(
            participants=participants,
            match_limit_per_player=kpi_limit,
            before_timestamp=None,
        )

        # Create match and player records
        match_id = match_pre_features.get("match_id")
        match_record = _MATCH_FETCHER.create_match_record(match_id=match_id, match_pre_features=match_pre_features)

        player_history_records = []
        for i, puuid in enumerate(participants):
            team_id = 100 if i < 5 else 200
            role_list = (
                match_pre_features.get("team1_roles", [])
                if team_id == 100
                else match_pre_features.get("team2_roles", [])
            )
            pick_list = (
                match_pre_features.get("team1_picks", [])
                if team_id == 100
                else match_pre_features.get("team2_picks", [])
            )
            role = role_list[i % 5] if role_list else "unknown"
            champion_id = pick_list[i % 5] if pick_list else 0

            player_history_record = _MATCH_FETCHER.create_player_history_record(
                match_id=match_id,
                puuid=puuid,
                team_id=team_id,
                role=role,
                champion_id=champion_id,
                summoner_level=summoner_levels.get(puuid),
                champion_mastery=champion_masteries.get(puuid),
                champion_total_mastery_score=champion_total_mastery_scores.get(puuid),
                rank_queue_data=rank_queue_data,
                kpis_data=kpis_data.get(puuid),
            )
            player_history_records.append(player_history_record)

        # Process data
        match_df = pd.DataFrame([match_record])
        player_history_df = pd.DataFrame(player_history_records)

        temp_handler = DataHandler(
            logger=logger,
            parquet_handler=_PARQUET_HANDLER,
            target_feature="team1_win",
            random_state=_RANDOM_SEED,
            master_registry=None,
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            match_tmp = os.path.join(tmpdir, "match.parquet")
            player_tmp = os.path.join(tmpdir, "player.parquet")
            _PARQUET_HANDLER.write_parquet(match_df, match_tmp)
            _PARQUET_HANDLER.write_parquet(player_history_df, player_tmp)
            temp_handler.join_match_and_player_data(match_parquet_path=match_tmp, player_parquet_path=player_tmp)

        combined_df = temp_handler.get_combined_dataframe()

        try:
            X = _FEATURE_ENGINEER.generate_new_features(combined_df.copy())
        except Exception:
            X = combined_df.copy()

        drop_cols = [col for col in ["team1_win", "match_id", "game_version"] if col in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        if model_loader.preprocessor is not None:
            try:
                X_pre = model_loader.preprocessor.transform(X if isinstance(X, pd.DataFrame) else pd.DataFrame(X))
            except Exception:
                X_pre = X
        else:
            X_pre = X

        if model_loader.feature_names:
            if isinstance(X_pre, pd.DataFrame):
                X_pre = X_pre.reindex(columns=model_loader.feature_names, fill_value=0.0)
                x = X_pre.values.astype(np.float32)
            else:
                x = X_pre
        else:
            x = X_pre.values if isinstance(X_pre, pd.DataFrame) else X_pre

        class_index, probs = model_loader.predict(x)

        idx_blue = 1
        idx_red = 0
        if hasattr(model_loader.model, "classes_"):
            classes = list(model_loader.model.classes_)
            if True in classes:
                idx_blue = classes.index(True)
            elif 1 in classes:
                idx_blue = classes.index(1)
            if False in classes:
                idx_red = classes.index(False)
            elif 0 in classes:
                idx_red = classes.index(0)

        team_blue_prob = float(probs[idx_blue]) if idx_blue < len(probs) else 0.5
        team_red_prob = float(probs[idx_red]) if idx_red < len(probs) else 0.5
        predicted_winner = "red" if team_red_prob > team_blue_prob else "blue"
        confidence = max(team_red_prob, team_blue_prob)

        # Create prediction ID for sharing
        pred_data = {
            "game_name": req.game_name,
            "tag_line": req.tag_line,
            "red_prob": team_red_prob,
            "blue_prob": team_blue_prob,
        }
        pred_id = _create_prediction_id(pred_data)

        duration_ms = (time.time() - start_time) * 1000
        monitoring.log_prediction(
            is_batch=False,
            num_samples=1,
            duration_ms=duration_ms,
            confidence=confidence,
        )

        return LiveGameResponse(
            has_active_game=True,
            game_id=match_id,
            team_red_win_probability=team_red_prob,
            team_blue_win_probability=team_blue_prob,
            predicted_winner=predicted_winner,
            confidence=confidence,
            share_url=_generate_share_url(pred_id),
        )

    except HTTPException:
        raise
    except Exception as e:
        monitoring.log_prediction(
            is_batch=False,
            num_samples=1,
            duration_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )
        return LiveGameResponse(
            has_active_game=False,
            error=f"Failed to check live game: {str(e)}",
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get application metrics."""
    metrics = monitoring.get_metrics()
    return MetricsResponse(**metrics)
