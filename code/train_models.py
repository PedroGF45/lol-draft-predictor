#!/usr/bin/env python
"""
Simplified training script for CI/CD model refresh.
Loads existing data, trains models, and saves artifacts.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add code directory to path
CODE_DIR = Path(__file__).parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_preparation.data_handler import DataHandler
from dotenv import load_dotenv
from feature_engineering.dimension_reducer import DimensionReducer
from feature_engineering.feature_engineer import FeatureEngineer
from helpers.master_data_registry import MasterDataRegistry
from helpers.parquet_handler import ParquetHandler
from helpers.preprocessing_artifacts import PreprocessingArtifacts
from modeling.model_builder import ModelBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_models(
    cleaned_timestamp: str = None,
    feature_eng_timestamp: str = None,
    models_path: str = "./models",
):
    """
    Train models using existing preprocessed data.
    
    Args:
        cleaned_timestamp: Timestamp of cleaned data to load (e.g., "20251222_114237")
        feature_eng_timestamp: Timestamp of feature engineered data to load
        models_path: Output path for trained models
    """
    RANDOM_SEED = 42
    
    # Load environment
    load_dotenv()
    
    # Get paths from env or defaults
    CLEANED_DATA_PATH = os.getenv("CLEANED_DATA_PATH", "./data/cleaned")
    FEATURE_ENGINEER_DATA_PATH = os.getenv("FEATURE_ENGINEER_DATA_PATH", "./data/feature_engineered")
    MODELS_PATH = models_path or os.getenv("MODELS_PATH", "./models")
    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "./data/raw")
    
    logger.info("=" * 60)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Initialize components
    parquet_handler = ParquetHandler(logger=logger, random_state=RANDOM_SEED)
    
    # Try to use registry if available
    registry_path = os.path.join(RAW_DATA_PATH, "master_registry.pkl")
    master_registry = None
    if os.path.exists(registry_path):
        try:
            master_registry = MasterDataRegistry(registry_path=registry_path, logger=logger)
            logger.info("Master registry loaded")
        except Exception as e:
            logger.warning(f"Could not load registry: {e}")
    
    # Create data handler
    data_handler = DataHandler(
        logger=logger,
        parquet_handler=parquet_handler,
        target_feature="team1_win",
        random_state=RANDOM_SEED,
        master_registry=master_registry,
    )
    
    # Load existing splits
    logger.info(f"Loading cleaned data from {CLEANED_DATA_PATH}...")
    
    if cleaned_timestamp:
        # Load specific timestamp
        data_handler.load_splits(input_dir=CLEANED_DATA_PATH, timestamp=cleaned_timestamp)
    else:
        # Find latest
        cleaned_dirs = [d for d in os.listdir(CLEANED_DATA_PATH) if os.path.isdir(os.path.join(CLEANED_DATA_PATH, d))]
        if cleaned_dirs:
            latest = sorted(cleaned_dirs)[-1]
            logger.info(f"Using latest cleaned data: {latest}")
            data_handler.load_splits(input_dir=CLEANED_DATA_PATH, timestamp=latest)
        else:
            raise FileNotFoundError(f"No cleaned data found in {CLEANED_DATA_PATH}. Run full pipeline first.")
    
    # Feature engineering
    logger.info("Performing feature engineering...")
    feature_engineer = FeatureEngineer(logger=logger, parquet_handler=parquet_handler, random_state=RANDOM_SEED)
    feature_engineer.perform_feature_engineering(data_handler=data_handler, output_dir=FEATURE_ENGINEER_DATA_PATH)
    
    # Dimension reduction
    logger.info("Performing dimension reduction (PCA)...")
    dimension_reducer = DimensionReducer(logger=logger, parquet_handler=parquet_handler)
    dimension_reducer.perform_dimension_reduction(
        data_handler=data_handler,
        pca_plot_dir=None,  # Skip plots in CI
        output_dir=FEATURE_ENGINEER_DATA_PATH,
    )
    
    # Build models
    logger.info("Building models...")
    model_builder = ModelBuilder(
        X_train=data_handler.get_data_train(),
        y_train=data_handler.get_labels_train(),
        X_test=data_handler.get_data_test(),
        y_test=data_handler.get_labels_test(),
        save_models=True,
        output_path=MODELS_PATH,
        logger=logger,
        auto_preprocess=False,
    )
    
    # Attach preprocessing artifacts
    try:
        numeric_features = getattr(dimension_reducer, "pca_input_features", list(data_handler.get_data_train().columns))
        preproc = PreprocessingArtifacts(
            numeric_features=numeric_features,
            quantile_transformer=feature_engineer.quantile_transformer,
            standard_scaler_input=feature_engineer.standard_scaler,
            pca_model=dimension_reducer.pca_model,
            pca_output_scaler=dimension_reducer.pca_output_scaler,
            pca_input_features=getattr(dimension_reducer, "pca_input_features", numeric_features),
        )
        model_builder.preprocessor = preproc
        logger.info("Attached preprocessing artifacts")
    except Exception as e:
        logger.warning(f"Failed to attach preprocessing artifacts: {e}")
    
    # Train
    model_builder.build_models()
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Models saved to: {MODELS_PATH}")
    logger.info(f"Best model: {model_builder.best_model_name}")
    logger.info("=" * 60)
    
    return MODELS_PATH


def main():
    parser = argparse.ArgumentParser(description="Train LoL draft prediction models")
    parser.add_argument(
        "--cleaned-timestamp",
        help="Timestamp of cleaned data to load (e.g., 20251222_114237)",
    )
    parser.add_argument(
        "--feature-timestamp",
        help="Timestamp of feature engineered data to load",
    )
    parser.add_argument(
        "--models-path",
        default="./models",
        help="Output path for trained models",
    )
    args = parser.parse_args()
    
    try:
        models_path = train_models(
            cleaned_timestamp=args.cleaned_timestamp,
            feature_eng_timestamp=args.feature_timestamp,
            models_path=args.models_path,
        )
        logger.info(f"\n✅ Training successful! Models at: {models_path}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
