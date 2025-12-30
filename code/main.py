import logging
import os

from data_extraction.data_miner import DataMiner, InvalidPatientZeroError
from data_extraction.match_fetcher import MatchFetcher
from data_extraction.requester import Requester
from data_preparation.data_cleaner import DataCleaner
from data_preparation.data_handler import DataHandler
from dotenv import load_dotenv
from feature_engineering.dimension_reducer import DimensionReducer
from feature_engineering.feature_engineer import FeatureEngineer
from helpers.champion_ids import fetch_latest_champion_ids
from helpers.master_data_registry import MasterDataRegistry
from helpers.parquet_handler import ParquetHandler
from helpers.preprocessing_artifacts import PreprocessingArtifacts
from modeling.deep_learner import DeepLearningClassifier
from modeling.model_builder import ModelBuilder
from visualization.data_visualizer import DataVisualizer

logging.basicConfig(
    level=logging.INFO,  # use DEBUG to see response snippets
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
EXPLORATION_DATA_PATH = os.getenv("EXPLORATION_DATA_PATH")
CLEANED_DATA_PATH = os.getenv("CLEANED_DATA_PATH")
PREPROCESSED_DATA_PATH = os.getenv("PREPROCESSED_DATA_PATH")
FEATURE_ENGINEER_DATA_PATH = os.getenv("FEATURE_ENGINEER_DATA_PATH")
VISUALIZATIONS_PATH = os.getenv("VISUALIZATIONS_PATH")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")
MATCHES_PARQUET_PATH = os.getenv("MATCHES_PARQUET_PATH")
PLAYERS_PARQUET_PATH = os.getenv("PLAYERS_PARQUET_PATH")
RANDOM_SEED = 42

# Concurrency tuning (bounded to respect Riot rate limits)
MATCH_FETCHER_MAX_WORKERS = int(os.getenv("MATCH_FETCHER_MAX_WORKERS", "8"))
DATA_MINER_MAX_WORKERS = int(os.getenv("DATA_MINER_MAX_WORKERS", "4"))

print("RIOT_API_KEY: ", RIOT_API_KEY)

region_v4 = "euw1"
region_v5 = "europe"
base_url_v4 = f"https://{region_v4}.api.riotgames.com"
base_url_v5 = f"https://{region_v5}.api.riotgames.com"
headers = {"X-Riot-Token": RIOT_API_KEY}

requester = Requester(base_url_v4=base_url_v4, base_url_v5=base_url_v5, headers=headers, logger=logger)

# get puuid of summoner
game_name = "Unefraise"
tag_line = "KARAP"
parquet_handler = ParquetHandler(logger=logger, random_state=RANDOM_SEED)

# ═══════════════════════════════════════════════════════════════════════════
# MASTER DATA REGISTRY - Single Source of Truth for Data Collection
# ═══════════════════════════════════════════════════════════════════════════
# Initialize the master registry that tracks all matches and prevents:
# - Duplicate data collection
# - Data leakage between train/test splits
# - Inconsistent incremental data gathering
# ═══════════════════════════════════════════════════════════════════════════

REGISTRY_PATH = os.path.join(RAW_DATA_PATH, "master_registry.pkl")
master_registry = MasterDataRegistry(registry_path=REGISTRY_PATH, logger=logger)

# Log current registry status
registry_stats = master_registry.get_statistics()
logger.info("╔═══════════════════════════════════════════════════════════════════╗")
logger.info("║          MASTER DATA REGISTRY STATUS                              ║")
logger.info("╠═══════════════════════════════════════════════════════════════════╣")
logger.info(f"║  Total Matches:        {registry_stats['total_matches']:>8}                           ║")
logger.info(f"║  Train Matches:        {registry_stats['train_matches']:>8}                           ║")
logger.info(f"║  Test Matches:         {registry_stats['test_matches']:>8}                           ║")
logger.info(f"║  Collection Sessions:  {registry_stats['collection_sessions']:>8}                           ║")
logger.info("╚═══════════════════════════════════════════════════════════════════╝")


# Phase 1: Discover matches (uncomment to run discovery)
# The master_registry automatically filters out duplicates!
data_miner = DataMiner(
    logger=logger,
    parquet_handler=parquet_handler,
    requester=requester,
    raw_data_path=RAW_DATA_PATH,
    patient_zero_game_name=game_name,
    patient_zero_tag_line=tag_line,
    checkpoint_loading_path="F:\\Code\\lol-draft-predictor\\data\\exploration\\checkpoints\\20251229_171629_1300293_players_427407_matches.pkl",
    checkpoint_save_path=CHECKPOINT_PATH,
    master_registry=master_registry,
    max_workers=DATA_MINER_MAX_WORKERS,
)  # ← Registry prevents duplicates

response = data_miner.start_search(
    search_mode="matches", target_number_of_players=1000, target_number_of_matches=1000000
)

"""
# Phase 2: Fetch and enrich matches
# The master_registry filters already-processed matches and registers new ones with game_version
match_fetcher = MatchFetcher(requester=requester, 
                             logger=logger, 
                             parquet_handler=parquet_handler, 
                             dataframe_target_path=RAW_DATA_PATH, 
                             checkpoint_loading_path=CHECKPOINT_PATH,
                             load_percentage=100,
                             random_state=RANDOM_SEED,
                             master_registry=master_registry,
                             max_workers=MATCH_FETCHER_MAX_WORKERS)  # ← Registry tracks match_id + game_version
                             
match_fetcher.fetch_match_data(parquet_path=MATCHES_PARQUET_PATH, match_limit_per_player=10)


# Phase 3: Data preparation and splitting
# The master_registry tracks train/test splits to prevent data leakage
data_handler = DataHandler(logger=logger, 
                           parquet_handler=parquet_handler, 
                           target_feature="team1_win", 
                           random_state=RANDOM_SEED,
                           master_registry=master_registry)  # ← Registry prevents data leakage

# Note: If you have existing preprocessed data and haven't initialized the registry yet,
# run: python code/migrate_existing_data_to_registry.py
# This will populate the registry from your existing matches/players data

# Load existing splits (if available)
data_handler.load_splits(input_dir=CLEANED_DATA_PATH, timestamp="20251222_114237")


 # Full pipeline with registry integration
data_handler.join_match_and_player_data(
    match_parquet_path=MATCHES_PARQUET_PATH,
    player_parquet_path=PLAYERS_PARQUET_PATH
)

# Split data - registry tracks which matches go to train vs test
data_handler.split_data()

# Validate no data leakage
master_registry.validate_no_leakage()  # Raises error if any overlap detected!

# IMPORTANT: After registering splits, drop tracking columns before model training
data_handler.drop_tracking_columns()  # Removes match_id and game_version

data_cleaner = DataCleaner(requester=requester, logger=logger, data_handler=data_handler, parquet_handler=parquet_handler)
data_cleaner.clean_data(output_path=CLEANED_DATA_PATH) 

# Export registry for inspection
master_registry.export_master_dataset(os.path.join(RAW_DATA_PATH, "registry_export.csv"))

# Get collection history
collection_summary = master_registry.get_collection_summary()
logger.info(f"Collection Summary:\n{collection_summary}")


data_visualizer = DataVisualizer(logger=logger, parquet_handler=parquet_handler, random_state=RANDOM_SEED)
data_visualizer.perform_eda(
    prefix="player_history",
    figsize=(12, 10),
    cmap="viridis",
    annot=True,
    fmt=".2f",
    sample_size=10000,
    dpi=100,
    n_estimators=5,
    n_repeats=2,
    data_handler=data_handler,
    data_output_path=VISUALIZATIONS_PATH
)


feature_engineer = FeatureEngineer(logger=logger, parquet_handler=parquet_handler, random_state=RANDOM_SEED)
feature_engineer.perform_feature_engineering(data_handler=data_handler, output_dir=FEATURE_ENGINEER_DATA_PATH)

dimension_reducer = DimensionReducer(logger=logger, parquet_handler=parquet_handler)
dimension_reducer.perform_dimension_reduction(
    data_handler=data_handler,
    pca_plot_dir=VISUALIZATIONS_PATH,
    output_dir=FEATURE_ENGINEER_DATA_PATH
)

data_visualizer = DataVisualizer(logger=logger, parquet_handler=parquet_handler, random_state=RANDOM_SEED)
data_visualizer.perform_eda(
    figsize=(12, 10),
    cmap="viridis",
    annot=True,
    fmt=".2f",
    sample_size=10000,
    dpi=100,
    n_estimators=5,
    n_repeats=2,
    data_handler=data_handler,
    data_output_path=VISUALIZATIONS_PATH
)


model_builder = ModelBuilder(
    X_train=data_handler.get_data_train(),
    y_train=data_handler.get_labels_train(),
    X_test=data_handler.get_data_test(),
    y_test=data_handler.get_labels_test(),
    save_models=True,
    output_path=MODELS_PATH,
    logger=logger,
    auto_preprocess=False
)




# Persist preprocessing artifacts for inference
try:
    # Use the exact feature set used to fit PCA (pre-PCA columns)
    numeric_features = getattr(dimension_reducer, "pca_input_features", list(data_handler.get_data_train().columns))
    # In FeatureEngineer, numerical transforms were applied on numeric features detected at the time.
    # To reproduce, we store the fitted transformers and the PCA configuration.
    preproc = PreprocessingArtifacts(
        numeric_features=numeric_features,
        quantile_transformer=feature_engineer.quantile_transformer,
        standard_scaler_input=feature_engineer.standard_scaler,
        pca_model=dimension_reducer.pca_model,
        pca_output_scaler=dimension_reducer.pca_output_scaler,
        pca_input_features=getattr(dimension_reducer, "pca_input_features", numeric_features),
    )
    # Attach to model_builder so it gets saved with artifacts
    model_builder.preprocessor = preproc
    logger.info("Attached training preprocessor artifacts for saving")
except Exception as e:
    logger.warning(f"Failed to attach preprocessing artifacts: {e}")

model_builder.build_models()
model_builder.plot_summary()"""
