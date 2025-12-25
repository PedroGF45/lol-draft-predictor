"""
Migration script to populate Master Data Registry from existing preprocessed data.

Use this script when you already have preprocessed matches/players data and want to
initialize the registry without re-running the entire data collection pipeline.
"""

from helpers.master_data_registry import MasterDataRegistry
from helpers.parquet_handler import ParquetHandler
import pandas as pd
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
RANDOM_SEED = 42

# Initialize components
parquet_handler = ParquetHandler(logger=logger, random_state=RANDOM_SEED)
REGISTRY_PATH = os.path.join(RAW_DATA_PATH, "master_registry.pkl")
registry = MasterDataRegistry(registry_path=REGISTRY_PATH, logger=logger)

logger.info("=" * 80)
logger.info("MASTER DATA REGISTRY - MIGRATION FROM EXISTING DATA")
logger.info("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Register all matches from preprocessed data
# ═══════════════════════════════════════════════════════════════════════════

# Update this path to your preprocessed matches file
PREPROCESSED_MATCHES_PATH = "F:\\Code\\lol-draft-predictor\\data\\preprocessed\\matches\\20251213_152226_484_matches_10_players_per_match.parquet"

logger.info(f"\nStep 1: Loading existing matches from {PREPROCESSED_MATCHES_PATH}")

if os.path.exists(PREPROCESSED_MATCHES_PATH):
    matches_df = parquet_handler.read_parquet(PREPROCESSED_MATCHES_PATH, load_percentage=1.0)
    logger.info(f"Loaded {len(matches_df)} matches")
    
    # Check if we have the required columns
    if 'match_id' not in matches_df.columns:
        logger.error("ERROR: 'match_id' column not found in matches dataframe!")
        exit(1)
    
    if 'game_version' not in matches_df.columns:
        logger.warning("WARNING: 'game_version' column not found. Using 'unknown' as default.")
        matches_df['game_version'] = 'unknown'
    
    # Show sample
    logger.info(f"\nSample of data to register:")
    logger.info(f"\n{matches_df[['match_id', 'game_version']].head(10)}")
    
    # Register matches with registry
    logger.info(f"\nRegistering {len(matches_df)} matches with registry...")
    collection_metadata = {
        'source': 'migration_script',
        'original_file': os.path.basename(PREPROCESSED_MATCHES_PATH),
        'migration_date': pd.Timestamp.now().isoformat()
    }
    
    new_matches_df, duplicates = registry.register_matches(
        matches_df[['match_id', 'game_version']],
        collection_metadata=collection_metadata
    )
    
    logger.info(f"✓ Registration complete: {len(new_matches_df)} new matches, {duplicates} duplicates")
else:
    logger.error(f"ERROR: File not found: {PREPROCESSED_MATCHES_PATH}")
    logger.error("Please update PREPROCESSED_MATCHES_PATH in this script to point to your matches file")
    exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Register train/test splits (if you have existing splits)
# ═══════════════════════════════════════════════════════════════════════════

logger.info("\n" + "=" * 80)
logger.info("Step 2: Registering train/test splits (if available)")
logger.info("=" * 80)

# Update these paths to your existing cleaned/split data
CLEANED_DATA_DIR = "F:\\Code\\lol-draft-predictor\\data\\cleaned\\20251218_210913"
TRAIN_DATA_PATH = os.path.join(CLEANED_DATA_DIR, "data_train.parquet")
TEST_DATA_PATH = os.path.join(CLEANED_DATA_DIR, "data_test.parquet")

if os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH):
    logger.info(f"\nLoading existing train/test splits...")
    
    train_df = parquet_handler.read_parquet(TRAIN_DATA_PATH, load_percentage=1.0)
    test_df = parquet_handler.read_parquet(TEST_DATA_PATH, load_percentage=1.0)
    
    logger.info(f"Train set: {len(train_df)} rows")
    logger.info(f"Test set: {len(test_df)} rows")
    
    # Extract match keys from splits
    # Note: This assumes match_id and game_version columns exist in the split data
    if 'match_id' in train_df.columns and 'game_version' in train_df.columns:
        train_keys = set(zip(train_df['match_id'].astype(str), train_df['game_version'].astype(str)))
        test_keys = set(zip(test_df['match_id'].astype(str), test_df['game_version'].astype(str)))
        
        logger.info(f"\nExtracted {len(train_keys)} unique train matches")
        logger.info(f"Extracted {len(test_keys)} unique test matches")
        
        # Register splits with registry
        logger.info(f"\nRegistering split assignments...")
        registry.assign_splits(train_keys, test_keys)
        
        # Validate no leakage
        logger.info(f"Validating data integrity...")
        try:
            registry.validate_no_leakage()
            logger.info("✓ No data leakage detected!")
        except ValueError as e:
            logger.error(f"✗ Data leakage detected: {e}")
            exit(1)
    else:
        logger.warning("WARNING: Could not find match_id/game_version columns in split data")
        logger.warning("You'll need to re-split your data to register split assignments")
        logger.warning("This is OK - just run data_handler.split_data() with the registry")
else:
    logger.info("\nNo existing splits found (this is OK if you haven't split yet)")
    logger.info("When you run data_handler.split_data(), it will register the splits")

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Show final registry statistics
# ═══════════════════════════════════════════════════════════════════════════

logger.info("\n" + "=" * 80)
logger.info("MIGRATION COMPLETE - FINAL REGISTRY STATUS")
logger.info("=" * 80)

stats = registry.get_statistics()
logger.info(f"\n{'Registry Statistics':^50}")
logger.info("-" * 50)
logger.info(f"Total Matches:        {stats['total_matches']:>8}")
logger.info(f"Train Matches:        {stats['train_matches']:>8}")
logger.info(f"Test Matches:         {stats['test_matches']:>8}")
logger.info(f"Unassigned Matches:   {stats['unassigned_matches']:>8}")
logger.info(f"Collection Sessions:  {stats['collection_sessions']:>8}")
logger.info("-" * 50)

if stats['game_versions']:
    logger.info(f"\n{'Game Version Breakdown':^50}")
    logger.info("-" * 50)
    for version, count in sorted(stats['game_versions'].items()):
        logger.info(f"{version:<30} {count:>8} matches")
    logger.info("-" * 50)

# Export registry for inspection
export_path = os.path.join(RAW_DATA_PATH, "registry_export_after_migration.csv")
registry.export_master_dataset(export_path)
logger.info(f"\n✓ Registry exported to: {export_path}")

# Show collection history
history_df = registry.get_collection_summary()
if len(history_df) > 0:
    logger.info(f"\n{'Collection History':^50}")
    logger.info("-" * 50)
    logger.info(f"\n{history_df.to_string()}")

logger.info("\n" + "=" * 80)
logger.info("NEXT STEPS:")
logger.info("=" * 80)
logger.info("1. Your registry is now populated with existing matches")
logger.info("2. Update main.py to use the registry (see examples in code)")
logger.info("3. When you run the pipeline again, duplicates will be filtered")
logger.info("4. If you didn't have splits yet, run data_handler.split_data()")
logger.info("   with master_registry parameter to register the splits")
logger.info("=" * 80)
