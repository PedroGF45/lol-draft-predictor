"""
Data Sync for Hugging Face Dataset Repository
Single script to handle all data syncing operations.

Usage:
    # Upload all data
    python code/scripts/data_sync.py push
    
    # Download all data
    python code/scripts/data_sync.py pull
    
    # Upload only registry files
    python code/scripts/data_sync.py push --mode registry
    
    # Upload latest processed data
    python code/scripts/data_sync.py push --mode latest
    
    # Preview what would be uploaded
    python code/scripts/data_sync.py push --dry-run
    
    # Upload specific paths
    python code/scripts/data_sync.py push --paths data/cleaned data/registry_export.csv
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.hf_data_repo import download_dataset_repo, upload_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("data_sync")


def get_all_data_paths(local_dir: Path) -> List[Path]:
    """Get all default data paths."""
    base = Path(local_dir).resolve()
    paths = [
        base / "exploration",
        base / "preprocessed",
        base / "cleaned",
        base / "feature_engineered",
        base / "visualizations",
        base / "registry_export.csv",
        base / "master_registry.pkl",
    ]
    return [p for p in paths if p.exists()]


def get_registry_paths(local_dir: Path) -> List[Path]:
    """Get registry file paths only."""
    base = Path(local_dir).resolve()
    paths = [
        base / "registry_export.csv",
        base / "master_registry.pkl",
    ]
    return [p for p in paths if p.exists()]


def get_latest_paths(local_dir: Path) -> List[Path]:
    """Get latest processed data paths."""
    base = Path(local_dir).resolve()
    paths = []
    
    # Latest cleaned
    cleaned_dir = base / "cleaned"
    if cleaned_dir.exists():
        timestamped = sorted([d for d in cleaned_dir.iterdir() if d.is_dir()], reverse=True)
        if timestamped:
            paths.append(timestamped[0])
            logger.info(f"Latest cleaned: {timestamped[0].name}")
    
    # Latest feature_engineered
    fe_dir = base / "feature_engineered"
    if fe_dir.exists():
        timestamped = sorted([d for d in fe_dir.iterdir() if d.is_dir()], reverse=True)
        if timestamped:
            paths.append(timestamped[0])
            logger.info(f"Latest feature_engineered: {timestamped[0].name}")
    
    # Add registry
    paths.extend(get_registry_paths(local_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync dataset with Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data
  python code/scripts/data_sync.py pull
  
  # Upload all data
  python code/scripts/data_sync.py push
  
  # Upload only registry
  python code/scripts/data_sync.py push --mode registry
  
  # Upload latest processed data
  python code/scripts/data_sync.py push --mode latest
  
  # Dry run (preview)
  python code/scripts/data_sync.py push --dry-run
  
  # Upload specific paths
  python code/scripts/data_sync.py push --paths data/cleaned data/registry_export.csv
        """
    )
    parser.add_argument(
        "action",
        choices=["pull", "push"],
        help="pull=download from HF, push=upload to HF"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "registry", "latest"],
        default="all",
        help="What to sync: all data (default), registry only, or latest processed"
    )
    parser.add_argument(
        "--local-dir",
        default="data",
        help="Local directory for dataset (default: data)"
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Specific paths to push (overrides --mode)"
    )
    parser.add_argument(
        "--repo-id",
        help="HF dataset repo ID (default: env HF_DATASET_REPO_ID)"
    )
    parser.add_argument(
        "--token",
        help="HF token (default: env HF_TOKEN)"
    )
    parser.add_argument(
        "--commit-message",
        help="Commit message for push"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading"
    )
    args = parser.parse_args()

    # Normalize path
    local_dir = Path(args.local_dir).resolve()
    
    if not local_dir.exists():
        logger.error(f"Local directory does not exist: {local_dir}")
        sys.exit(1)

    # PULL ACTION
    if args.action == "pull":
        logger.info(f"📥 Downloading dataset to: {local_dir}")
        download_dataset_repo(
            local_dir=str(local_dir),
            repo_id=args.repo_id,
            token=args.token,
            logger=logger
        )
        logger.info("✅ Download complete!")
        return

    # PUSH ACTION
    # Determine which paths to upload
    if args.paths:
        # User specified paths
        paths = [Path(p).resolve() for p in args.paths]
    elif args.mode == "registry":
        logger.info("📋 Mode: registry files only")
        paths = get_registry_paths(local_dir)
    elif args.mode == "latest":
        logger.info("🔄 Mode: latest processed data")
        paths = get_latest_paths(local_dir)
    else:  # all
        logger.info("📦 Mode: all data")
        paths = get_all_data_paths(local_dir)

    # Filter existing paths
    existing = [p for p in paths if p.exists()]
    missing = [p for p in paths if not p.exists()]
    
    if missing:
        logger.warning("⚠️  Missing paths (will skip):")
        for p in missing:
            logger.warning(f"  - {p}")
    
    if not existing:
        logger.warning("❌ No paths to upload")
        return
    
    # Show what will be uploaded
    logger.info(f"📤 Uploading {len(existing)} path(s):")
    total_size = 0
    for p in existing:
        if p.is_dir():
            file_count = sum(1 for _ in p.rglob('*') if _.is_file())
            logger.info(f"  📁 {p.name} ({file_count} files)")
        else:
            size_mb = p.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info(f"  📄 {p.name} ({size_mb:.2f} MB)")
    
    if args.dry_run:
        logger.info("🔍 DRY RUN - No files uploaded")
        return

    # Check for HF token
    if not args.token and not os.getenv('HF_TOKEN'):
        logger.error("❌ HF_TOKEN not set. Set it with:")
        logger.error("   Windows: $env:HF_TOKEN='your_token'")
        logger.error("   Linux/Mac: export HF_TOKEN='your_token'")
        logger.error("   Or use: --token your_token")
        sys.exit(1)

    # Upload
    upload_paths(
        paths=[str(p) for p in existing],
        local_root=str(local_dir),
        repo_root="",
        repo_id=args.repo_id,
        token=args.token,
        commit_message=args.commit_message or f"Sync data ({args.mode} mode)",
        logger=logger,
    )
    
    logger.info("✅ Upload complete!")


if __name__ == "__main__":
    main()
