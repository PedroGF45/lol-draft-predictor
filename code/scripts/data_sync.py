import argparse
import logging
import sys
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv

    # Load .env from repo root
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass  # dotenv optional

# Add parent directory to path to import helpers
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.hf_data_repo import download_dataset_repo, upload_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("data_sync")


def _default_paths(local_dir: Path) -> List[Path]:
    base = local_dir
    return [
        base / "exploration",
        base / "preprocessed",
        base / "cleaned",
        base / "feature_engineered",
        base / "registry_export.csv",
        base / "master_registry.pkl",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync dataset with Hugging Face Hub (dataset repo)")
    parser.add_argument("--action", choices=["pull", "push"], required=True, help="pull=download, push=upload")
    parser.add_argument("--local-dir", default="data", help="Local directory for dataset contents")
    parser.add_argument("--paths", nargs="*", help="Specific paths to push (relative or absolute)")
    parser.add_argument("--repo-id", help="Dataset repo id (overrides env HF_DATASET_REPO_ID)")
    parser.add_argument("--token", help="HF token (overrides env HF_TOKEN)")
    parser.add_argument("--commit-message", help="Optional commit message for push")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)

    if args.action == "pull":
        download_dataset_repo(local_dir=str(local_dir), repo_id=args.repo_id, token=args.token, logger=logger)
        return

    # push
    if args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        paths = _default_paths(local_dir)

    existing = [p for p in paths if p.exists()]
    if not existing:
        logger.warning("No existing paths to push. Nothing to do.")
        return

    upload_paths(
        paths=[str(p) for p in existing],
        local_root=str(local_dir),
        repo_root="",
        repo_id=args.repo_id,
        token=args.token,
        commit_message=args.commit_message or "Sync dataset",
        logger=logger,
    )


if __name__ == "__main__":
    main()
