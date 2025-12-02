import os
import pickle
import tempfile
from typing import Any, Optional
import logging


def save_checkpoint(logger: logging.Logger, state: dict[str, Any], path: str) -> None:
    """
    Atomically persist a checkpoint to disk using pickle.

    Writes to a temporary file in the same directory and renames on success
    to avoid partial writes. "path" must be a file path (e.g., .../checkpoint.pkl).
    """
    try:
        if not path or path.endswith(os.sep):
            raise ValueError("'path' must be a file path, not a directory")

        target_dir = os.path.dirname(path) or "."
        os.makedirs(target_dir, exist_ok=True)

        # Ensure temp file is created in the same directory for atomic replace
        with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tmp:
            tmp_path = tmp.name
            pickle.dump(state, tmp)
            logger.debug(f"Temporary checkpoint written to {tmp_path}")

        os.replace(tmp_path, path)
        logger.info(f"Checkpoint saved to {path}")

    except Exception as e:
        logger.error(f"Saving checkpoint failed: {e}")


def load_checkpoint(logger: logging.Logger, path: str) -> Optional[dict[str, Any]]:
    """
    Load a checkpoint if it exists; return None if missing or invalid.
    """
    if not path or path.endswith(os.sep):
        logger.warning("'path' should be a file path; received a directory-like path")
        return None

    if not os.path.exists(path):
        logger.info("Checkpoint file not found; starting fresh")
        return None

    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
            logger.info(f"Checkpoint loaded from {path}")
            return state
    except Exception as e:
        logger.warning(f"Loading checkpoint failed: {e}")
        return None
