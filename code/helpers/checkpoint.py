import logging
import os
import pickle
import tempfile
from typing import Any, Optional


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
    # If caller passed a directory, attempt to load the latest .pkl checkpoint inside
    try:
        if not path:
            logger.info("No checkpoint path provided; starting fresh")
            return None

        if os.path.isdir(path):
            logger.info(f"Checkpoint path is a directory; searching for latest checkpoint in {path}")
            candidates = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(".pkl") and os.path.isfile(os.path.join(path, f))
            ]
            if not candidates:
                logger.info("No checkpoint files found in directory; starting fresh")
                return None
            # pick most recently modified file
            latest = max(candidates, key=os.path.getmtime)
            path_to_load = latest
        else:
            path_to_load = path

        if not os.path.exists(path_to_load):
            logger.info("Checkpoint file not found; starting fresh")
            return None

        with open(path_to_load, "rb") as f:
            state = pickle.load(f)
            logger.info(f"Checkpoint loaded from {path_to_load}")
            return state
    except Exception as e:
        logger.warning(f"Loading checkpoint failed: {e}")
        return None
