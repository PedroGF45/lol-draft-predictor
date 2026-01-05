import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from huggingface_hub import HfApi, create_repo, snapshot_download
except ImportError as e:  # pragma: no cover - dependency handled by workflows
    raise ImportError("huggingface_hub is required for dataset syncing. Install with `pip install huggingface_hub`.\nOriginal error: " + str(e))


DEFAULT_DATASET_REPO_ID = "PedroGF45/lol-draft-predictor-data"


def _normalize_repo_path(path: Path) -> str:
    """
    Normalize a path for use in Hugging Face repos.
    Always uses forward slashes, regardless of OS.
    
    Args:
        path: Path object to normalize
    
    Returns:
        String path with forward slashes
    """
    return path.as_posix()


def _get_repo_id(repo_id: Optional[str] = None) -> str:
    return repo_id or os.getenv("HF_DATASET_REPO_ID") or DEFAULT_DATASET_REPO_ID


def _get_token(token: Optional[str] = None) -> Optional[str]:
    return token or os.getenv("HF_TOKEN")


def download_dataset_repo(
    local_dir: str = "data",
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Download the HF dataset repo into a local directory.
    Cross-platform compatible path handling.
    
    Args:
        local_dir: Local directory to download to
        repo_id: HF dataset repo ID
        token: HF authentication token
        allow_patterns: Optional patterns to filter downloads
        logger: Optional logger instance
    
    Returns:
        String path to downloaded directory
    """
    log = logger or logging.getLogger(__name__)
    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)

    local_path = Path(local_dir).resolve()
    local_path.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Downloading dataset repo '{repo_id}' into {local_path}")
    log.info(f"Platform: {sys.platform}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
        log.info("Dataset download complete")
    except Exception as e:
        log.error(f"Failed to download dataset: {e}")
        raise
    
    return str(local_path)


def _ensure_repo(repo_id: str, token: Optional[str]) -> None:
    api = HfApi()
    create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)


def upload_folder(
    folder_path: str,
    path_in_repo: Optional[str] = None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Upload a directory to the dataset repo, preserving structure.
    Uses forward slashes for repo paths on all platforms.
    
    Args:
        folder_path: Local folder path to upload
        path_in_repo: Target path in repo (uses forward slashes)
        repo_id: HF dataset repo ID
        token: HF authentication token
        commit_message: Commit message
        logger: Optional logger instance
    """
    log = logger or logging.getLogger(__name__)
    folder = Path(folder_path).resolve()
    
    if not folder.exists():
        log.warning(f"Skip upload: folder does not exist: {folder}")
        return
    
    if not folder.is_dir():
        log.warning(f"Skip upload: not a directory: {folder}")
        return

    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)
    _ensure_repo(repo_id, token)

    # Ensure path uses forward slashes for repo
    target_path = path_in_repo or _normalize_repo_path(Path(folder.name))
    log.info(f"Uploading folder {folder.name} to {repo_id}:{target_path}")

    api = HfApi()
    try:
        api.upload_folder(
            folder_path=str(folder),
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=commit_message or f"Update {folder.name}",
            allow_patterns=None,
        )
        log.info(f"✓ Uploaded folder {folder.name}")
    except Exception as e:
        log.error(f"Failed to upload folder {folder.name}: {e}")
        raise


def upload_file(
    file_path: str,
    path_in_repo: Optional[str] = None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Upload a single file to the dataset repo.
    Uses forward slashes for repo paths on all platforms.
    
    Args:
        file_path: Local file path to upload
        path_in_repo: Target path in repo (uses forward slashes)
        repo_id: HF dataset repo ID
        token: HF authentication token
        commit_message: Commit message
        logger: Optional logger instance
    """
    log = logger or logging.getLogger(__name__)
    fp = Path(file_path).resolve()
    
    if not fp.exists():
        log.warning(f"Skip upload: file does not exist: {fp}")
        return
    
    if not fp.is_file():
        log.warning(f"Skip upload: not a file: {fp}")
        return

    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)
    _ensure_repo(repo_id, token)

    # Ensure path uses forward slashes for repo
    target_path = path_in_repo or _normalize_repo_path(Path(fp.name))
    file_size_mb = fp.stat().st_size / 1024 / 1024
    log.info(f"Uploading file {fp.name} ({file_size_mb:.2f} MB) to {repo_id}:{target_path}")

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=str(fp),
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=commit_message or f"Update {fp.name}",
        )
        log.info(f"✓ Uploaded file {fp.name}")
    except Exception as e:
        log.error(f"Failed to upload file {fp.name}: {e}")
        raise


def upload_paths(
    paths: Iterable[str],
    local_root: str = "data",
    repo_root: str = "",
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Upload multiple paths, preserving relative layout from local_root.
    Cross-platform compatible - handles Windows, Linux, and macOS paths.
    
    Args:
        paths: Iterable of local paths to upload
        local_root: Local root directory (used for computing relative paths)
        repo_root: Root path in the repo
        repo_id: HF dataset repo ID
        token: HF authentication token
        commit_message: Commit message for all uploads
        logger: Optional logger instance
    """
    log = logger or logging.getLogger(__name__)
    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)
    _ensure_repo(repo_id, token)

    local_root_path = Path(local_root).resolve()
    repo_root_path = Path(repo_root) if repo_root else Path("")
    
    log.info(f"Uploading from local root: {local_root_path}")
    log.info(f"Uploading to repo root: {_normalize_repo_path(repo_root_path) or '(root)'}")
    
    upload_count = 0
    error_count = 0
    
    for raw in paths:
        path = Path(raw).resolve()
        if not path.exists():
            log.warning(f"Skip missing path: {path}")
            continue

        try:
            # Try to compute relative path from local_root
            rel = path.relative_to(local_root_path)
            # Combine repo_root with relative path, ensure forward slashes
            path_in_repo = _normalize_repo_path(repo_root_path / rel)
        except ValueError:
            # Path is not relative to local_root, use just the name
            log.warning(f"Path {path} is not under {local_root_path}, using name only")
            path_in_repo = _normalize_repo_path(repo_root_path / path.name)

        try:
            if path.is_dir():
                upload_folder(
                    folder_path=str(path),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    token=token,
                    commit_message=commit_message,
                    logger=log,
                )
            else:
                upload_file(
                    file_path=str(path),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    token=token,
                    commit_message=commit_message,
                    logger=log,
                )
            upload_count += 1
        except Exception as e:
            log.error(f"Error uploading {path}: {e}")
            error_count += 1
    
    log.info(f"Upload summary: {upload_count} succeeded, {error_count} failed")
