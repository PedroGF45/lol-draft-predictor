import logging
import os
from pathlib import Path
from typing import Iterable, Optional

try:
    from huggingface_hub import HfApi, create_repo, snapshot_download
except ImportError as e:  # pragma: no cover - dependency handled by workflows
    raise ImportError("huggingface_hub is required for dataset syncing. Install with `pip install huggingface_hub`.\nOriginal error: " + str(e))


DEFAULT_DATASET_REPO_ID = "PedroGF45/lol-draft-predictor-data"


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
    """Download the HF dataset repo into a local directory."""

    log = logger or logging.getLogger(__name__)
    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading dataset repo '{repo_id}' into {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    log.info("Dataset download complete")
    return str(local_dir)


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
    """Upload a directory to the dataset repo, preserving structure."""

    log = logger or logging.getLogger(__name__)
    folder = Path(folder_path)
    if not folder.exists():
        log.warning(f"Skip upload: folder does not exist: {folder}")
        return

    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)
    _ensure_repo(repo_id, token)

    target_path = path_in_repo or folder.as_posix()
    log.info(f"Uploading folder {folder} to {repo_id}:{target_path}")

    api = HfApi()
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=target_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=commit_message or f"Update {folder.name}",
        allow_patterns=None,
    )
    log.info(f"Uploaded folder {folder}")


def upload_file(
    file_path: str,
    path_in_repo: Optional[str] = None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Upload a single file to the dataset repo."""

    log = logger or logging.getLogger(__name__)
    fp = Path(file_path)
    if not fp.exists():
        log.warning(f"Skip upload: file does not exist: {fp}")
        return

    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)
    _ensure_repo(repo_id, token)

    target_path = path_in_repo or fp.as_posix()
    log.info(f"Uploading file {fp} to {repo_id}:{target_path}")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(fp),
        path_in_repo=target_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=commit_message or f"Update {fp.name}",
    )
    log.info(f"Uploaded file {fp}")


def upload_paths(
    paths: Iterable[str],
    local_root: str = "data",
    repo_root: str = "",
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Upload multiple paths, preserving relative layout from local_root."""

    log = logger or logging.getLogger(__name__)
    repo_id = _get_repo_id(repo_id)
    token = _get_token(token)
    _ensure_repo(repo_id, token)

    local_root_path = Path(local_root).resolve()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            log.warning(f"Skip missing path: {path}")
            continue

        try:
            rel = path.resolve().relative_to(local_root_path)
            path_in_repo = Path(repo_root) / rel
        except ValueError:
            path_in_repo = Path(repo_root) / path.name

        if path.is_dir():
            upload_folder(
                folder_path=str(path),
                path_in_repo=path_in_repo.as_posix(),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
                logger=log,
            )
        else:
            upload_file(
                file_path=str(path),
                path_in_repo=path_in_repo.as_posix(),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
                logger=log,
            )
