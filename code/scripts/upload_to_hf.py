#!/usr/bin/env python
"""Upload model artifacts to Hugging Face Hub."""
import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def find_best_model_artifacts(models_path: str) -> dict:
    """Find best model artifacts based on best_overall.json."""
    models_dir = Path(models_path)
    best_overall = models_dir / "best_overall.json"
    
    if not best_overall.exists():
        # Fallback: use LogisticRegression best model
        lr_best = models_dir / "LogisticRegression" / "best.json"
        if lr_best.exists():
            import json
            with open(lr_best) as f:
                best = json.load(f)
            run_dir = models_dir / best.get("run_dir", "")
        else:
            raise FileNotFoundError("No best model found. Train a model first.")
    else:
        import json
        with open(best_overall) as f:
            best = json.load(f)
        run_dir = models_dir / best.get("run_dir", "")
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Collect artifacts
    artifacts = {
        "model.pkl": run_dir / "model.pkl",
        "info.json": run_dir / "info.json",
        "metrics.json": run_dir / "metrics.json",
        "params.json": run_dir / "params.json",
    }
    
    # Add preprocessor if exists
    preprocessor = run_dir / "preprocessor.pkl"
    if preprocessor.exists():
        artifacts["preprocessor.pkl"] = preprocessor
    
    # Verify all exist
    missing = [k for k, v in artifacts.items() if not v.exists()]
    if missing:
        print(f"Warning: Missing artifacts: {missing}")
    
    return {k: v for k, v in artifacts.items() if v.exists()}


def upload_to_hf(repo_id: str, artifacts: dict, token: str = None, commit_message: str = None):
    """Upload artifacts to Hugging Face Hub."""
    api = HfApi()
    
    # Create repo if doesn't exist
    try:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"✓ Repo ready: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload each artifact
    for name, path in artifacts.items():
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=commit_message or f"Update {name}",
            )
            print(f"✓ Uploaded: {name}")
        except Exception as e:
            print(f"✗ Failed to upload {name}: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Upload model artifacts to Hugging Face Hub")
    parser.add_argument("--repo-id", default="PedroGF45/lol-draft-predictor", help="HF repo ID")
    parser.add_argument("--models-path", default="./models", help="Path to models directory")
    parser.add_argument("--token", help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--commit-message", help="Commit message")
    args = parser.parse_args()
    
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not set. Get a token from https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    print(f"Finding best model in {args.models_path}...")
    artifacts = find_best_model_artifacts(args.models_path)
    
    print(f"\nFound {len(artifacts)} artifacts:")
    for name in artifacts:
        print(f"  - {name}")
    
    print(f"\nUploading to {args.repo_id}...")
    upload_to_hf(args.repo_id, artifacts, token, args.commit_message)
    
    print(f"\n✅ Upload complete! View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
