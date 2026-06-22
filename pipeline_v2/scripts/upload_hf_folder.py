#!/usr/bin/env python3
"""Upload a local folder to a Hugging Face model repository."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--commit-message", default="Upload model artifacts")
    parser.add_argument("--revision", default="")
    parser.add_argument("--large-folder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).resolve()
    if not folder.exists():
        raise FileNotFoundError(folder)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    if args.large_folder:
        result = api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(folder),
        )
    else:
        result = api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(folder),
            commit_message=args.commit_message,
            revision=args.revision or None,
        )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "repo_type": args.repo_type,
                "folder": str(folder),
                "result": str(result),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
