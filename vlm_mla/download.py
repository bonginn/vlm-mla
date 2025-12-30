from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True, type=str)
    p.add_argument("--local_dir", required=True, type=str)
    p.add_argument("--revision", default=None, type=str)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), type=str)
    args = p.parse_args()

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        revision=args.revision,
        token=args.token,
        local_dir_use_symlinks=False,
    )
    print(f"[ok] downloaded {args.repo_id} -> {args.local_dir}")


if __name__ == "__main__":
    main()
