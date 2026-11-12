"""Download one PEFT adapter from Hugging Face into checkpoints/<name>."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", help="Hugging Face model repository")
    parser.add_argument("name", help="local directory name below checkpoints/")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1] / "checkpoints", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    target = args.root / args.name
    snapshot_download(args.repository, revision=args.revision, local_dir=target)
    if not (target / "adapter_config.json").is_file():
        raise SystemExit(f"{target} is not a PEFT adapter")
    print(target)


if __name__ == "__main__":
    main()
