"""Download a pinned official speech checkpoint into the repository cache."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="facebook/wav2vec2-base")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/wav2vec_model_path.txt")
    args = parser.parse_args()
    info = HfApi().model_info(args.repo, revision=args.revision)
    names = {file.rfilename for file in info.siblings}
    weights = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    if weights not in names:
        raise ValueError("No supported PyTorch checkpoint in the official repository")
    print(f"Caching {args.repo} revision {info.sha} ({weights})", flush=True)
    snapshot = Path(snapshot_download(args.repo, revision=info.sha,
                                     cache_dir=ROOT / ".cache/huggingface/hub",
                                     allow_patterns=[weights, "*.json", "README.md"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(str(snapshot.relative_to(ROOT)) + "\n")
    manifest = {"repo": args.repo, "revision": info.sha, "weights": weights,
                "weights_sha256": hashlib.sha256((snapshot / weights).read_bytes()).hexdigest(),
                "snapshot_relative_to_repo": str(snapshot.relative_to(ROOT))}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
