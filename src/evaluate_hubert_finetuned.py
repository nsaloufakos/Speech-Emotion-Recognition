"""Fine-tune HuBERT on original audio with the fixed matched SER cohort."""

import argparse
from functools import partial
import hashlib
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch

import run_experiments as experiments
from models import finetuned_hubert_linear as hubert

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "output/pitch_comparison_hubert/matched_recordings.csv"
SNAPSHOT_POINTER = ROOT / ".cache/hubert_model_path.txt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=hubert.CONFIG["epochs"])
    parser.add_argument("--seeds", type=int, nargs="+", default=experiments.CONFIG["seeds"])
    parser.add_argument("--folds", type=int, nargs="+", choices=experiments.CONFIG["folds"],
                        default=experiments.CONFIG["folds"])
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.folds)) != len(args.folds):
        parser.error("seeds and folds must be unique")
    if experiments.DEVICE.type != "cuda":
        parser.error("CUDA is required for this runner")
    if not SNAPSHOT_POINTER.is_file():
        parser.error("Cache the pinned HuBERT checkpoint first using src/cache_pretrained_model.py")

    checkpoint = (ROOT / SNAPSHOT_POINTER.read_text().strip()).resolve(strict=True)
    data_dir = ROOT / "data/RAVDESS"
    splits = ROOT / "output/splits.csv"
    cohort = pd.read_csv(COHORT).file
    config = {
        **hubert.CONFIG,
        "checkpoint": str(checkpoint),
        "checkpoint_revision": checkpoint.name,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "folds": args.folds,
        "cohort_sha256": sha(COHORT),
        "splits_sha256": sha(splits),
        "torch_version": str(torch.__version__),
        "transformers_version": version("transformers"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    experiments.SPLITS_PATH = splits
    experiments.run_experiment(
        partial(hubert.HuBERTClassifier, checkpoint=checkpoint),
        partial(hubert.extract_features, checkpoint=checkpoint, data_dir=data_dir),
        config,
        hubert.OUTPUT_DIR,
        data_dir=data_dir,
        dataset="original",
        experiment_root=args.output_root,
        recording_files=cohort,
    )


if __name__ == "__main__":
    main()
