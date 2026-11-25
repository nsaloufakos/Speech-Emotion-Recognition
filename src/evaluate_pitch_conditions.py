"""Run matched HuBERT LOSO comparisons without overwriting historical results."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ser-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ser-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ser-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, HubertModel

import run_experiments as experiments
from models.baseline_hubert_linear import HuBERTClassifier, CONFIG as MODEL_CONFIG

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "original": ROOT / "data/RAVDESS",
    "normalized_only": ROOT / "data/audio_variants/normalized_only",
    "normalized_psola": ROOT / "data/audio_variants/normalized_psola",
    "normalized_psola_actor_mean": ROOT / "data/audio_variants/normalized_psola_actor_mean",
    "normalized_psola_actor_mean_cap4": ROOT / "data/audio_variants/normalized_psola_actor_mean_cap4",
    "normalized_psola_actor_praat_median": ROOT / "data/audio_variants/normalized_psola_actor_praat_median",
    "normalized_psola_actor_praat_median_cap4": ROOT / "data/audio_variants/normalized_psola_actor_praat_median_cap4",
}
BASE_CONDITIONS = ["original", "normalized_psola_actor_praat_median_cap4"]


class FrozenFeatures:
    def __init__(self, snapshot, device, output_root, batch_size=8):
        self.snapshot = snapshot
        self.device = device
        self.output_root = output_root
        self.batch_size = batch_size
        self.processor = None
        self.backbone = None

    def extract(self, df, condition):
        cache = self.output_root / condition / "features"
        cache.mkdir(parents=True, exist_ok=True)
        manifest_path = cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        signature = {"snapshot": str(self.snapshot.resolve()), "torch": torch.__version__,
                     "sample_rate": 16000, "pooling": "mean_over_final_layer_time", "device": str(self.device),
                     "attention_implementation": "eager",
                     "packages": {name: importlib.metadata.version(name) for name in ["transformers", "librosa", "numpy", "soundfile"]}}
        if manifest.get("signature") != signature:
            manifest = {"signature": signature, "recordings": {}}
        features = {}
        pending = []
        for row in df.itertuples():
            path = CONDITIONS[condition] / f"Actor_{row.actor_id:02d}" / f"{row.file}.wav"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stored = cache / f"{row.file}.npy"
            if manifest["recordings"].get(row.file) == digest and stored.exists():
                features[row.file] = np.load(stored)
            else:
                pending.append((row.file, path, digest))
        print(f"{condition}: {len(features)} cached embeddings, {len(pending)} to extract", flush=True)
        if pending and self.backbone is None:
            self.processor = AutoFeatureExtractor.from_pretrained(self.snapshot, local_files_only=True)
            self.backbone = HubertModel.from_pretrained(self.snapshot, local_files_only=True,
                                                        attn_implementation="eager").to(self.device)
            self.backbone.eval()
            self.backbone.requires_grad_(False)
        started = time.perf_counter()
        # Batch only recordings with exactly equal resampled sample counts. This avoids
        # padding, which can change HuBERT's group normalization and time pooling.
        groups = {}
        for item in pending:
            info = sf.info(item[1])
            count = (info.frames * 16000 + info.samplerate - 1) // info.samplerate
            groups.setdefault(count, []).append(item)
        batches = [group[start:start + self.batch_size] for group in groups.values()
                   for start in range(0, len(group), self.batch_size)]
        encoded = 0
        with torch.inference_mode():
            for batch in batches:
                audio = [librosa.load(path, sr=16000, mono=True)[0] for _, path, _ in batch]
                if len({len(y) for y in audio}) != 1:
                    raise ValueError("Equal-length batching prediction disagrees with decoded audio")
                inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
                output = self.backbone(inputs.input_values.to(self.device))
                embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()
                if embeddings.shape != (len(batch), 768) or not np.isfinite(embeddings).all():
                    raise ValueError("Invalid HuBERT embeddings")
                for (name, _, digest), embedding in zip(batch, embeddings):
                    embedding = embedding[None, :]
                    temporary = cache / f"{name}.partial.npy"
                    np.save(temporary, embedding)
                    temporary.replace(cache / f"{name}.npy")
                    features[name] = embedding
                    manifest["recordings"][name] = digest
                previous_encoded = encoded
                encoded += len(batch)
                temporary_manifest = manifest_path.with_suffix(".partial.json")
                temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
                temporary_manifest.replace(manifest_path)
                if encoded // 48 > previous_encoded // 48 or encoded == len(pending):
                    print(f"{condition}: {encoded}/{len(pending)} embeddings | {time.perf_counter() - started:.1f}s", flush=True)
        return features


def write_comparison(output_root, requested):
    records = []
    for condition in requested:
        path = output_root / condition / "hubert_baseline/evaluation.json"
        if path.exists():
            evaluation = json.loads(path.read_text())
            records.append({"condition": condition, "source": "matched_rerun",
                            "seeds_completed": evaluation["seeds_completed"],
                            "folds_per_seed": evaluation["folds_per_seed"], **evaluation["summary"]})
    if records:
        pd.DataFrame(records).to_csv(output_root / "comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", choices=list(CONDITIONS), nargs="+", default=BASE_CONDITIONS)
    parser.add_argument("--matched-recordings", type=Path,
                        default=ROOT / "output/pitch_comparison_hubert/matched_recordings.csv",
                        help="Reuse the fixed 1,429-recording comparison cohort")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, 25)))
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/current_hubert")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--variant-root", type=Path, help="Optional root containing regenerated normalized conditions")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Extract eight original embeddings, then exit")
    parser.add_argument("--benchmark-recordings", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.variant_root is not None:
        for name in CONDITIONS:
            if name != "original":
                CONDITIONS[name] = args.variant_root.resolve() / name
    snapshot = args.snapshot or Path((ROOT / ".cache/hubert_model_path.txt").read_text().strip())
    if not snapshot.is_absolute():
        snapshot = ROOT / snapshot
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    device = torch.device(device)
    print("Feature extraction device:", device, flush=True)
    # Tiny linear heads train efficiently on CPU and follow the original runner.
    experiments.DEVICE = torch.device("cpu")
    torch.set_num_threads(4)
    if args.benchmark:
        df = pd.read_csv(CONDITIONS["original"] / "index.csv").sort_values("file").head(args.benchmark_recordings)
        started = time.perf_counter()
        FrozenFeatures(snapshot, device, args.output_root, args.feature_batch_size).extract(df, "original")
        print(f"{len(df)}-recording benchmark: {time.perf_counter() - started:.1f}s", flush=True)
        return
    index_names = list(dict.fromkeys(args.conditions))
    indices = {name: pd.read_csv(CONDITIONS[name] / "index.csv") for name in index_names}
    if args.matched_recordings is not None:
        cohort = pd.read_csv(args.matched_recordings)
        if cohort.file.duplicated().any():
            raise ValueError("Explicit comparison cohort contains duplicate filenames")
        common = set(cohort.file)
        for name, df in indices.items():
            missing = common - set(df.file)
            if missing:
                raise ValueError(f"Explicit cohort has recordings absent from {name}: {sorted(missing)}")
    else:
        common = set.intersection(*(set(df["file"]) for df in indices.values()))
    if not common:
        raise ValueError("No common recordings across conditions")
    selected = sorted(common)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "model": MODEL_CONFIG, "conditions_for_matching": index_names, "recordings": selected,
        "seeds": args.seeds, "folds": args.folds, "feature_device": str(device),
        "feature_batch_size": args.feature_batch_size, "feature_batching": "equal_sample_counts_without_padding",
        "linear_classifier_device": "cpu", "snapshot": str(snapshot.resolve()),
        "source_indices_sha256": {name: hashlib.sha256((CONDITIONS[name] / "index.csv").read_bytes()).hexdigest()
                                  for name in index_names},
        "splits_sha256": hashlib.sha256(experiments.SPLITS_PATH.read_bytes()).hexdigest(),
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "librosa", "numpy", "pandas", "scikit-learn"]},
    }
    if args.matched_recordings is not None:
        configuration["matched_recordings_path"] = str(args.matched_recordings.resolve())
        configuration["matched_recordings_sha256"] = hashlib.sha256(args.matched_recordings.read_bytes()).hexdigest()
        configuration["preprocessing_configs_sha256"] = {
            name: hashlib.sha256((CONDITIONS[name] / "generation_config.json").read_bytes()).hexdigest()
            for name in args.conditions if (CONDITIONS[name] / "generation_config.json").exists()
        }
    config_path = output_root / "comparison_config.json"
    if args.variant_root is not None:
        configuration["variant_root"] = str(args.variant_root.resolve())
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Comparison configuration changed; use a separate --output-root")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    pd.DataFrame({"file": selected}).to_csv(output_root / "matched_recordings.csv", index=False)
    extractor = FrozenFeatures(snapshot, device, output_root, args.feature_batch_size)
    for condition in args.conditions:
        evaluation_path = output_root / condition / "hubert_baseline/evaluation.json"
        if evaluation_path.exists() and not args.features_only:
            print(f"Reusing completed {condition} evaluation", flush=True)
            continue
        df = indices[condition]
        df = df[df["file"].isin(common)].copy()
        features = extractor.extract(df, condition)
        if args.features_only:
            continue
        summary = experiments.run_experiment(
            HuBERTClassifier, lambda _: features,
            {**MODEL_CONFIG, "seeds": args.seeds, "folds": args.folds}, Path("hubert_baseline"),
            data_dir=CONDITIONS[condition], dataset=condition, experiment_root=output_root,
            recording_files=selected,
        )
        print(condition, summary.to_dict("records"), flush=True)
        write_comparison(output_root, args.conditions)
    write_comparison(output_root, args.conditions)


if __name__ == "__main__":
    main()
