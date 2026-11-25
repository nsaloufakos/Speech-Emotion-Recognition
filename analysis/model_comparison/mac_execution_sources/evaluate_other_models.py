"""Matched original/full/capped comparisons for the remaining P20 baselines."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ser-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ser-matplotlib")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model

import run_experiments as experiments
from training_execution import run_epoch
from mfcc_execution import DenseBiLSTM
from models.baseline_mfcc_bilstm import CONFIG as MFCC_CONFIG
from models.baseline_wav2vec_linear import Wav2VecClassifier, CONFIG as WAV_CONFIG

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "original": ROOT / "data/RAVDESS",
    "normalized_psola_actor_praat_median": ROOT / "data/audio_variants/normalized_psola_actor_praat_median",
    "normalized_psola_actor_praat_median_cap4": ROOT / "data/audio_variants/normalized_psola_actor_praat_median_cap4",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(df, condition, model, device, snapshot, output, feature_batch_size):
    cache = output / condition / "features"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "manifest.json"
    signature = {"model": model, "sample_rate": 16000,
                 "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "librosa", "numpy", "soundfile")}}
    if model == "wav2vec":
        signature.update(snapshot=str(snapshot.relative_to(ROOT)), device=str(device),
                         pooling="mean_over_final_layer_time", attention_implementation="eager",
                         batching="equal_lengths_without_padding", feature_batch_size=feature_batch_size)
    else:
        signature.update(n_mfcc=40, feature_layout="time_by_coefficient", librosa_mfcc_defaults=True)
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if manifest.get("signature") != signature:
        manifest = {"signature": signature, "recordings": {}}
    result, pending = {}, []
    for row in df.itertuples():
        source = CONDITIONS[condition] / f"Actor_{row.actor_id:02d}" / f"{row.file}.wav"
        digest = sha(source)
        stored = cache / f"{row.file}.npy"
        if manifest["recordings"].get(row.file) == digest and stored.exists():
            result[row.file] = np.load(stored)
        else:
            pending.append((row.file, source, digest))
    print(f"{model}/{condition}: {len(result)} cached features, {len(pending)} pending", flush=True)
    def save(name, value, digest):
        if not np.isfinite(value).all():
            raise ValueError(f"Invalid features: {name}")
        temporary = cache / f"{name}.partial.npy"
        np.save(temporary, value)
        temporary.replace(cache / f"{name}.npy")
        result[name] = value
        manifest["recordings"][name] = digest
    start = time.perf_counter()
    if model == "mfcc":
        for i, (name, source, digest) in enumerate(pending, 1):
            y, sr = librosa.load(source, sr=16000, mono=True)
            save(name, librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, digest)
            if i % 96 == 0 or i == len(pending):
                path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"MFCC {condition}: {i}/{len(pending)} | {time.perf_counter()-start:.1f}s", flush=True)
    elif pending:
        processor = AutoFeatureExtractor.from_pretrained(snapshot, local_files_only=True)
        backbone = Wav2Vec2Model.from_pretrained(snapshot, local_files_only=True,
                                                attn_implementation="eager").to(device).eval()
        backbone.requires_grad_(False)
        groups = {}
        for item in pending:
            info = sf.info(item[1])
            count = (info.frames * 16000 + info.samplerate - 1) // info.samplerate
            groups.setdefault(count, []).append(item)
        batches = [group[start:start+feature_batch_size] for group in groups.values()
                   for start in range(0, len(group), feature_batch_size)]
        count = 0
        with torch.inference_mode():
            for batch in batches:
                audio = [librosa.load(source, sr=16000, mono=True)[0] for _, source, _ in batch]
                if len({len(y) for y in audio}) != 1:
                    raise ValueError("Padded encoder features are not allowed")
                values = processor(audio, sampling_rate=16000, padding=False, return_tensors="pt").input_values.to(device)
                embeddings = backbone(values).last_hidden_state.mean(dim=1).cpu().numpy()
                if embeddings.shape != (len(batch), 768):
                    raise ValueError("Invalid wav2vec embedding shape")
                for (name, _, digest), embedding in zip(batch, embeddings):
                    save(name, embedding[None, :], digest)
                previous = count
                count += len(batch)
                temporary = path.with_suffix(".partial.json")
                temporary.write_text(json.dumps(manifest, indent=2) + "\n")
                temporary.replace(path)
                if count // 96 != previous // 96 or count == len(pending):
                    print(f"wav2vec {condition}: {count}/{len(pending)} | {time.perf_counter()-start:.1f}s", flush=True)
        del backbone
        if device.type == "mps":
            torch.mps.empty_cache()
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def train(task):
    model, condition, selected, output, device, configuration, threads = task
    output = Path(output)
    result = output / condition / f"{model}_baseline/evaluation.json"
    if result.exists():
        print(f"Reusing completed {model}/{condition}", flush=True)
        return condition
    torch.set_num_threads(threads)
    experiments.DEVICE = torch.device(device if model == "mfcc" else "cpu")
    experiments.run_epoch = run_epoch
    directory = output / condition / "features"
    prepared = {file: np.load(directory / f"{file}.npy") for file in selected}
    classifier = DenseBiLSTM if model == "mfcc" else Wav2VecClassifier
    experiments.run_experiment(classifier, lambda _: prepared, configuration,
                               Path(f"{model}_baseline"), data_dir=CONDITIONS[condition],
                               dataset=condition, experiment_root=output, recording_files=selected)
    return condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mfcc", "wav2vec"], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--conditions", choices=list(CONDITIONS), nargs="+", default=list(CONDITIONS))
    parser.add_argument("--matched-recordings", type=Path, default=ROOT / "output/pitch_comparison_hubert/matched_recordings.csv")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 25)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--training-workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.training_workers, args.threads, args.feature_batch_size) < 1:
        parser.error("Counts must be positive")
    torch.set_num_threads(args.threads)
    output = (args.output_root or ROOT / f"output/pitch_comparison_praat_median_{args.model}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = sorted(pd.read_csv(args.matched_recordings).file.tolist())
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate cohort filenames")
    snapshot = None
    if args.model == "wav2vec":
        snapshot = Path((ROOT / ".cache/wav2vec_model_path.txt").read_text().strip())
        if not snapshot.is_absolute():
            snapshot = ROOT / snapshot
    model_configuration = {**(MFCC_CONFIG if args.model == "mfcc" else WAV_CONFIG),
                           "epochs": args.epochs, "seeds": args.seeds, "folds": args.folds}
    configuration = {"model": args.model, "training": model_configuration, "recordings": selected,
                     "conditions": list(CONDITIONS), "device": args.device, "threads_per_worker": args.threads,
                     "classifier_device": args.device if args.model == "mfcc" else "cpu",
                     "epoch_execution": "deferred_metric_transfers", "mfcc_execution": "dense_equivalent_to_packed" if args.model == "mfcc" else None,
                     "feature_batch_size": args.feature_batch_size,
                     "snapshot": str(snapshot.relative_to(ROOT)) if snapshot else None,
                     "splits_sha256": sha(ROOT / "output/splits.csv"), "cohort_sha256": sha(args.matched_recordings),
                     "source_indices_sha256": {name: sha(path / "index.csv") for name, path in CONDITIONS.items()},
                     "preprocessing_configs_sha256": {name: sha(path / "generation_config.json") for name, path in CONDITIONS.items() if name != "original"},
                     "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "librosa", "numpy", "pandas", "scikit-learn")},
                     "code_sha256": {name: sha(ROOT / "src" / name) for name in ("evaluate_other_models.py", "training_execution.py", "mfcc_execution.py", "run_experiments.py", "models/baseline_mfcc_bilstm.py", "models/baseline_wav2vec_linear.py")}}
    path = output / "comparison_config.json"
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError("Comparison configuration changed; use another output root")
    path.write_text(json.dumps(configuration, indent=2) + "\n")
    pd.DataFrame({"file": selected}).to_csv(output / "matched_recordings.csv", index=False)
    for condition in args.conditions:
        index = pd.read_csv(CONDITIONS[condition] / "index.csv")
        if set(selected) - set(index.file):
            raise ValueError(f"Cohort missing in {condition}")
        df = index[index.file.isin(selected)].copy()
        features(df, condition, args.model, torch.device(args.device), snapshot, output, args.feature_batch_size)
    tasks = [(args.model, condition, selected, str(output), args.device, model_configuration, args.threads) for condition in args.conditions]
    if args.training_workers > 1:
        with ProcessPoolExecutor(max_workers=args.training_workers) as pool:
            list(pool.map(train, tasks))
    else:
        for task in tasks:
            train(task)
    rows = []
    for condition in CONDITIONS:
        path = output / condition / f"{args.model}_baseline/evaluation.json"
        if path.exists():
            evaluation = json.loads(path.read_text())
            rows.append({"model": args.model, "condition": condition, **evaluation["summary"]})
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
