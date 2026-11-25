"""Regenerate a saved PSOLA condition without refitting its speaker baselines.

The saved diagnostics supply source hashes and constant shifts. Regeneration
writes a separate dataset with fresh file hashes: FLOAT WAV headers contain a
creation timestamp, so an identical waveform need not have the old file hash.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ser-numba")

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf

from pitch_processing import psola_pitch_shift

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "normalized_psola_actor_praat_median_cap4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(task):
    record, directory = task
    relative = Path(f"Actor_{int(record['actor_id']):02d}") / f"{record['file']}.wav"
    source = ROOT / "data/RAVDESS" / relative
    destination = Path(directory) / relative
    if sha(source) != record["source_sha256"]:
        raise ValueError(f"Original audio changed: {source}")
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output directory: {destination}")
    audio, rate = librosa.load(source, sr=None, mono=True)
    shifted = psola_pitch_shift(audio, rate, record["applied_shift_semitones"], 50., 800.)
    processed = librosa.resample(shifted, orig_sr=rate, target_sr=16000)
    if not np.isfinite(processed).all():
        raise ValueError(f"Nonfinite synthesized audio: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.wav")
    sf.write(temporary, processed, 16000, subtype="FLOAT")
    saved, saved_rate = sf.read(temporary, dtype="float64")
    if saved_rate != 16000 or not np.array_equal(saved, processed.astype(np.float32).astype(np.float64)):
        raise ValueError(f"Saved waveform verification failed: {destination}")
    temporary.replace(destination)
    return {**record, "archived_output_sha256": record["output_sha256"],
            "output_sha256": sha(destination),
            "output_samples_sha256": hashlib.sha256(saved.astype('<f4').tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default=DEFAULT,
                        help="Existing condition directory under data/audio_variants")
    parser.add_argument("--output-root", type=Path, default=ROOT / ".cache/regenerated_variants",
                        help="Fresh root for regenerated datasets; archived metadata is never overwritten")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Only process the first N successful records")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if Path(args.condition).name != args.condition:
        parser.error("condition must be a directory name, not a path")
    if parselmouth.__version__ != "0.4.7" or parselmouth.PRAAT_VERSION != "6.1.38":
        parser.error("Restoration requires Parselmouth 0.4.7 / synthesis Praat 6.1.38")
    directory = ROOT / "data/audio_variants" / args.condition
    records = pd.read_csv(directory / "diagnostics.csv", float_precision="round_trip")
    records = records[records.status == "ok"].sort_values("file")
    if args.limit is not None:
        records = records.head(args.limit)
    output = args.output_root.resolve() / args.condition
    if output.exists():
        parser.error("Output already exists; choose a fresh --output-root")
    output.mkdir(parents=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(restore, [(row, str(output)) for row in records.to_dict("records")]), 1):
            results.append(result)
            if i % 96 == 0 or i == len(records):
                print(f"Regenerated {i}/{len(records)}", flush=True)
    pd.DataFrame(results).to_csv(output / "diagnostics.csv", index=False)
    index = pd.read_csv(directory / "index.csv")
    index[index.file.isin(records.file)].to_csv(output / "index.csv", index=False)
    actors = directory / "actor_baselines.csv"
    if actors.exists():
        shutil.copyfile(actors, output / actors.name)
    config_path = directory / "generation_config.json"
    config = json.loads(config_path.read_text())
    config["selected_files"] = records.file.tolist()
    config["regeneration"] = {"archived_configuration_sha256": sha(config_path),
                              "archived_diagnostics_sha256": sha(directory / "diagnostics.csv"),
                              "code_sha256": sha(Path(__file__)),
                              "versions": {p: importlib.metadata.version(p) for p in ("librosa", "numpy", "soundfile", "praat-parselmouth")},
                              "selected_files": records.file.tolist(),
                              "note": "Uses saved shifts; fresh WAV metadata and hashes; no baseline refit"}
    (output / "generation_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
