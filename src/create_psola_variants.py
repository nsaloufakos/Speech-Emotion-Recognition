"""Generate a separately indexed, resumable PSOLA normalization condition."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ser-numba")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from create_audio_variants import CONFIG
from pitch_processing import psola_pitch_shift

ROOT = Path(__file__).resolve().parents[1]


def estimate(audio):
    f0, voiced, probability = librosa.pyin(
        audio, sr=CONFIG["sample_rate"], fmin=CONFIG["f0_min"], fmax=CONFIG["f0_max"],
        frame_length=2048, hop_length=512,
    )
    return f0, voiced & np.isfinite(f0), probability


def process_recording(task):
    row, source_root, output_root = task
    actor = f"Actor_{int(row['actor_id']):02d}"
    filename = f"{row['file']}.wav"
    source = Path(source_root) / actor / filename
    destination = Path(output_root) / actor / filename
    diagnostic = {"file": row["file"], "actor_id": int(row["actor_id"]), "emotion": row["emotion"],
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    native, native_sr = librosa.load(source, sr=None, mono=True)
    original = librosa.resample(native, orig_sr=native_sr, target_sr=CONFIG["sample_rate"])
    f0, valid, probability = estimate(original)
    diagnostic.update(original_voiced_frames=int(valid.sum()), source_sample_rate=native_sr)
    if valid.sum() < 3:
        return {**diagnostic, "status": "excluded", "reason": "fewer_than_three_pyin_voiced_frames"}
    mean = float(f0[valid].mean())
    steps = float(12 * np.log2(CONFIG["target_f0"] / mean))
    diagnostic.update(original_mean_f0_hz=mean, applied_shift_semitones=steps,
                      original_mean_voicing_probability=float(probability[valid].mean()))
    try:
        shifted = psola_pitch_shift(native, native_sr, steps, CONFIG["f0_min"], CONFIG["f0_max"])
    except ValueError as error:
        if str(error) != "Praat found no voiced pitch points":
            raise
        return {**diagnostic, "status": "excluded", "reason": "no_praat_pitch_points"}
    processed = librosa.resample(shifted, orig_sr=native_sr, target_sr=CONFIG["sample_rate"])
    if processed.shape != original.shape or not np.isfinite(processed).all():
        raise ValueError(f"Invalid output for {filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.wav")
    sf.write(temporary, processed, CONFIG["sample_rate"], subtype="FLOAT")
    temporary.replace(destination)
    saved, saved_sr = sf.read(destination, dtype="float64")
    if saved_sr != CONFIG["sample_rate"] or not np.array_equal(saved, processed.astype(np.float32).astype(np.float64)):
        raise ValueError(f"Saved audio verification failed for {filename}")
    out_f0, out_valid, out_probability = estimate(saved)
    shared = valid & out_valid
    diagnostic.update(
        status="ok", duration_seconds=len(saved) / saved_sr,
        output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        output_mean_f0_hz=float(out_f0[out_valid].mean()) if out_valid.any() else None,
        output_voiced_frames=int(out_valid.sum()), shared_voiced_frames=int(shared.sum()),
        output_mean_voicing_probability=float(out_probability[out_valid].mean()) if out_valid.any() else None,
        shared_median_shift_semitones=float(np.median(12 * np.log2(out_f0[shared] / f0[shared]))) if shared.any() else None,
        output_peak=float(np.max(np.abs(saved))),
    )
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/audio_variants/normalized_psola")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = ROOT / "data/RAVDESS"
    index_path = source_root / "index.csv"
    index = pd.read_csv(index_path).sort_values("file")
    if args.limit is not None:
        index = index.head(args.limit)
    configuration = {
        "method": "pyin_arithmetic_mean_16k_then_psola_at_native_rate_then_resample_16k",
        "config": CONFIG, "frame_length": 2048, "hop_length": 512, "praat_time_step": .01,
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "selected_files": index["file"].tolist(),
        "versions": {p: importlib.metadata.version(p) for p in ["librosa", "numpy", "praat-parselmouth", "soundfile", "numba", "soxr"]},
    }
    config_path = output / "generation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Existing generation config differs; select a new output directory")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    diagnostics_path = output / "diagnostics.jsonl"
    completed = {}
    if diagnostics_path.exists():
        for line in diagnostics_path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["file"]] = item
    rows = index.to_dict("records")
    tasks = []
    for row in rows:
        item = completed.get(row["file"])
        source = source_root / f"Actor_{int(row['actor_id']):02d}" / f"{row['file']}.wav"
        destination = output / f"Actor_{int(row['actor_id']):02d}" / f"{row['file']}.wav"
        valid_cache = item is not None and item["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        if valid_cache and item["status"] == "ok":
            valid_cache = destination.exists() and item["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
        if not valid_cache:
            tasks.append((row, str(source_root), str(output)))
    started = time.perf_counter()
    print(f"Generating {len(tasks)} recordings; reusing {len(rows) - len(tasks)} verified results", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, diagnostics_path.open("a") as log:
        for count, result in enumerate(pool.map(process_recording, tasks), start=1):
            completed[result["file"]] = result
            log.write(json.dumps(result, allow_nan=False) + "\n")
            log.flush()
            if count % 24 == 0 or count == len(tasks):
                print(f"{count}/{len(tasks)} processed | {(time.perf_counter() - started):.1f}s", flush=True)
    included = []
    for row in rows:
        result = completed[row["file"]]
        if result["status"] == "ok":
            included.append({**row, "duration_seconds": round(result["duration_seconds"], 6)})
    result_index = pd.DataFrame(included, columns=index.columns)
    result_index["class_count"] = result_index.groupby("emotion")["emotion"].transform("size")
    result_index["actor_count"] = result_index.groupby("actor_id")["actor_id"].transform("size")
    result_index.to_csv(output / "index.csv", index=False)
    diagnostics = pd.DataFrame([completed[row["file"]] for row in rows])
    diagnostics.to_csv(output / "diagnostics.csv", index=False)
    exclusions = diagnostics[diagnostics["status"] != "ok"]
    exclusions.to_csv(output / "exclusions.csv", index=False)
    summary = {"input_recordings": len(rows), "included": len(included), "excluded": len(exclusions),
               "excluded_files": exclusions["file"].tolist(), "generation_seconds_this_run": time.perf_counter() - started}
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
