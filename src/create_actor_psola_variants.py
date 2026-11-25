"""Create batch speaker-normalized PSOLA datasets, with full and capped shifts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from create_psola_variants import CONFIG, ROOT, estimate
from pitch_processing import psola_pitch_shift

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def generate(task):
    row, input_diagnostic, variants = task
    actor = f"Actor_{int(row['actor_id']):02d}"
    filename = f"{row['file']}.wav"
    source = ROOT / "data/RAVDESS" / actor / filename
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != input_diagnostic["source_sha256"]:
        raise ValueError(f"Source changed since F0 estimation: {source}")
    native, native_sr = librosa.load(source, sr=None, mono=True)
    results = []
    for variant in variants:
        destination = Path(variant["output_dir"]) / actor / filename
        item = {"file": row["file"], "actor_id": int(row["actor_id"]), "emotion": row["emotion"],
                "source_sha256": source_hash, "source_sample_rate": native_sr,
                "actor_baseline_hz": variant["baseline_hz"],
                "unclipped_actor_shift_semitones": variant["requested_shift"],
                "applied_shift_semitones": variant["applied_shift"],
                "original_utterance_mean_f0_hz": input_diagnostic.get("original_mean_f0_hz"),
                "variant": variant["name"]}
        try:
            shifted = psola_pitch_shift(native, native_sr, variant["applied_shift"], CONFIG["f0_min"], CONFIG["f0_max"])
        except ValueError as error:
            if str(error) != "Praat found no voiced pitch points":
                raise
            results.append({**item, "status": "excluded", "reason": "no_praat_pitch_points"})
            continue
        processed = librosa.resample(shifted, orig_sr=native_sr, target_sr=CONFIG["sample_rate"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.wav")
        sf.write(temporary, processed, CONFIG["sample_rate"], subtype="FLOAT")
        temporary.replace(destination)
        saved, sr = sf.read(destination, dtype="float64")
        expected_count = int(np.ceil(len(native) * CONFIG["sample_rate"] / native_sr))
        if sr != CONFIG["sample_rate"] or len(saved) != expected_count or not np.array_equal(saved, processed.astype(np.float32).astype(np.float64)):
            raise ValueError(f"Saved audio verification failed: {destination}")
        f0, valid, probability = estimate(saved)
        item.update(status="ok", duration_seconds=len(saved) / sr,
                    output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    output_mean_f0_hz=float(f0[valid].mean()) if valid.any() else None,
                    output_voiced_frames=int(valid.sum()),
                    output_mean_voicing_probability=float(probability[valid].mean()) if valid.any() else None,
                    output_peak=float(np.max(np.abs(saved))))
        results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Smoke-run limit; use a separate output root")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/audio_variants")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    source_index = ROOT / "data/RAVDESS/index.csv"
    diagnostic_path = ROOT / "data/audio_variants/normalized_psola/diagnostics.csv"
    index = pd.read_csv(source_index).sort_values("file")
    diagnostics = pd.read_csv(diagnostic_path)
    # These estimates came from original audio; no normalized pitch or labels enter the baseline.
    valid_estimates = diagnostics[(diagnostics.original_voiced_frames >= 3) & diagnostics.original_mean_f0_hz.notna()]
    baselines = valid_estimates.groupby("actor_id").agg(
        actor_baseline_hz=("original_mean_f0_hz", "mean"),
        baseline_recordings=("file", "size"),
        median_utterance_mean_hz=("original_mean_f0_hz", "median"),
    )
    if set(baselines.index) != set(index.actor_id):
        raise ValueError("An actor has no usable baseline estimates")
    baselines["requested_shift_semitones"] = 12 * np.log2(CONFIG["target_f0"] / baselines.actor_baseline_hz)
    for row in index.itertuples():
        source = ROOT / "data/RAVDESS" / f"Actor_{row.actor_id:02d}" / f"{row.file}.wav"
        estimate_row = diagnostics[diagnostics.file == row.file]
        if len(estimate_row) != 1 or hashlib.sha256(source.read_bytes()).hexdigest() != estimate_row.iloc[0].source_sha256:
            raise ValueError(f"Missing or stale original F0 diagnostics for {row.file}")
    selected_index = index.head(args.limit) if args.limit is not None else index
    variants = {"normalized_psola_actor_mean": None, "normalized_psola_actor_mean_cap4": 4.}
    states = {}
    for name, cap in variants.items():
        output = args.output_root.resolve() / name
        output.mkdir(parents=True, exist_ok=True)
        config = {
            "scope": "batch_speaker_normalization", "baseline": "arithmetic_mean_of_valid_utterance_mean_f0",
            "baseline_source": "all_available_original_utterances_for_each_actor_including_test_batch",
            "labels_used_for_baseline": False, "requires_batch_or_prior_speaker_calibration": True,
            "target_f0_hz": CONFIG["target_f0"], "max_abs_shift_semitones": cap,
            "baseline_estimator": {"sample_rate": 16000, "fmin": 50., "fmax": 800., "frame_length": 2048, "hop_length": 512},
            "praat_time_step": .01, "synthesis": "native_rate_psola_then_resample_16k",
            "source_index_sha256": hashlib.sha256(source_index.read_bytes()).hexdigest(),
            "source_diagnostics_sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
            "psola_code_sha256": hashlib.sha256((ROOT / "src/pitch_processing.py").read_bytes()).hexdigest(),
            "selected_files": selected_index.file.tolist(),
            "versions": {p: importlib.metadata.version(p) for p in ["librosa", "numpy", "praat-parselmouth", "soundfile"]},
        }
        config_path = output / "generation_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f"Changed configuration for {name}; select another output root")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        actors = baselines.copy()
        actors["applied_shift_semitones"] = actors.requested_shift_semitones if cap is None else actors.requested_shift_semitones.clip(-cap, cap)
        actors["idealized_actor_mean_after_shift_hz"] = actors.actor_baseline_hz * 2 ** (actors.applied_shift_semitones / 12)
        actors.to_csv(output / "actor_baselines.csv")
        log_path = output / "diagnostics.jsonl"
        completed = {}
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if line.strip():
                    item = json.loads(line)
                    completed[item["file"]] = item
        states[name] = {"output": output, "actors": actors, "completed": completed, "log_path": log_path}
    inputs = diagnostics.set_index("file").to_dict("index")
    tasks = []
    for row in selected_index.to_dict("records"):
        needed = []
        for name, state in states.items():
            cached = state["completed"].get(row["file"])
            destination = state["output"] / f"Actor_{int(row['actor_id']):02d}" / f"{row['file']}.wav"
            reusable = cached is not None and cached["source_sha256"] == inputs[row["file"]]["source_sha256"]
            if reusable and cached["status"] == "ok":
                reusable = destination.exists() and cached["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
            if not reusable:
                actor = state["actors"].loc[row["actor_id"]]
                needed.append({"name": name, "output_dir": str(state["output"]),
                               "baseline_hz": float(actor.actor_baseline_hz),
                               "requested_shift": float(actor.requested_shift_semitones),
                               "applied_shift": float(actor.applied_shift_semitones)})
        if needed:
            # Convert missing original means to JSON-compatible nulls for excluded pYIN inputs.
            input_item = {key: None if pd.isna(value) else value for key, value in inputs[row["file"]].items()}
            tasks.append((row, input_item, needed))
    print(f"Processing {len(tasks)} original recordings for {len(states)} actor-level conditions", flush=True)
    started = time.perf_counter()
    logs = {name: state["log_path"].open("a") for name, state in states.items()}
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for count, results in enumerate(pool.map(generate, tasks), start=1):
                for item in results:
                    name = item["variant"]
                    states[name]["completed"][item["file"]] = item
                    logs[name].write(json.dumps(item, allow_nan=False) + "\n")
                    logs[name].flush()
                if count % 24 == 0 or count == len(tasks):
                    print(f"{count}/{len(tasks)} recordings | {time.perf_counter() - started:.1f}s", flush=True)
    finally:
        for log in logs.values():
            log.close()
    for name, state in states.items():
        rows = []
        for row in selected_index.to_dict("records"):
            item = state["completed"][row["file"]]
            if item["status"] == "ok":
                rows.append({**row, "duration_seconds": round(item["duration_seconds"], 6)})
        result_index = pd.DataFrame(rows, columns=index.columns)
        result_index["class_count"] = result_index.groupby("emotion")["emotion"].transform("size")
        result_index["actor_count"] = result_index.groupby("actor_id")["actor_id"].transform("size")
        result_index.to_csv(state["output"] / "index.csv", index=False)
        table = pd.DataFrame([state["completed"][name] for name in selected_index.file])
        table.to_csv(state["output"] / "diagnostics.csv", index=False)
        exclusions = table[table.status != "ok"]
        exclusions.to_csv(state["output"] / "exclusions.csv", index=False)
        summary = {"input_recordings": len(selected_index), "included": len(result_index), "excluded": len(exclusions),
                   "actor_count": len(baselines), "scope": "all_utterance_batch_speaker_normalization",
                   "elapsed_seconds_this_run": time.perf_counter() - started}
        (state["output"] / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(name, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
