"""Create actor-normalized PSOLA audio from audited active Praat medians.

Uses the previous PSOLA generator unchanged, isolating baseline estimation.
Both full and pre-specified +/-4-ST conditions are generated separately.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from create_actor_psola_variants import generate
from create_psola_variants import CONFIG, ROOT
from audit_pitch import semitones, sha
import numpy as np
import pandas as pd
import parselmouth

VARIANTS = {"normalized_psola_actor_praat_median": None,
            "normalized_psola_actor_praat_median_cap4": 4.}


def load_baselines(index, audit):
    config = json.loads((audit / "config.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    quality = pd.read_csv(audit / "actor_quality.csv").set_index("actor_id")
    estimates, inputs, hashes = [], {}, {}
    for row in index.itertuples():
        source = ROOT / "data/RAVDESS" / f"Actor_{row.actor_id:02d}" / f"{row.file}.wav"
        record_path = audit / "records" / f"{row.file}.json"
        record = json.loads(record_path.read_text())
        if record["audit_fingerprint"] != fingerprint or record["source_sha256"] != sha(source):
            raise ValueError(f"Stale pitch audit: {row.file}")
        track_path = audit / "tracks" / f"{row.file}.npz"
        if record["track_sha256"] != sha(track_path):
            raise ValueError(f"Changed pitch track: {row.file}")
        with np.load(track_path) as track:
            # These masks/settings are exactly those used in the reported candidate.
            valid = (track["active"] & np.isfinite(track["praat_16k"])
                     & (track["strength_16k"] >= .6))
            if valid.sum() < 10:
                raise ValueError(f"Fewer than ten selected Praat frames: {row.file}")
            centre = float(np.median(semitones(track["praat_16k"][valid])))
        estimates.append({"actor_id": row.actor_id, "file": row.file, "median_st": centre,
                          "selected_frames": int(valid.sum())})
        inputs[row.file] = {"source_sha256": record["source_sha256"],
                            "original_mean_f0_hz": record["measurements"]["pyin_raw"].get("mean_hz")}
        hashes[row.file] = sha(record_path)
    estimates = pd.DataFrame(estimates)
    actors = estimates.groupby("actor_id").agg(
        actor_baseline_st=("median_st", "median"), baseline_recordings=("file", "size"),
        selected_frames=("selected_frames", "sum"))
    actors["actor_baseline_hz"] = 100 * 2 ** (actors.actor_baseline_st / 12)
    actors["requested_shift_semitones"] = 12 * np.log2(CONFIG["target_f0"] / actors.actor_baseline_hz)
    actors["review_flags"] = quality.loc[actors.index, "review_flags"]
    actors["baseline_shift_ci_low_st"] = quality.loc[actors.index, "praat_strong_shift_ci_low_st"]
    actors["baseline_shift_ci_high_st"] = quality.loc[actors.index, "praat_strong_shift_ci_high_st"]
    if not np.allclose(actors.actor_baseline_hz, quality.loc[actors.index, "praat_strong_baseline_hz"], rtol=0, atol=1e-10):
        raise ValueError("Recomputed baselines differ from audited candidate")
    return actors, inputs, hashes, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "analysis/pitch_validation/f0_audit")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/audio_variants")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conditions", nargs="+", choices=list(VARIANTS),
                        default=["normalized_psola_actor_praat_median_cap4"])
    parser.add_argument("--limit", type=int, help="Smoke-run limit; use a separate output root")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if parselmouth.__version__ != "0.4.7" or parselmouth.PRAAT_VERSION != "6.1.38":
        raise ValueError("Use the previous synthesis engine: Parselmouth 0.4.7 / Praat 6.1.38")
    index_path = ROOT / "data/RAVDESS/index.csv"
    index = pd.read_csv(index_path).sort_values("file")
    audit = args.audit_dir.resolve()
    actors, inputs, hashes, audit_config = load_baselines(index, audit)
    selected = index.groupby("actor_id", sort=True).head(1).head(args.limit) if args.limit else index
    states = {}
    for name in args.conditions:
        cap = VARIANTS[name]
        output = args.output_root.resolve() / name
        output.mkdir(parents=True, exist_ok=True)
        configuration = {
            "scope": "batch_speaker_normalization", "baseline": "median_of_utterance_medians_in_semitones",
            "baseline_detector": "Praat_7.0.03_filtered_autocorrelation",
            "baseline_source": "all_available_original_utterances_for_each_actor_including_test_batch",
            "labels_used_for_baseline": False, "requires_batch_or_prior_speaker_calibration": True,
            "baseline_filter": {"active_rms_db": -35., "rms_window_samples": 512, "praat_strength_min": .6,
                                "minimum_frames_per_utterance": 10, "measurement_grid_seconds": .032},
            "flag_policy": "retain_all_actors_without_per_actor_tuning_and_report_flags",
            "target_f0_hz": CONFIG["target_f0"], "max_abs_shift_semitones": cap,
            "synthesis": "unchanged_native_rate_PSOLA_then_resample_16k_FLOAT_WAV",
            "synthesis_praat_version": parselmouth.PRAAT_VERSION,
            "synthesis_pitch_floor_hz": CONFIG["f0_min"], "synthesis_pitch_ceiling_hz": CONFIG["f0_max"],
            "synthesis_praat_time_step": .01, "training_gain_matching": False,
            "source_index_sha256": sha(index_path), "audit_config_sha256": sha(audit / "config.json"),
            "audit_parameters": audit_config["parameters"], "audit_records_sha256": hashes,
            "audit_quality_table_sha256": sha(audit / "actor_quality.csv"),
            "selected_files": selected.file.tolist(),
            "code_sha256": {p: sha(ROOT / "src" / p) for p in ("create_praat_median_variants.py", "create_actor_psola_variants.py", "create_psola_variants.py", "pitch_processing.py", "audit_pitch.py")},
            "versions": {p: importlib.metadata.version(p) for p in ("librosa", "numpy", "praat-parselmouth", "soundfile")},
        }
        path = output / "generation_config.json"
        if path.exists() and json.loads(path.read_text()) != configuration:
            raise ValueError(f"Generation configuration changed for {name}; choose another output root")
        path.write_text(json.dumps(configuration, indent=2) + "\n")
        baselines = actors.copy()
        baselines["applied_shift_semitones"] = baselines.requested_shift_semitones if cap is None else baselines.requested_shift_semitones.clip(-cap, cap)
        baselines["idealized_actor_baseline_after_shift_hz"] = baselines.actor_baseline_hz * 2 ** (baselines.applied_shift_semitones / 12)
        baselines.to_csv(output / "actor_baselines.csv")
        log = output / "diagnostics.jsonl"
        completed = {r["file"]: r for r in (json.loads(line) for line in log.read_text().splitlines() if line.strip())} if log.exists() else {}
        states[name] = {"output": output, "actors": baselines, "log": log, "completed": completed}
    tasks = []
    for row in selected.to_dict("records"):
        needed = []
        for name, state in states.items():
            record = state["completed"].get(row["file"])
            destination = state["output"] / f"Actor_{int(row['actor_id']):02d}" / f"{row['file']}.wav"
            reusable = record is not None and record["source_sha256"] == inputs[row["file"]]["source_sha256"]
            if reusable and record["status"] == "ok":
                reusable = destination.exists() and sha(destination) == record["output_sha256"]
            if not reusable:
                actor = state["actors"].loc[row["actor_id"]]
                needed.append({"name": name, "output_dir": str(state["output"]),
                               "baseline_hz": float(actor.actor_baseline_hz),
                               "requested_shift": float(actor.requested_shift_semitones),
                               "applied_shift": float(actor.applied_shift_semitones)})
        if needed:
            tasks.append((row, inputs[row["file"]], needed))
    print(f"Generating {len(tasks)} recordings for {len(states)} conditions", flush=True)
    start = time.perf_counter()
    logs = {name: state["log"].open("a") for name, state in states.items()}
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for count, results in enumerate(pool.map(generate, tasks), 1):
                for result in results:
                    name = result["variant"]
                    states[name]["completed"][result["file"]] = result
                    logs[name].write(json.dumps(result, allow_nan=False) + "\n")
                    logs[name].flush()
                if count % 24 == 0 or count == len(tasks):
                    print(f"{count}/{len(tasks)} | {time.perf_counter() - start:.1f}s", flush=True)
    finally:
        for log in logs.values():
            log.close()
    for name, state in states.items():
        diagnostics = pd.DataFrame([state["completed"][file] for file in selected.file])
        successful = diagnostics[diagnostics.status == "ok"].set_index("file")
        result = selected[selected.file.isin(successful.index)].copy()
        result["duration_seconds"] = result.file.map(successful.duration_seconds).round(6)
        result["class_count"] = result.groupby("emotion").emotion.transform("size")
        result["actor_count"] = result.groupby("actor_id").actor_id.transform("size")
        result.to_csv(state["output"] / "index.csv", index=False)
        diagnostics.to_csv(state["output"] / "diagnostics.csv", index=False)
        diagnostics[diagnostics.status != "ok"].to_csv(state["output"] / "exclusions.csv", index=False)
        summary = {"input_recordings": len(selected), "included": len(result), "excluded": len(selected) - len(result),
                   "actor_count": len(actors), "elapsed_seconds_this_run": time.perf_counter() - start}
        (state["output"] / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(name, summary, flush=True)


if __name__ == "__main__":
    main()
