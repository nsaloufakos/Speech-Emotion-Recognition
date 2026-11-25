"""Remeasure every generated robust-median recording with current Praat."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
import time

from audit_pitch import ROOT, SR, HOP, parse_pitch, nearest_track, semitones, sha
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

CONDITIONS = ("normalized_psola_actor_praat_median", "normalized_psola_actor_praat_median_cap4")
SCRIPT = ROOT / "src/measure_praat_pitch.praat"


def process(task):
    row, condition, binary, audit, output, method = task
    source = ROOT / "data/audio_variants" / condition / f"Actor_{int(row['actor_id']):02d}" / f"{row['file']}.wav"
    destination = Path(output) / condition / "records" / f"{row['file']}.json"
    cache = destination.with_suffix(".npz")
    signature = {"source_sha256": sha(source), "original_track_sha256": sha(Path(audit) / "tracks" / f"{row['file']}.npz"),
                 "requested_shift_st": float(row["applied_shift_semitones"]),
                 "expected_actor_baseline_hz": float(row["idealized_actor_baseline_after_shift_hz"]), **method}
    if destination.exists() and cache.exists():
        previous = json.loads(destination.read_text())
        if all(previous.get(k) == v for k, v in signature.items()) and previous.get("track_sha256") == sha(cache):
            return previous
    y, sr = sf.read(source, dtype="float64")
    if sr != SR or y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError(f"Invalid generated waveform: {source}")
    with tempfile.TemporaryDirectory(prefix="ser-validate-praat-") as temporary:
        pitch = Path(temporary) / "out.Pitch"
        result = subprocess.run([binary, "--run", str(SCRIPT), str(source), str(pitch)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Praat failed: {result.stdout} {result.stderr}")
        raw = parse_pitch(pitch)
    original = dict(np.load(Path(audit) / "tracks" / f"{row['file']}.npz"))
    times = original["time"]
    f0, strength = nearest_track(raw, times)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=HOP)[0]
    if len(rms) != len(times):
        raise ValueError(f"Original/output frame count differs: {source}")
    active = (rms >= rms.max() * 10 ** (-35 / 20)) & (rms > 1e-12)
    valid = active & np.isfinite(f0) & (strength >= .6)
    original_valid = original["active"] & np.isfinite(original["praat_16k"]) & (original["strength_16k"] >= .6)
    shared = original_valid & np.isfinite(f0) & (strength >= .6)
    requested = float(row["applied_shift_semitones"])
    changes = semitones(f0[shared]) - semitones(original["praat_16k"][shared])
    error = changes - requested
    stored = {"file": row["file"], "actor_id": int(row["actor_id"]), "condition": condition, **signature,
              "requested_shift_st": requested, "output_active_strong_frames": int(valid.sum()),
              "original_active_strong_frames": int(original_valid.sum()), "shared_strong_frames": int(shared.sum()),
              "expected_actor_baseline_hz": float(row["idealized_actor_baseline_after_shift_hz"]),
              "output_peak": float(np.abs(y).max())}
    if valid.sum() >= 10:
        stored["output_utterance_median_st"] = float(np.median(semitones(f0[valid])))
    if shared.any():
        stored.update(measured_median_shift_st=float(np.median(changes)),
                      median_abs_error_st=float(np.median(np.abs(error))),
                      p90_abs_error_st=float(np.percentile(np.abs(error), 90)),
                      frames_over_1st=int((np.abs(error) > 1).sum()))
    np.savez_compressed(cache, time=times, f0=f0, strength=strength, active=active, shared=shared,
                        raw_times=raw[0], raw_f0=raw[1], raw_strength=raw[2])
    stored["track_sha256"] = sha(cache)
    destination.write_text(json.dumps(stored, indent=2, allow_nan=False) + "\n")
    return stored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--praat", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "analysis/pitch_validation/f0_audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/pitch_validation/praat_median_experiment")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=[CONDITIONS[1]])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    method = {"praat_sha256": sha(args.praat), "script_sha256": sha(SCRIPT), "python_sha256": sha(__file__)}
    for condition in args.conditions:
        (output / condition / "records").mkdir(parents=True, exist_ok=True)
        directory = ROOT / "data/audio_variants" / condition
        actors = pd.read_csv(directory / "actor_baselines.csv").set_index("actor_id")
        index = pd.read_csv(directory / "index.csv")
        for row in index.to_dict("records"):
            actor = actors.loc[row["actor_id"]]
            row.update(applied_shift_semitones=float(actor.applied_shift_semitones),
                       idealized_actor_baseline_after_shift_hz=float(actor.idealized_actor_baseline_after_shift_hz))
            tasks.append((row, condition, str(args.praat.resolve()), str(args.audit_dir.resolve()), str(output), method))
    start = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, record in enumerate(pool.map(process, tasks), 1):
            records.append(record)
            if count % 96 == 0 or count == len(tasks):
                print(f"Validated {count}/{len(tasks)} outputs | {time.perf_counter() - start:.1f}s", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "pitch_validation.csv", index=False)
    actors = frame.groupby(["condition", "actor_id"]).agg(
        output_median_st=("output_utterance_median_st", "median"),
        usable_recordings=("output_utterance_median_st", "count"),
        expected_baseline_hz=("expected_actor_baseline_hz", "first"),
        median_abs_frame_error_st=("median_abs_error_st", "median"))
    actors["measured_baseline_hz"] = 100 * 2 ** (actors.output_median_st / 12)
    actors["baseline_residual_st"] = 12 * np.log2(actors.measured_baseline_hz / actors.expected_baseline_hz)
    actors.to_csv(output / "output_actor_baselines.csv")
    summaries = []
    for condition, group in frame.groupby("condition"):
        summary = {"condition": condition, "recordings": len(group),
                   "shared_frames": int(group.shared_strong_frames.sum()),
                   "frames_over_1st": int(group.frames_over_1st.fillna(0).sum()),
                   "median_per_recording_absolute_shift_error_st": float(np.median(np.abs(group.measured_median_shift_st - group.requested_shift_st))),
                   "median_per_recording_frame_mae_st": float(group.median_abs_error_st.median()),
                   "max_actor_baseline_residual_st": float(actors.loc[condition].baseline_residual_st.abs().max())}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (output / "pitch_validation_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
