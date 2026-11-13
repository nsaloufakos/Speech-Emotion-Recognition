"""Compare pitch/voicing in original, historical, and freshly normalized audio."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ser-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ser-numba")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ser-cache")

import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from create_audio_variants import CONFIG, normalize_f0

ROOT = Path(__file__).resolve().parents[1]
FRAME_LENGTH = 2048
HOP_LENGTH = 512


def estimate(audio):
    f0, voiced, probability = librosa.pyin(
        audio, sr=CONFIG["sample_rate"], fmin=CONFIG["f0_min"],
        fmax=CONFIG["f0_max"], frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
    )
    return f0, voiced & np.isfinite(f0), probability


def selected_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def compare(original, processed, expected_steps):
    original_f0, original_valid, original_probability = original
    processed_f0, processed_valid, processed_probability = processed
    shared = original_valid & processed_valid
    gained = ~original_valid & processed_valid
    lost = original_valid & ~processed_valid
    shifts = 12 * np.log2(processed_f0[shared] / original_f0[shared])
    return {
        "original_voiced_frames": int(original_valid.sum()),
        "processed_voiced_frames": int(processed_valid.sum()),
        "shared_voiced_frames": int(shared.sum()),
        "new_voiced_frames": int(gained.sum()),
        "lost_voiced_frames": int(lost.sum()),
        "original_mean_hz": selected_mean(original_f0, original_valid),
        "processed_mean_hz": selected_mean(processed_f0, processed_valid),
        "shared_original_mean_hz": selected_mean(original_f0, shared),
        "shared_processed_mean_hz": selected_mean(processed_f0, shared),
        "new_processed_mean_hz": selected_mean(processed_f0, gained),
        "new_processed_mean_voicing_probability": selected_mean(processed_probability, gained),
        "shared_processed_mean_voicing_probability": selected_mean(processed_probability, shared),
        "shared_original_mean_voicing_probability": selected_mean(original_probability, shared),
        "shared_shift_median_semitones": float(np.median(shifts)) if shifts.size else None,
        "shared_shift_10th_90th_percentiles": np.percentile(shifts, [10, 90]).tolist() if shifts.size else None,
        "shared_shift_median_error_semitones": float(np.median(shifts - expected_steps)) if shifts.size else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="03-01-03-01-01-01-03.wav")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    filename = Path(args.file).name
    actor = int(Path(filename).stem.split("-")[-1])
    actor_dir = f"Actor_{actor:02d}"
    output = args.output_dir or ROOT / "analysis/pitch_validation" / f"investigation_{Path(filename).stem}"
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "original": ROOT / "data/RAVDESS" / actor_dir / filename,
        "saved": ROOT / "data/audio_variants/normalized_only" / actor_dir / filename,
    }
    audio = {name: librosa.load(path, sr=CONFIG["sample_rate"], mono=True)[0]
             for name, path in paths.items()}
    print("Generating fresh normalization using normalize_f0()...", flush=True)
    fresh = normalize_f0(audio["original"])
    if fresh is None:
        raise ValueError("Original recording has fewer than three valid voiced frames")
    fresh_path = output / "fresh_normalized.wav"
    sf.write(fresh_path, fresh, CONFIG["sample_rate"], subtype="FLOAT")
    audio["fresh"] = librosa.load(fresh_path, sr=CONFIG["sample_rate"], mono=True)[0]
    if len({len(a) for a in audio.values()}) != 1:
        raise ValueError("Sample counts differ; shared frame positions require aligned recordings")

    tracks = {}
    for name, waveform in audio.items():
        print(f"Estimating {name} pitch...", flush=True)
        tracks[name] = estimate(waveform)
    original_f0, original_valid, _ = tracks["original"]
    steps = float(12 * np.log2(CONFIG["target_f0"] / original_f0[original_valid].mean()))
    summary = {
        "file": filename,
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "scipy", "librosa", "soundfile", "soxr", "numba", "pandas", "matplotlib"]},
        "config": CONFIG,
        "frame_length": FRAME_LENGTH,
        "hop_length": HOP_LENGTH,
        "input_paths": {name: str(path) for name, path in paths.items()},
        "duration_seconds": len(audio["original"]) / CONFIG["sample_rate"],
        "current_formula_shift_semitones": steps,
        "historical_applied_shift": "Unknown; current formula is not a recovered historical shift",
        "comparisons": {name: compare(tracks["original"], tracks[name], steps)
                        for name in ["saved", "fresh"]},
        "saved_vs_fresh_waveform_max_absolute_difference": float(np.max(np.abs(audio["saved"] - audio["fresh"]))),
        "limits": "pYIN estimates and shared voiced frames are not independent ground truth. Frame centers assume sample alignment; windows overlap.",
    }

    times = librosa.frames_to_time(np.arange(len(original_f0)), sr=CONFIG["sample_rate"], hop_length=HOP_LENGTH)
    columns = {"frame": np.arange(len(times)), "time_seconds": times}
    for name, (f0, valid, probability) in tracks.items():
        columns[f"{name}_f0_hz"] = f0
        columns[f"{name}_voiced"] = valid
        columns[f"{name}_voicing_probability"] = probability
        if name != "original":
            columns[f"{name}_new_voiced"] = ~original_valid & valid
            columns[f"{name}_lost_voiced"] = original_valid & ~valid
            shared = original_valid & valid
            shift = np.full(len(times), np.nan)
            shift[shared] = 12 * np.log2(f0[shared] / original_f0[shared])
            columns[f"{name}_shared_shift_semitones"] = shift
    pd.DataFrame(columns).to_csv(output / "frames.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for ax, name in zip(axes[:2], ["saved", "fresh"]):
        f0, valid, _ = tracks[name]
        gained = ~original_valid & valid
        ax.plot(times, original_f0, ".-", label="Original voiced F0", alpha=.7)
        ax.plot(times, original_f0 * 2 ** (steps / 12), "--", label="Original F0 × requested ratio", alpha=.7)
        ax.plot(times, f0, ".-", label=f"{name.capitalize()} voiced F0", alpha=.7)
        ax.scatter(times[gained], f0[gained], facecolors="none", edgecolors="red", s=70,
                   label="Voiced only after processing", zorder=5)
        ax.axhline(CONFIG["target_f0"], color="black", linestyle=":", label="200 Hz target")
        ax.set(title=f"Original vs {name} normalized recording", ylabel="Estimated F0 (Hz)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=.2)
    for name, (_, _, probability) in tracks.items():
        axes[2].plot(times, probability, label=name.capitalize())
    axes[2].set(xlabel="Frame center time (seconds)", ylabel="pYIN voiced probability",
                title="Detector probability (distinct from the final voiced/unvoiced path)", ylim=(0, 1))
    axes[2].legend()
    axes[2].grid(alpha=.2)
    fig.savefig(output / "pitch_tracks.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Artifacts saved to {output}")


if __name__ == "__main__":
    main()
