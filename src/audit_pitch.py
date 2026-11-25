"""Audit original RAVDESS F0 using pYIN and current Praat filtered AC.

Writes separate diagnostic artifacts; never alters audio or training defaults.
Requires the official Praat executable, not the older Parselmouth dependency.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ser-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ser-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PRAAT_SCRIPT = Path(__file__).with_suffix(".praat")
SR, FRAME, HOP = 16000, 2048, 512
PARAMETERS = {
    "sr": SR, "pyin_frame": FRAME, "pyin_hop": HOP, "floor": 50.,
    "top": 800., "praat_step": .01, "sensitivity_top": 1200.,
    "active_rms_db": -35., "rms_frame": 512,
    "pyin_probability": .5, "praat_strength": .6, "agreement_st": .5,
    "minimum_utterance_frames": 10, "target_hz": 200.,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semitones(f0):
    return 12 * np.log2(np.maximum(f0, 1e-30) / 100.)


def describe(f0, mask, hop=HOP / SR):
    values = f0[mask & np.isfinite(f0) & (f0 > 0)]
    result = {"frames": len(values), "voiced_seconds": len(values) * hop}
    if len(values):
        st = semitones(values)
        result.update(mean_hz=float(values.mean()), median_st=float(np.median(st)),
                      mean_st=float(st.mean()), median_hz=float(100 * 2 ** (np.median(st) / 12)),
                      iqr_st=float(np.diff(np.percentile(st, [25, 75]))[0]))
    return result


def parse_pitch(path):
    """Read selected candidate 1 from the documented Praat long-text format."""
    source = Path(path).read_text()
    header = {key: float(re.search(rf"(?m)^{key} = ([^\s]+)", source)[1])
              for key in ("nx", "dx", "x1")}
    frames = re.split(r"\n    frames \[\d+\]:", source)[1:]
    if len(frames) != int(header["nx"]):
        raise ValueError("Praat frame count mismatch")
    values = [re.search(r"candidates \[1\]:\s+frequency = ([^\s]+)\s+strength = ([^\s]+)", f)
              for f in frames]
    if any(v is None for v in values):
        raise ValueError("Missing selected Praat candidate")
    f0 = np.array([float(v[1]) for v in values])
    strength = np.array([float(v[2]) for v in values])
    f0[f0 <= 0] = np.nan
    return header["x1"] + np.arange(len(f0)) * header["dx"], f0, strength


def nearest_track(track, times):
    t, f0, strength = track
    indices = np.rint((times - t[0]) / .01).astype(int)
    bounded = np.clip(indices, 0, len(t) - 1)
    valid = (indices == bounded) & (np.abs(t[bounded] - times) <= .005001)
    return np.where(valid, f0[bounded], np.nan), np.where(valid, strength[bounded], np.nan)


def measure(y, native_path, binary):
    with tempfile.TemporaryDirectory(prefix="ser-pitch-audit-") as temporary:
        base = Path(temporary)
        sf.write(base / "input.wav", y, SR, subtype="DOUBLE")
        result = subprocess.run([binary, "--run", str(PRAAT_SCRIPT), str(base / "input.wav"),
                                 str(native_path), str(base / "track")], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Praat exit {result.returncode}: {result.stdout} {result.stderr}")
        praat = {name: parse_pitch(base / f"track_{name}.Pitch")
                 for name in ("16k", "native", "top1200")}
    f0, voiced, probability = librosa.pyin(y, sr=SR, fmin=50., fmax=800.,
                                         frame_length=FRAME, hop_length=HOP)
    times = np.arange(len(f0)) * HOP / SR
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=HOP)[0]
    relative_db = 20 * np.log10(np.maximum(rms, 1e-15) / max(float(rms.max()), 1e-15))
    tracks = {"time": times, "pyin_f0": f0, "pyin_probability": probability,
              "rms_db": relative_db, "active": (relative_db >= -35) & (rms > 1e-12)}
    for name, track in praat.items():
        tracks[f"praat_{name}"], tracks[f"strength_{name}"] = nearest_track(track, times)
        tracks[f"native_times_{name}"], tracks[f"native_f0_{name}"], tracks[f"native_strength_{name}"] = track
    return tracks


def masks(tracks):
    p, q = tracks["pyin_f0"], tracks["praat_16k"]
    active = tracks["active"]
    both = np.isfinite(p) & np.isfinite(q)
    strong = active & both & (tracks["pyin_probability"] >= .5) & (tracks["strength_16k"] >= .6)
    difference = np.abs(semitones(p) - semitones(q))
    return {
        "pyin_raw": np.isfinite(p),
        "pyin_active": active & np.isfinite(p),
        **{f"pyin_prob{str(threshold).replace('.', '')}": active & np.isfinite(p) & (tracks["pyin_probability"] >= threshold)
           for threshold in (.1, .3, .5)},
        "praat_active": active & np.isfinite(q),
        "praat_strong": active & np.isfinite(q) & (tracks["strength_16k"] >= .6),
        "shared_active": active & both,
        "shared_strong": strong,
        "consensus": strong & (difference <= .5),
    }


def summary(tracks):
    selected = masks(tracks)
    p, q = tracks["pyin_f0"], tracks["praat_16k"]
    result = {name: describe(q if name.startswith("praat") or name == "consensus" else p, mask)
              for name, mask in selected.items()}
    result["total_frames"] = len(p)
    result["active_frames"] = int(tracks["active"].sum())
    for name in ("shared_active", "shared_strong"):
        delta = semitones(p[selected[name]]) - semitones(q[selected[name]])
        result[name].update(disagree_over_1st=int((np.abs(delta) > 1).sum()),
                            octave_like=int((np.abs(np.abs(delta) - 12) < .75).sum()),
                            two_octave_like=int((np.abs(np.abs(delta) - 24) < .75).sum()))
        if len(delta):
            result[name].update(median_abs_difference_st=float(np.median(np.abs(delta))),
                                p90_abs_difference_st=float(np.percentile(np.abs(delta), 90)))
    result["pyin_quiet_frames"] = int((np.isfinite(p) & ~tracks["active"]).sum())
    for name in ("native", "top1200"):
        other = tracks[f"praat_{name}"]
        common = selected["praat_active"] & np.isfinite(other)
        delta = np.abs(semitones(q[common]) - semitones(other[common]))
        result[f"sensitivity_{name}"] = {"shared_frames": len(delta), "over_1st": int((delta > 1).sum()),
                                           **({"median_abs_difference_st": float(np.median(delta))} if len(delta) else {})}
    return result


def process(task):
    row, binary, output, fingerprint = task
    source = ROOT / "data/RAVDESS" / f"Actor_{int(row['actor_id']):02d}" / f"{row['file']}.wav"
    output = Path(output)
    cache = output / "tracks" / f"{row['file']}.npz"
    diagnostic = output / "records" / f"{row['file']}.json"
    signature = {"source_sha256": sha(source), "audit_fingerprint": fingerprint}
    if cache.exists() and diagnostic.exists():
        stored = json.loads(diagnostic.read_text())
        if all(stored.get(k) == v for k, v in signature.items()) and stored.get("track_sha256") == sha(cache):
            return stored
    native, sr = sf.read(source, dtype="float64", always_2d=True)
    native = native.mean(axis=1)
    # Match the original baseline pipeline's float32 loading/resampling.
    y = librosa.resample(native.astype(np.float32), orig_sr=sr, target_sr=SR)
    tracks = measure(y, source, binary)
    np.savez_compressed(cache, **tracks)
    stored = {**row, **signature, "track_sha256": sha(cache), "sample_rate": sr,
              "measurements": summary(tracks)}
    diagnostic.write_text(json.dumps(stored, indent=2, allow_nan=False) + "\n")
    return stored


def controls(binary, output):
    """Known-frequency checks catch unit, parsing and time-alignment mistakes."""
    rng = np.random.default_rng(473)
    t = np.arange(SR) / SR
    results = []
    with tempfile.TemporaryDirectory(prefix="ser-audit-controls-") as directory:
        for name in ("sine", "harmonic", "harmonic_noise_10db", "silence"):
            for frequency in ([0] if name == "silence" else [80, 120, 200, 320, 500, 700]):
                if name == "silence":
                    y = np.zeros_like(t)
                elif name == "sine":
                    y = .2 * np.sin(2 * np.pi * frequency * t)
                else:
                    y = sum(.2 / k * np.sin(2 * np.pi * frequency * k * t + .3 * k)
                            for k in range(1, 9))
                    if name == "harmonic_noise_10db":
                        noise = rng.normal(size=len(t))
                        y += noise * np.sqrt(np.mean(y ** 2) / (10 * np.mean(noise ** 2)))
                source = Path(directory) / "control.wav"
                sf.write(source, y, SR, subtype="DOUBLE")
                tracks = measure(y, source, binary)
                interior = (tracks["time"] > .15) & (tracks["time"] < .85)
                for engine in ("pyin_f0", "praat_16k", "praat_top1200"):
                    valid = interior & np.isfinite(tracks[engine])
                    error = (np.abs(12 * np.log2(tracks[engine][valid] / frequency))
                             if frequency else np.array([]))
                    result = {"signal": name, "expected_hz": frequency, "engine": engine,
                              "voiced_fraction": float(valid.sum() / interior.sum()),
                              "median_abs_error_st": float(np.median(error)) if len(error) else None}
                    result["within_02st_and_80pct_voiced"] = bool(
                        valid.sum() >= .8 * interior.sum() and len(error) and np.median(error) <= .2
                    ) if frequency else bool(not valid.any())
                    results.append(result)
                    # Noisy signals are stress measurements, not assertions that a
                    # detector is exact; retain their errors and coverage in full.
                    if frequency and name != "harmonic_noise_10db" and not result["within_02st_and_80pct_voiced"]:
                        raise AssertionError(f"Known-pitch control failed: {result}")
                    if not frequency and valid.any():
                        raise AssertionError(f"Silence marked voiced: {result}")
    (output / "synthetic_controls.json").write_text(json.dumps(results, indent=2) + "\n")
    flagged = sum(not r["within_02st_and_80pct_voiced"] for r in results)
    print(f"Completed {len(results)} engine/signal controls; {flagged} noisy stress flags", flush=True)


def actor_summaries(records, output):
    previous = pd.read_csv(ROOT / "data/audio_variants/normalized_psola_actor_mean/actor_baselines.csv").set_index("actor_id")
    rows = []
    for actor in sorted({int(r["actor_id"]) for r in records}):
        subset = [r for r in records if int(r["actor_id"]) == actor]
        row = {"actor_id": actor, "recordings": len(subset),
               "old_baseline_hz": float(previous.loc[actor, "actor_baseline_hz"]),
               "old_shift_st": float(previous.loc[actor, "requested_shift_semitones"])}
        for method in ("pyin_raw", "pyin_active", "pyin_prob01", "pyin_prob03", "pyin_prob05", "praat_active", "praat_strong", "consensus"):
            estimates = [r["measurements"][method] for r in subset
                         if r["measurements"][method]["frames"] >= 10]
            centres = np.array([m["median_st"] for m in estimates])
            row[f"{method}_recordings"] = len(estimates)
            row[f"{method}_seconds"] = sum(m["voiced_seconds"] for m in estimates)
            if len(centres):
                centre = float(np.median(centres))
                row[f"{method}_baseline_hz"] = 100 * 2 ** (centre / 12)
                row[f"{method}_shift_st"] = 12 - centre
                # Resample utterances, never overlapping frames, to assess stability.
                rng = np.random.default_rng(473 + actor)
                bootstrap = np.median(rng.choice(centres, size=(2000, len(centres))), axis=1)
                low, high = np.percentile(12 - bootstrap, [2.5, 97.5])
                row[f"{method}_shift_ci_low_st"] = low
                row[f"{method}_shift_ci_high_st"] = high
        row["shared_active_frames"] = sum(r["measurements"]["shared_active"]["frames"] for r in subset)
        row["shared_over_1st"] = sum(r["measurements"]["shared_active"]["disagree_over_1st"] for r in subset)
        row["octave_like_frames"] = sum(r["measurements"]["shared_active"]["octave_like"] for r in subset)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "actor_baselines.csv", index=False)
    return frame


def review_plots(records, output):
    examples = [r for r in records if r["file"].split("-")[2:6] == ["03", "01", "01", "01"]]
    worst = sorted(records, key=lambda r: r["measurements"]["shared_active"].get("disagree_over_1st", 0), reverse=True)[:12]
    chosen = {r["file"]: r for r in examples + worst}
    review = []
    for file, record in chosen.items():
        tr = dict(np.load(output / "tracks" / f"{file}.npz"))
        source = ROOT / "data/RAVDESS" / f"Actor_{int(record['actor_id']):02d}" / f"{file}.wav"
        y, sr = sf.read(source)
        if y.ndim > 1:
            y = y.mean(axis=1)
        selected = masks(tr)
        discrepancy = np.abs(semitones(tr["pyin_f0"]) - semitones(tr["praat_16k"]))
        possible = np.flatnonzero(selected["shared_active"] & (tr["strength_16k"] >= .6))
        if not len(possible):
            possible = np.flatnonzero(selected["shared_active"])
        if not len(possible):
            continue
        # Inspect a disagreement when present, otherwise an agreed voiced point.
        index = possible[np.nanargmax(discrepancy[possible])]
        centre = float(tr["time"][index])
        p, q = float(tr["pyin_f0"][index]), float(tr["praat_16k"][index])
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
        spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=240))
        db = librosa.amplitude_to_db(spectrum, ref=np.max)
        axes[0, 0].pcolormesh(np.arange(db.shape[1]) * 240 / sr,
                              np.fft.rfftfreq(2048, 1 / sr), db, vmin=-65, vmax=0,
                              shading="auto", cmap="gray_r")
        for key, label, color in (("pyin_f0", "pYIN", "tab:orange"), ("praat_16k", "Praat filtered AC", "tab:blue")):
            axes[0, 0].plot(tr["time"], tr[key], ".", ms=3, label=label, color=color)
        axes[0, 0].axvline(centre, color="red", lw=.8)
        axes[0, 0].set(ylim=(40, 1000), xlabel="Time (s)", ylabel="Frequency (Hz)", title=file)
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].plot(tr["time"], tr["pyin_probability"], label="pYIN voiced probability")
        axes[0, 1].plot(tr["time"], tr["strength_16k"], label="Praat AC strength (not probability)")
        axes[0, 1].plot(tr["time"], (tr["rms_db"] + 60) / 60, alpha=.5, label="Relative RMS: -60 to 0 dB")
        axes[0, 1].axvline(centre, color="red", lw=.8)
        axes[0, 1].set(ylim=(-.05, 1.05), xlabel="Time (s)", title="Confidence and activity")
        axes[0, 1].legend(fontsize=8)
        lo, hi = max(0, round((centre - .06) * sr)), min(len(y), round((centre + .06) * sr))
        window = y[lo:hi]
        nfft = 2 ** int(np.ceil(np.log2(len(window) * 8)))
        amplitude = np.abs(np.fft.rfft((window - window.mean()) * np.hanning(len(window)), n=nfft))
        freq = np.fft.rfftfreq(nfft, 1 / sr)
        axes[1, 0].plot(freq, 20 * np.log10(np.maximum(amplitude, 1e-12) / max(amplitude.max(), 1e-12)), color="black", lw=.8)
        for f0, color, label in ((q, "tab:blue", f"Praat {q:.1f} Hz"), (p, "tab:orange", f"pYIN {p:.1f} Hz")):
            for harmonic in range(1, int(1500 / f0) + 1):
                axes[1, 0].axvline(harmonic * f0, color=color, alpha=.3, label=label if harmonic == 1 else None)
        axes[1, 0].set(xlim=(0, 1500), ylim=(-65, 3), xlabel="Frequency (Hz)", ylabel="Relative dB", title=f"120-ms spectrum at {centre:.3f}s: harmonic grids")
        axes[1, 0].legend(fontsize=8)
        half_width = max(.015, 2 / min(p, q))
        lo, hi = max(0, round((centre - half_width) * sr)), min(len(y), round((centre + half_width) * sr))
        axes[1, 1].plot((np.arange(lo, hi) / sr - centre) * 1000, y[lo:hi], color="black", lw=.8)
        for f0, color, label in ((q, "tab:blue", "Praat periods"), (p, "tab:orange", "pYIN periods")):
            for k in range(-int(half_width * f0), int(half_width * f0) + 1):
                axes[1, 1].axvline(k * 1000 / f0, color=color, alpha=.35, label=label if k == 0 else None)
        axes[1, 1].set(xlabel="Time relative to selected frame (ms)", title="Waveform: inspect repetition, not individual peaks")
        axes[1, 1].legend(fontsize=8)
        destination = output / "review" / f"{file}.png"
        fig.savefig(destination, dpi=130)
        plt.close(fig)
        review.append({"file": file, "time": centre, "pyin_hz": p, "praat_hz": q,
                       "difference_st": float(discrepancy[index]), "plot": str(destination.relative_to(output)),
                       "verdict": "requires inspection; neither detector is ground truth"})
    pd.DataFrame(review).to_csv(output / "review_windows.csv", index=False)


def report(records, actors, output, config):
    totals = {key: sum(r["measurements"][key] if isinstance(r["measurements"][key], int) else 0 for r in records)
              for key in ("total_frames", "active_frames", "pyin_quiet_frames")}
    for group in ("shared_active", "shared_strong"):
        totals[group] = {key: sum(r["measurements"][group][key] for r in records)
                         for key in ("frames", "disagree_over_1st", "octave_like", "two_octave_like")}
    totals["recordings"] = len(records)
    totals["actors"] = len(actors)
    (output / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    rows = []
    for r in records:
        flat = {k: r[k] for k in ("file", "actor_id", "emotion", "intensity", "statement", "repetition")}
        for method, values in r["measurements"].items():
            if isinstance(values, dict):
                flat.update({f"{method}_{k}": v for k, v in values.items()})
        rows.append(flat)
    pd.DataFrame(rows).to_csv(output / "recordings.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    for method, label in (("pyin_raw", "pYIN raw"), ("pyin_prob05", "pYIN probability >= 0.5"),
                          ("praat_strong", "Praat active + strength >= 0.6"), ("consensus", "Strong agreement within 0.5 ST")):
        axes[0].plot(actors.actor_id, actors[f"{method}_baseline_hz"], ".-", label=label)
        axes[1].plot(actors.actor_id, actors[f"{method}_shift_st"], ".-", label=label)
    axes[0].plot(actors.actor_id, actors.old_baseline_hz, "k--", label="Old mean of utterance means in Hz")
    axes[0].set(ylabel="Actor centre (Hz)", title="Median of utterance log-F0 medians; >=10 selected frames per utterance")
    axes[1].set(ylabel="Shift to 200 Hz (semitones)", xlabel="Actor", xticks=actors.actor_id)
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output / "actor_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--praat", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/pitch_validation/f0_audit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-controls", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    binary = str(args.praat.resolve())
    version = subprocess.run([binary, "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if not version:
        raise ValueError("Praat did not report its version")
    output = args.output_dir.resolve()
    for directory in ("tracks", "records", "review"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    config = {"parameters": PARAMETERS, "praat_version": version, "praat_sha256": sha(binary),
              "python_code_sha256": sha(__file__), "praat_script_sha256": sha(PRAAT_SCRIPT),
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "librosa", "soundfile", "pandas", "numba")}}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Audit configuration changed; choose a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if not args.skip_controls:
        controls(binary, output)
    index = pd.read_csv(ROOT / "data/RAVDESS/index.csv", dtype={"statement": str, "repetition": str}).sort_values("file")
    if args.limit:
        # A smoke run covers different actors instead of only Actor 1.
        index = index.groupby("actor_id", sort=True).head(1).head(args.limit)
    tasks = [(row, binary, str(output), signature) for row in index.to_dict("records")]
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(process, tasks), 1):
            records.append(record)
            if i % 24 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} recordings; {time.perf_counter() - started:.1f}s", flush=True)
    actors = actor_summaries(records, output)
    report(records, actors, output, config)
    review_plots(records, output)
    print(f"Audit saved to {output}", flush=True)


if __name__ == "__main__":
    main()
