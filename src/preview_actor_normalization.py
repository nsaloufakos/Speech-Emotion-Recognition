"""Create actor listening comparisons across three emotions and four conditions."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import librosa
import numpy as np
import soundfile as sf

CONDITIONS = {
    "original": ROOT / "data/RAVDESS",
    "utterance_psola": ROOT / "data/audio_variants/normalized_psola",
    "actor_psola": ROOT / "data/audio_variants/normalized_psola_actor_mean",
    "actor_psola_cap4": ROOT / "data/audio_variants/normalized_psola_actor_mean_cap4",
    "praat_actor_psola": ROOT / "data/audio_variants/normalized_psola_actor_praat_median",
    "praat_actor_psola_cap4": ROOT / "data/audio_variants/normalized_psola_actor_praat_median_cap4",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", nargs="+", type=int, default=[3, 5, 21])
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
                        default=["original", "praat_actor_psola_cap4"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/pitch_validation/actor_normalization_listening")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    conditions = {name: CONDITIONS[name] for name in args.conditions}
    # Normal intensity, statement 1, repetition 1 for every emotion and condition.
    emotions = {"neutral": 1, "happy": 3, "angry": 5}
    manifest = {"conditions_in_order": list(conditions), "emotions_in_each_condition": list(emotions),
                "listening_sample_rate": 16000,
                "matching": "RMS matched over the concatenated three utterances within each condition; preserves between-utterance gain differences within a condition",
                "utterance_gap_seconds": .4, "condition_gap_seconds": 1., "actors": {}}
    for actor in args.actors:
        waveforms = {}
        details = {}
        for name, root in conditions.items():
            clips = []
            paths = []
            for emotion, code in emotions.items():
                path = root / f"Actor_{actor:02d}" / f"03-01-{code:02d}-01-01-01-{actor:02d}.wav"
                audio, _ = librosa.load(path, sr=16000, mono=True)
                clips.append(audio)
                paths.append(str(path))
            waveforms[name] = clips
            details[name] = {"paths": paths}
        rms = lambda clips: float(np.sqrt(np.mean(np.concatenate(clips).astype(np.float64) ** 2)))
        target = rms(waveforms["original"])
        gains = {name: target / max(rms(clips), 1e-15) for name, clips in waveforms.items()}
        peak = max(max(float(np.abs(clip).max()) for clip in clips) * gains[name] for name, clips in waveforms.items())
        common_scale = min(1., .95 / peak)
        sequence = []
        for name, clips in waveforms.items():
            details[name]["listening_gain"] = gains[name] * common_scale
            for clip in clips:
                sequence.extend([clip * gains[name] * common_scale, np.zeros(6400)])
            sequence.append(np.zeros(16000))
        path = output / f"actor_{actor:02d}_listening_sequence.wav"
        result = np.concatenate(sequence)
        if not np.isfinite(result).all() or np.abs(result).max() >= 1:
            raise ValueError("Invalid listening sequence")
        sf.write(path, result, 16000, subtype="PCM_24")
        manifest["actors"][str(actor)] = {"conditions": details, "file": str(path), "duration_seconds": len(result) / 16000}
        print(path, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
