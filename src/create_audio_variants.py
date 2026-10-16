from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

DATA_DIR = Path("data/RAVDESS")
OUTPUT_DIR = Path("data/audio_variants")
NOISE_DIR = Path("data/background_noise")

CONFIG = {
    "sample_rate" : 16000,
    "seed" : 42,
    "target_f0" : 200.0,
    "f0_min" : 50.0,
    "f0_max" : 800.0,
    "pitch_steps" : [-1.0, 1.0],
    "stretch_rates" : [0.9, 1.1],
    "noise_snr_db" : [20.0, 30.0]
}

SETTINGS = {
    "normalized_only" : {
        "normalize" : True,
        "pitch" : False,
        "stretch" : False,
        "noise" : False
    },
    "pitch_only" : {
        "normalize" : False,
        "pitch" : True,
        "stretch" : False,
        "noise" : False
    },
    "stretch_only" : {
        "normalize" : False,
        "pitch" : False,
        "stretch" : True,
        "noise" : False
    },
    "noise_only" : {
        "normalize" : False,
        "pitch" : False,
        "stretch" : False,
        "noise" : True
    },
    "pitch_stretch" : {
        "normalize" : False,
        "pitch" : True,
        "stretch" : True,
        "noise" : False
    },
    "pitch_noise" : {
        "normalize" : False,
        "pitch" : True,
        "stretch" : False,
        "noise" : True
    },
    "stretch_noise" : {
        "normalize" : False,
        "pitch" : False,
        "stretch" : True,
        "noise" : True
    },
    "all_augmentations" : {
        "normalize" : False,
        "pitch" : True,
        "stretch" : True,
        "noise" : True
    },
    "normalized_pitch" : {
        "normalize" : True,
        "pitch" : True,
        "stretch" : False,
        "noise" : False
    },
    "normalized_stretch" : {
        "normalize" : True,
        "pitch" : False,
        "stretch" : True,
        "noise" : False
    },
    "normalized_noise" : {
        "normalize" : True,
        "pitch" : False,
        "stretch" : False,
        "noise" : True
    },
    "normalized_pitch_stretch" : {
        "normalize" : True,
        "pitch" : True,
        "stretch" : True,
        "noise" : False
    },
    "normalized_pitch_noise" : {
        "normalize" : True,
        "pitch" : True,
        "stretch" : False,
        "noise" : True
    },
    "normalized_stretch_noise" : {
        "normalize" : True,
        "pitch" : False,
        "stretch" : True,
        "noise" : True
    },
    "norm_and_aug" : {
        "normalize" : True,
        "pitch" : True,
        "stretch" : True,
        "noise" : True
    }
}
# ACTIVE_SETTINGS = list(SETTINGS)
ACTIVE_SETTINGS = ["normalized_only"]

def normalize_f0(audio):
    f0, voiced, _ = librosa.pyin(
        audio,
        sr = CONFIG["sample_rate"],
        fmin = CONFIG["f0_min"],
        fmax = CONFIG["f0_max"]
    )

    voiced_f0 = f0[voiced & np.isfinite(f0)]

    if len(voiced_f0) < 3:
        return None

    mean_f0 = voiced_f0.mean()
    steps = 12 * np.log2(CONFIG["target_f0"] / mean_f0)

    return librosa.effects.pitch_shift(
        audio,
        sr = CONFIG["sample_rate"],
        n_steps = float(steps)
    )

def augment_audio(audio, condition, noise_files, rng):
    audio = audio.copy()

    if condition["pitch"]:
        steps = rng.uniform(*CONFIG["pitch_steps"])

        audio = librosa.effects.pitch_shift(
            audio,
            sr = CONFIG["sample_rate"],
            n_steps = float(steps)
        )

    if condition["stretch"]:
        rate = rng.uniform(*CONFIG["stretch_rates"])
        audio = librosa.effects.time_stretch(audio, rate = float(rate))

    if condition["noise"]:
        noise_path = noise_files[int(rng.integers(len(noise_files)))]
        noise, _ = librosa.load(
            noise_path,
            sr = CONFIG["sample_rate"],
            mono = True
        )

        if len(noise) == 0 or not np.isfinite(noise).all():
            raise ValueError(f"Invalid noise recording: {noise_path}")

        noise = noise - noise.mean()

        start = int(rng.integers(len(noise)))
        indices = np.arange(len(audio)) + start
        noise = np.take(noise, indices, mode = "wrap")

        noise_rms = np.sqrt(np.mean(noise.astype(np.float64) ** 2))
        if noise_rms <= 1e-8:
            raise ValueError(f"Silent noise segment: {noise_path}")

        audio_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        snr_db = rng.uniform(*CONFIG["noise_snr_db"])
        scale = audio_rms / (noise_rms * 10 ** (snr_db / 20))

        audio = audio + noise * scale

    return audio.astype(np.float32)

def create_files(df, name, noise_files):
    condition = SETTINGS[name]
    output_dir = OUTPUT_DIR / name
    rng = np.random.default_rng(CONFIG["seed"])

    output_dir.mkdir(parents = True)

    rows = []
    for row in df.to_dict("records"):
        actor_dir = f"Actor_{int(row['actor_id']):02d}"
        filename = f"{row['file']}.wav"

        source_path = DATA_DIR / actor_dir / filename
        destination = output_dir / actor_dir / filename

        audio, sample_rate = librosa.load(
            source_path,
            sr = CONFIG["sample_rate"],
            mono = True
        )

        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Invalid audio: {source_path}")

        if condition["normalize"]:
            audio = normalize_f0(audio)

            if audio is None:
                continue

        if any(condition[key] for key in ["pitch", "stretch", "noise"]):
            audio = augment_audio(
                audio,
                condition,
                noise_files,
                rng
            )

        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Invalid output: {destination}")

        destination.parent.mkdir(parents = True, exist_ok = True)

        sf.write(
            destination,
            audio,
            sample_rate,
            subtype = "FLOAT"
        )

        rows.append({
            **row,
            "duration_seconds": round(len(audio) / sample_rate, 6)
        })

    index = pd.DataFrame(rows, columns = df.columns)
    index["class_count"] = (index.groupby("emotion")["emotion"].transform("size"))
    index["actor_count"] = (index.groupby("actor_id")["actor_id"].transform("size"))
    index.to_csv(output_dir / "index.csv", index = False)

def main():
    df = pd.read_csv(DATA_DIR / "index.csv")
    df = df.sort_values("file")

    noise_files = sorted(NOISE_DIR.rglob("*.wav"))
    for name in ACTIVE_SETTINGS:
        if (OUTPUT_DIR / name).exists():
            print(f"Skipping {name}: folder already exists")
            continue

        create_files(df, name, noise_files)

if __name__ == "__main__":
    main()