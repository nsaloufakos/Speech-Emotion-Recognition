import pandas as pd
from pathlib import Path
from collections import Counter
import wave
import re

EMOTIONS = {
    "01" : "neutral",
    "02" : "calm",
    "03" : "happy",
    "04" : "sad",
    "05" : "angry",
    "06" : "fearful",
    "07" : "disgust",
    "08" : "surprised"
}

INTESITY = {
    "01" : "normal",
    "02" : "strong"
}

GENDER = {
    0 : "female",
    1 : "male"
}

FILENAME_PATTERN = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")

def main():
    rows = []

    dir_path = Path("data/RAVDESS")
    for path in dir_path.rglob("*.wav"):

        if path.is_file():
            match = FILENAME_PATTERN.fullmatch(path.stem)
            _, _, emotion, intensity, statement, repetition, actor = (match.groups())

            with wave.open(str(path), "rb") as audio:
                sample_rate = audio.getframerate()
                frame_count = audio.getnframes()

            duration = frame_count / sample_rate

            rows.append({
                "file" : path.stem,
                "emotion" : EMOTIONS[emotion],
                "intensity" : INTESITY[intensity],
                "actor_id" : int(actor),
                "gender": GENDER[int(actor) % 2],
                "statement" : statement,
                "repetition" : repetition,
                "duration_seconds" : round(duration, 6)
            })

    df = pd.DataFrame(rows)

    df["class_count"] = df.groupby("emotion")["emotion"].transform("size")
    df["actor_count"] = df.groupby("actor_id")["actor_id"].transform("size")

    df.to_csv(dir_path / "index.csv", index=False)

if __name__ == "__main__":
    main()