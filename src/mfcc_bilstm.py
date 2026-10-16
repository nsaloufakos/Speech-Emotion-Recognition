from pathlib import Path
import pandas as pd
import librosa

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from run_experiments import DATA_DIR, EMOTIONS, run_experiment

CONFIG = {
    "sample_rate": 16000,
    "n_mfcc": 40,
    "hidden_size": 64,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 30
}

OUTPUT_DIR = Path("output/baseline")

class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = CONFIG["n_mfcc"],
            hidden_size = CONFIG["hidden_size"],
            batch_first = True,
            bidirectional = True
        )
        self.classifier = nn.Linear(CONFIG["hidden_size"] * 2, len(EMOTIONS))

    def forward(self, audio, lengths):
        packed = pack_padded_sequence(
            audio,
            lengths.cpu(),
            batch_first = True,
            enforce_sorted = False
        )

        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat([hidden[-2], hidden[-1]], dim = 1)

        return self.classifier(representation)

def extract_features(df: pd.DataFrame):
    features = {}

    for row in df.itertuples():
        path = DATA_DIR / f"Actor_{row.actor_id:02d}" / f"{row.file}.wav"

        audio, sample_rate = librosa.load(
            path,
            sr = CONFIG["sample_rate"]
        )

        mfcc = librosa.feature.mfcc(
            y = audio,
            sr = sample_rate,
            n_mfcc = CONFIG["n_mfcc"]
        )

        features[row.file] = mfcc.T

    return features

if __name__ == "__main__":
    run_experiment(BiLSTM, extract_features, CONFIG, OUTPUT_DIR)