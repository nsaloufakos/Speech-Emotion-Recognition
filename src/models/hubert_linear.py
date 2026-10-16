from pathlib import Path

import librosa
import torch

from torch import nn
from transformers import AutoFeatureExtractor, HubertModel

from run_experiments import DATA_DIR, DEVICE, EMOTIONS, run_experiment

OUTPUT_DIR = Path("output/hubert")

CONFIG = {
    "checkpoint": "facebook/hubert-base-ls960",
    "sample_rate": 16000,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 30,
}

class HuBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Linear(768, len(EMOTIONS))

    def forward(self, audio, lengths):
        return self.classifier(audio.squeeze(1))

def extract_features(df):
    features = {}

    processor = AutoFeatureExtractor.from_pretrained(CONFIG["checkpoint"])

    backbone = HubertModel.from_pretrained(CONFIG["checkpoint"]).to(DEVICE)
    backbone.eval()
    backbone.requires_grad_(False)

    with torch.no_grad():
        for row in df.itertuples():
            path = DATA_DIR / f"Actor_{row.actor_id:02d}" / f"{row.file}.wav"

            audio, sample_rate = librosa.load(
                path,
                sr = CONFIG["sample_rate"]
            )

            inputs = processor(
                audio,
                sampling_rate = sample_rate,
                return_tensors = "pt"
            )

            output = backbone(inputs.input_values.to(DEVICE))

            features[row.file] = (output.last_hidden_state.mean(dim = 1).cpu().numpy())

    return features

if __name__ == "__main__":
    run_experiment(HuBERTClassifier, extract_features, CONFIG, OUTPUT_DIR)