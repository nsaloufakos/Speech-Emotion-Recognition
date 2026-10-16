from pathlib import Path

import librosa
import torch

from torch import nn
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from run_experiments import DATA_DIR, EMOTIONS, run_experiment

OUTPUT_DIR = Path("output/wav2vec_finetuned")

CONFIG = {
    "checkpoint": "facebook/wav2vec2-base",
    "sample_rate": 16000,
    "batch_size": 1,
    "learning_rate": 0.00001,
    "epochs": 30,
    "frozen_backbone": False,
    "pooling": "mean_over_time",
    "apply_spec_augment": False,
    "gradient_checkpointing": True
}

class Wav2VecClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Wav2Vec2Model.from_pretrained(
            CONFIG["checkpoint"],
            apply_spec_augment = CONFIG["apply_spec_augment"]
        )
        self.backbone.requires_grad_(True)

        if CONFIG["gradient_checkpointing"]:
            self.backbone.gradient_checkpointing_enable()

        self.classifier = nn.Linear(
            self.backbone.config.hidden_size,
            len(EMOTIONS)
        )

    def forward(self, audio, lengths):
        representations = []

        for waveform, length in zip(audio, lengths):
            output = self.backbone(
                waveform[:int(length)].unsqueeze(0),
                return_dict = True
            )

            representations.append(output.last_hidden_state.mean(dim = 1))

        return self.classifier(torch.cat(representations, dim = 0))

def extract_features(df):
    features = {}
    processor = AutoFeatureExtractor.from_pretrained(CONFIG["checkpoint"])

    for row in df.itertuples():
        path = (DATA_DIR / f"Actor_{row.actor_id:02d}" / f"{row.file}.wav")

        audio, sample_rate = librosa.load(
            path,
            sr = CONFIG["sample_rate"]
        )

        inputs = processor(
            audio,
            sampling_rate = sample_rate,
            return_tensors = "pt"
        )

        features[row.file] = (inputs.input_values.squeeze(0).numpy())

    return features

if __name__ == "__main__":
    run_experiment(Wav2VecClassifier, extract_features, CONFIG, OUTPUT_DIR)