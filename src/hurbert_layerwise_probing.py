from pathlib import Path

import librosa
import pandas as pd
import torch

from torch import nn
from transformers import AutoFeatureExtractor, HubertModel

from run_experiments import DATA_DIR, DEVICE, EMOTIONS, run_experiment

OUTPUT_DIR = Path("output/hubert_layerwise_probing")

CONFIG = {
    "checkpoint": "facebook/hubert-base-ls960",
    "sample_rate": 16000,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 30,
    "layers": list(range(1, 13)),
    "frozen_backbone": True,
    "pooling": "mean_over_time"
}

class HuBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Linear(768, len(EMOTIONS))

    def forward(self, audio, lengths):
        return self.classifier(audio.squeeze(1))

def extract_features(df):
    features = {layer: {} for layer in CONFIG["layers"]}

    processor = AutoFeatureExtractor.from_pretrained(CONFIG["checkpoint"])

    backbone = HubertModel.from_pretrained(CONFIG["checkpoint"]).to(DEVICE)
    backbone.eval()
    backbone.requires_grad_(False)

    with torch.no_grad():
        for index, row in enumerate(df.itertuples(), start = 1):
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

            output = backbone(
                inputs.input_values.to(DEVICE),
                output_hidden_states = True,
                return_dict = True
            )

            for layer in features:
                features[layer][row.file] = (output.hidden_states[layer].mean(dim = 1).cpu().numpy())

            if index % 100 == 0 or index == len(df):
                print(f"Extracted features for {index}/{len(df)} recordings")

    return features

def run_probing():
    df = pd.read_csv(DATA_DIR / "index.csv")
    print("HuBERT probing starting")

    features = extract_features(df)
    results = []

    for layer in CONFIG["layers"]:
        print(f"Probing HuBERT layer {layer}")

        config = {**CONFIG, "feature_layer": layer}

        def layer_features(df):
            return features[layer]

        summary = run_experiment(
            HuBERTClassifier,
            layer_features,
            config,
            OUTPUT_DIR / f"layer_{layer:02d}"
        )

        results.append({
            "layer": layer,
            **summary.iloc[0].to_dict()
        })

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "layer_results.csv", index = False)
    print(f"HuBERT probing complete | Layer comparison saved to f{OUTPUT_DIR / 'layer_results.csv'}")

if __name__ == "__main__":
    run_probing()