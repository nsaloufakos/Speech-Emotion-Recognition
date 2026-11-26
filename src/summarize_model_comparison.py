"""Verify the matched P20 baseline matrix and export report-ready data/figures."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ser-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["original", "normalized_psola_actor_praat_median", "normalized_psola_actor_praat_median_cap4"]
LABELS = ["Original", "New full", "New cap 4"]
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lfs_pointer_digest(path):
    if path.stat().st_size >= 1024:
        return None
    contents = path.read_bytes()
    if not contents.startswith(LFS_HEADER):
        return None
    for line in contents.splitlines():
        if line.startswith(b"oid sha256:"):
            return line.removeprefix(b"oid sha256:").decode("ascii")
    raise ValueError(f"Missing SHA-256 in Git LFS pointer: {path}")


def source_root(model, condition, mfcc_root=None):
    if model == "mfcc" and mfcc_root is not None:
        return mfcc_root.resolve()
    if model == "hubert":
        return ROOT / ("output/pitch_comparison_hubert" if condition == "original" else "output/pitch_comparison_praat_median_hubert")
    return ROOT / f"output/pitch_comparison_praat_median_{model}"


def metrics(frame):
    matrix = np.bincount(frame.label.to_numpy(int) * 8 + frame.predicted_label.to_numpy(int), minlength=64).reshape(8, 8)
    recall = np.diag(matrix) / matrix.sum(axis=1)
    return float(recall.mean()), float(np.trace(matrix) / matrix.sum()), recall, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="Export only completed cells, clearly marked partial")
    parser.add_argument("--mfcc-output-root", type=Path, help="Use a separate complete CUDA MFCC rerun")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/model_comparison")
    parser.add_argument("--archived-audio", action="store_true",
                        help="Use saved generation hashes for missing variant WAVs and Git LFS pointers")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = sorted(pd.read_csv(ROOT / "output/pitch_comparison_hubert/matched_recordings.csv").file)
    splits = pd.read_csv(ROOT / "output/splits.csv")
    test = splits[(splits.split == "test") & splits.file.isin(selected)]
    expected = test.set_index("file")[["actor_id", "fold", "label"]].sort_index()
    assert len(expected) == len(selected) and not expected.index.duplicated().any()
    rows, seeds, emotions, actors, provenance, contrasts = [], [], [], [], {}, []
    missing, saved_predictions, confusions = [], {}, {}
    archived_audio_records = 0
    for model in ("mfcc", "wav2vec", "hubert"):
        for condition in CONDITIONS:
            root = source_root(model, condition, args.mfcc_output_root)
            directory = root / condition / f"{model}_baseline"
            path = directory / "evaluation.json"
            if not path.exists():
                missing.append(f"{model}/{condition}")
                continue
            ev = json.loads(path.read_text())
            if ev["seeds_completed"] != 3 or ev["folds_per_seed"] != 24:
                raise ValueError(f"Incomplete seeds/folds: {path}")
            config = json.loads((root / "comparison_config.json").read_text())
            if config["recordings"] != selected or config["splits_sha256"] != sha(ROOT / "output/splits.csv"):
                raise ValueError(f"Unmatched cohort/splits: {path}")
            pred = pd.read_csv(directory / "predictions.csv")
            if len(pred) != 3 * len(selected) or pred[["seed", "file"]].duplicated().any():
                raise ValueError("Missing or duplicate held-out predictions")
            feature_manifest = json.loads((root / condition / "features/manifest.json").read_text())
            data = ROOT / "data/RAVDESS" if condition == "original" else ROOT / "data/audio_variants" / condition
            if condition != "original" and config.get("variant_root"):
                data = Path(config["variant_root"]) / condition
            archived = None
            for record in test.itertuples():
                audio = data / f"Actor_{record.actor_id:02d}" / f"{record.file}.wav"
                exists = audio.exists()
                pointer_digest = lfs_pointer_digest(audio) if exists else None
                if exists and pointer_digest is None:
                    digest = sha(audio)
                elif args.archived_audio and condition != "original":
                    if archived is None:
                        archived = pd.read_csv(data / "diagnostics.csv").set_index("file")
                        if archived.index.duplicated().any():
                            raise ValueError(f"Duplicate archived diagnostics: {data}")
                    digest = archived.loc[record.file, "output_sha256"]
                    if pointer_digest is not None and pointer_digest != digest:
                        raise ValueError(f"Git LFS pointer disagrees with archived waveform hash: {audio}")
                    archived_audio_records += 1
                else:
                    raise FileNotFoundError(f"Missing waveform bytes for {audio}. Restore it or explicitly use --archived-audio.")
                if digest != feature_manifest["recordings"].get(record.file):
                    raise ValueError(f"Changed evaluation audio: {model}/{condition}/{record.file}")
            local = []
            matrices = []
            for seed in (42, 123, 999):
                frame = pred[pred.seed == seed].set_index("file").sort_index()
                if not frame[["actor_id", "fold", "label"]].equals(expected):
                    raise ValueError(f"Wrong test membership: {model}/{condition}/{seed}")
                uar, war, recall, matrix = metrics(frame)
                row = {"model": model, "condition": condition, "seed": seed, "uar": uar, "war": war}
                local.append(row)
                seeds.append(row)
                matrices.append(matrix)
                for name, value in zip(EMOTIONS, recall):
                    emotions.append({**row, "emotion": name, "recall": float(value)})
                for actor, group in frame.groupby("actor_id"):
                    au, aw, _, _ = metrics(group)
                    actors.append({"model": model, "condition": condition, "seed": seed,
                                   "actor_id": int(actor), "uar": au, "war": aw})
            score = pd.DataFrame(local)
            row = {"model": model, "condition": condition, "unique_recordings": len(selected),
                   "held_out_predictions": len(pred), "scope": "single_utterance" if condition == "original" else "batch_speaker_adaptation"}
            for metric in ("uar", "war"):
                for stat in ("mean", "std"):
                    value = float(getattr(score[metric], stat)())
                    if not np.isclose(value, ev["summary"][f"{metric}_{stat}"], rtol=0, atol=1e-12):
                        raise ValueError("Stored/recomputed score mismatch")
                    row[f"{metric}_{stat}"] = value
            rows.append(row)
            saved_predictions[(model, condition)] = pred
            total = np.sum(matrices, axis=0)
            confusions[(model, condition)] = total / total.sum(axis=1, keepdims=True)
            provenance[f"{model}/{condition}"] = {"evaluation": str(path.relative_to(ROOT)),
                                                   "evaluation_sha256": sha(path), "predictions_sha256": sha(directory / "predictions.csv"),
                                                   "configuration_sha256": sha(root / "comparison_config.json")}
    if missing and not args.allow_incomplete:
        raise ValueError(f"Still missing completed evaluations: {missing}")
    table = pd.DataFrame(rows)
    seed_table = pd.DataFrame(seeds)
    table.to_csv(out / "main_results.csv", index=False)
    seed_table.to_csv(out / "seed_scores.csv", index=False)
    pd.DataFrame(emotions).to_csv(out / "per_emotion_recall.csv", index=False)
    pd.DataFrame(actors).to_csv(out / "per_actor_scores.csv", index=False)
    for model, group in seed_table.groupby("model"):
        wide = group.pivot(index="seed", columns="condition", values="uar")
        for condition in CONDITIONS[1:]:
            if condition not in wide or "original" not in wide:
                continue
            delta = 100 * (wide[condition] - wide.original)
            contrasts.append({"model": model, "condition": condition, "reference": "original",
                              "mean_delta_uar_pp": float(delta.mean()),
                              "paired_seed_delta_uar_pp": {str(k): float(v) for k, v in delta.items()}})
    (out / "paired_comparisons.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    # Conditional uncertainty: resample held-out actors, paired across conditions
    # and seeds. This does not retrain models or remove LOSO training overlap.
    rng = np.random.default_rng(473)
    actor_ids = sorted(test.actor_id.unique())
    weights = rng.multinomial(len(actor_ids), np.ones(len(actor_ids)) / len(actor_ids), size=10000)
    bootstrap = []
    for model in ("mfcc", "wav2vec", "hubert"):
        if not all((model, c) in saved_predictions for c in CONDITIONS):
            continue
        cubes = {}
        for condition in CONDITIONS:
            pred = saved_predictions[(model, condition)]
            cube = np.zeros((3, len(actor_ids), 8, 8), dtype=np.int64)
            for si, seed in enumerate((42, 123, 999)):
                for ai, actor in enumerate(actor_ids):
                    cube[si, ai] = metrics(pred[(pred.seed == seed) & (pred.actor_id == actor)])[3]
            cubes[condition] = cube
        def sampled_uar(cube):
            counts = np.einsum("ba,saij->bsij", weights, cube)
            return (np.diagonal(counts, axis1=2, axis2=3) / counts.sum(axis=3)).mean(axis=(1, 2))
        reference = sampled_uar(cubes["original"])
        for condition in CONDITIONS[1:]:
            differences = 100 * (sampled_uar(cubes[condition]) - reference)
            low, high = np.percentile(differences, [2.5, 97.5])
            bootstrap.append({"model": model, "condition": condition, "reference": "original",
                              "actor_resampling_interval_low_pp": float(low), "actor_resampling_interval_high_pp": float(high),
                              "bootstrap_draws": 10000, "actors": len(actor_ids),
                              "scope": "paired held-out actor resampling, average over fixed seed predictions; conditional on fitted models; overlapping LOSO training sets and exploratory selection remain"})
    (out / "actor_resampling_intervals.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    for model in ("mfcc", "wav2vec", "hubert"):
        if not all((model, condition) in confusions for condition in CONDITIONS):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        for ax, condition, label in zip(axes, CONDITIONS, LABELS):
            matrix = confusions[(model, condition)]
            ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            for i in range(8):
                for j in range(8):
                    ax.text(j, i, f"{matrix[i,j]*100:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if matrix[i,j] > .55 else "black")
            ax.set(xticks=range(8), xticklabels=EMOTIONS, yticks=range(8), yticklabels=EMOTIONS,
                   xlabel="Predicted emotion", ylabel="True emotion", title=label)
            ax.tick_params(axis="x", labelrotation=65, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
        fig.suptitle(f"{model}: row-normalized confusion percentages across three seeds; 1,429 unique recordings")
        fig.savefig(out / f"{model}_confusions.png", dpi=150)
        plt.close(fig)
    examples = []
    for model in ("mfcc", "wav2vec", "hubert"):
        keys = ((model, "original"), (model, CONDITIONS[2]))
        if not all(key in saved_predictions for key in keys):
            continue
        original = saved_predictions[keys[0]].query("seed == 42").set_index("file")
        normalized = saved_predictions[keys[1]].query("seed == 42").set_index("file")
        joined = original[["actor_id", "emotion", "label", "predicted_label"]].join(normalized[["predicted_label"]], rsuffix="_new")
        for kind, mask in (("original_correct_new_wrong", (joined.predicted_label == joined.label) & (joined.predicted_label_new != joined.label)),
                           ("original_wrong_new_correct", (joined.predicted_label != joined.label) & (joined.predicted_label_new == joined.label))):
            for file, row in joined[mask].sort_index().head(5).iterrows():
                examples.append({"model": model, "seed": 42, "category": kind, "file": file,
                                 "actor_id": int(row.actor_id), "true_emotion": row.emotion,
                                 "original_prediction": EMOTIONS[int(row.predicted_label)],
                                 "new_capped_prediction": EMOTIONS[int(row.predicted_label_new)]})
    pd.DataFrame(examples).to_csv(out / "prediction_examples.csv", index=False)
    (out / "results_validation.json").write_text(json.dumps({
        "all_nine_cells_complete": not missing, "missing": missing, "unique_recordings": len(selected),
        "seeds": [42,123,999], "folds_per_seed": 24, "metrics_independently_reproduced": True,
        "held_out_membership_verified": True, "audio_hashes_verified": archived_audio_records == 0,
        "archived_audio_hashes_matched_to_generation_records": archived_audio_records,
        "provenance": provenance,
        "interpretation": "Within-model comparisons control hardware, architecture and hyperparameters. Frozen encoder classifier heads train on CPU. Three-seed differences are descriptive, not significance tests."
    }, indent=2) + "\n")
    lines = ["# Matched P20 model comparison", "", "All completed cells use the same 1,429 recordings, 24 LOSO folds and seeds 42, 123 and 999. Values below are mean +/- sample standard deviation in percent. Original audio uses no speaker calibration; normalized conditions use each speaker's available batch.", ""]
    if missing:
        lines.extend(["**Partial matrix: some model conditions are still running.**", ""])
    lines.extend(["| Model | Audio condition | UAR (%) | WAR (%) |", "|---|---|---:|---:|"])
    for row in table.itertuples():
        label = LABELS[CONDITIONS.index(row.condition)]
        lines.append(f"| {row.model} | {label} | {row.uar_mean*100:.2f} +/- {row.uar_std*100:.2f} | {row.war_mean*100:.2f} +/- {row.war_std*100:.2f} |")
    lines.extend(["", "Prediction examples are concrete seed-42 classification changes, not causal diagnoses. Confusion figures summarize the same recordings across seeds, not 4,287 independent examples. Hardware/execution differs across model families but is fixed within each model's original/full/capped comparison.", "", "actor_resampling_intervals.json resamples the 24 test actors jointly across conditions and averages over the three fixed seed predictions (10,000 draws). These are conditional uncertainty summaries; they do not account for retraining, overlapping LOSO training sets, or exploratory method selection, and are not confirmatory significance tests.", "", "Data: main_results.csv, seed_scores.csv, per_emotion_recall.csv, per_actor_scores.csv, paired_comparisons.json, prediction_examples.csv and results_validation.json.", ""])
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(table[["model", "condition", "uar_mean", "uar_std", "war_mean", "war_std"]].to_string(index=False))
    print("Missing:", missing)


if __name__ == "__main__":
    main()
