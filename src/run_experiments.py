from pathlib import Path
from time import perf_counter
import pandas as pd
import torch
import json

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_recall_fscore_support

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

CONFIG = {
    "validation_actors": 2,
    "split_seed": 42,
    "folds": list(range(1, 25)),
    "seeds": [42, 123, 999]
}

#DATA_DIR = Path("data/RAVDESS")
DATA_DIR = Path("data/audio_variants/normalized_only")
DATASET = "original" if DATA_DIR.name == "RAVDESS" else DATA_DIR.name.title()
SPLITS_PATH = Path("output/splits.csv")

def create_splits(df: pd.DataFrame):
    loso = LeaveOneGroupOut()
    validation = GroupShuffleSplit(
        n_splits = 1,
        test_size = CONFIG["validation_actors"],
        random_state = CONFIG["split_seed"]
    )
    folds = []

    for remaining_idx, test_idx in loso.split(df, groups = df["actor_id"]):
        remaining = df.iloc[remaining_idx]
        train_idx, val_idx = next(validation.split(remaining, groups = remaining["actor_id"]))

        train = remaining.iloc[train_idx].copy()
        val = remaining.iloc[val_idx].copy()
        test = df.iloc[test_idx].copy()

        train["split"] = "train"
        val["split"] = "val"
        test["split"] = "test"

        train_actors = set(train["actor_id"])
        val_actors = set(val["actor_id"])
        test_actors = set(test["actor_id"])

        assert train_actors.isdisjoint(val_actors)
        assert train_actors.isdisjoint(test_actors)
        assert val_actors.isdisjoint(test_actors)

        fold = pd.concat([train, val, test])
        fold["fold"] = int(test["actor_id"].iloc[0])
        folds.append(fold)

    return pd.concat(folds, ignore_index = True)

def load_splits(df):
    if SPLITS_PATH.exists():
        return pd.read_csv(SPLITS_PATH)

    original = pd.read_csv(Path("data/RAVDESS/index.csv"))
    original["label"] = original["emotion"].map({emotion: i for i, emotion in enumerate(EMOTIONS)})

    splits = create_splits(original)
    SPLITS_PATH.parent.mkdir(parents = True, exist_ok = True)
    splits.to_csv(SPLITS_PATH, index = False)

    return splits

def run_epoch(model, loader, optimizer = None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0
    labels = []
    predictions = []

    with torch.set_grad_enabled(training):
        for audio, lengths, targets in loader:
            audio = audio.to(DEVICE)
            targets = targets.to(DEVICE)

            if training:
                optimizer.zero_grad()

            scores = model(audio, lengths)
            loss = nn.functional.cross_entropy(scores, targets)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(targets)
            labels.extend(targets.cpu().tolist())
            predictions.extend(scores.argmax(dim = 1).cpu().tolist())

    metrics = calculate_metrics(labels, predictions)
    metrics["loss"] = total_loss / len(labels)

    return metrics, predictions

def run_fold(fold, features, seed, model_class, config, output_dir):
    torch.manual_seed(seed)

    fold_id = int(fold["fold"].iloc[0])
    started = perf_counter()

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents = True, exist_ok = True)

    checkpoint_path = checkpoint_dir / f"seed_{seed}_fold_{fold_id:02d}.pt"

    train = fold[fold["split"] == "train"]
    val = fold[fold["split"] == "val"]
    test = fold[fold["split"] == "test"]

    train_loader = create_loader(train, features, config["batch_size"], shuffle = True)
    val_loader = create_loader(val, features, config["batch_size"])
    test_loader = create_loader(test, features, config["batch_size"])

    model = model_class().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = config["learning_rate"]
    )

    history = []
    best_uar = -1
    best_epoch = 0

    for epoch in range(1, config["epochs"] + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer)
        val_metrics, _ = run_epoch(model, val_loader)

        history.append({
            "seed": seed,
            "fold": fold_id,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_uar": train_metrics["uar"],
            "val_uar": val_metrics["uar"]
        })

        if val_metrics["uar"] > best_uar:
            best_uar = val_metrics["uar"]
            best_epoch = epoch

            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location = DEVICE, weights_only = True))
    test_metrics, predictions = run_epoch(model, test_loader)

    result = test[["file", "actor_id", "emotion", "label"]].copy()
    result["predicted_label"] = predictions
    result["predicted_emotion"] = [EMOTIONS[label] for label in predictions]
    result["fold"] = fold_id
    result["seed"] = seed

    fold_metrics = {
        "seed": seed,
        "fold": fold_id,
        "best_epoch": best_epoch,
        "train_recordings": len(train),
        "val_recordings": len(val),
        "test_recordings": len(test),
        "val_uar": best_uar,
        "val_loss": history[best_epoch - 1]["val_loss"],
        **test_metrics,
        "seconds": round(perf_counter() - started, 2)
    }

    print(f"{output_dir.name} | Seed {seed} | Fold {fold_id:02d} complete | val uar: {fold_metrics['val_uar']:.3f}")

    return result, pd.DataFrame(history), fold_metrics

def calculate_metrics(labels, predictions):
    return {
        "uar": recall_score(
            labels,
            predictions,
            labels = range(len(EMOTIONS)),
            average = "macro",
            zero_division = 0
        ),
        "war": accuracy_score(labels, predictions)
    }

def evaluate_predictions(labels, predictions):
    class_ids = range(len(EMOTIONS))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels = class_ids,
        zero_division = 0
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels = class_ids
    )

    normalized = confusion_matrix(
        labels,
        predictions,
        labels = class_ids,
        normalize = "true"
    )

    return {
        **calculate_metrics(labels, predictions),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "recordings_evaluated": int(support.sum()),
        "per_emotion": {
            emotion: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
            for i, emotion in enumerate(EMOTIONS)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist()
    }

def collate_batch(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor([len(sequence) for sequence in sequences])
    audio = pad_sequence(sequences, batch_first = True)
    labels = torch.tensor(labels)

    return audio, lengths, labels

def create_loader(df, features, batch_size, shuffle = False):
    recordings = [
        (
            torch.tensor(features[row.file], dtype = torch.float32),
            int(row.label)
        )
        for row in df.itertuples()
    ]

    return DataLoader(
        recordings,
        batch_size = batch_size,
        shuffle = shuffle,
        collate_fn = collate_batch
    )

def run_experiment(model_class, extract_features, config, output_dir):
    config = {**CONFIG, **config}

    output_dir = Path("output") / DATASET / output_dir.name
    output_dir.mkdir(parents = True, exist_ok = True)

    df = pd.read_csv(DATA_DIR / "index.csv")
    df["label"] = df["emotion"].map({emotion: i for i, emotion in enumerate(EMOTIONS)})

    splits = load_splits(df)
    splits = splits[splits["file"].isin(df["file"])].copy()

    with (output_dir / "config.json").open("w") as file:
        json.dump({**config, "splits_path": str(SPLITS_PATH)}, file, indent = 2)


    features = extract_features(df)
    print(f"Starting {output_dir.name}")

    predictions = []
    histories = []
    fold_results = []
    seed_results = []

    for seed in config["seeds"]:
        results = []

        for fold_id in config["folds"]:
            fold = splits[splits["fold"] == fold_id]

            if fold.empty:
                raise ValueError(f"Fold {fold_id} is missing from {SPLITS_PATH}.")

            result, history, metrics = run_fold(fold, features, seed, model_class, config, output_dir)

            results.append(result)
            predictions.append(result)
            histories.append(history)
            fold_results.append(metrics)

            pd.concat(predictions, ignore_index = True).to_csv(output_dir / "predictions.csv", index = False)
            pd.concat(histories, ignore_index = True).to_csv(output_dir / "history.csv", index = False)

        seed_predictions = pd.concat(results, ignore_index = True)
        metrics = evaluate_predictions(seed_predictions["label"], seed_predictions["predicted_label"])
        seed_folds = [row for row in fold_results if row["seed"] == seed]

        seed_results.append({
            "seed": seed,
            "folds_completed": len(config["folds"]),
            "val_uar": (sum(row["val_uar"] for row in seed_folds) / len(seed_folds)),
            **metrics
        })

        print(f"{output_dir.name} | Seed {seed} complete")

    scores = pd.DataFrame(seed_results)
    summary = {}

    for metric in ["val_uar", "uar", "war", "macro_f1", "weighted_f1"]:
        summary[f"{metric}_mean"] = scores[metric].mean()
        summary[f"{metric}_std"] = scores[metric].std()

    summary = pd.DataFrame([summary])

    evaluation = {
        "seeds_completed": len(scores),
        "folds_per_seed": len(config["folds"]),
        "emotion_order": EMOTIONS,
        "fold_results": fold_results,
        "seed_results": seed_results,
        "summary": {
            name: None if pd.isna(value) else float(value)
            for name, value in summary.iloc[0].items()
        }
    }

    with (output_dir / "evaluation.json").open("w") as file:
        json.dump(evaluation, file, indent = 2, allow_nan = False)

    print(f"{output_dir.name} complete | Results saved to {output_dir}\n")

    return summary