from pathlib import Path
import pandas as pd
import torch
import json

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

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

DATA_DIR = Path("data/RAVDESS")
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

    splits = create_splits(df)

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
    run_dir = output_dir / f"seed_{seed}" / f"fold_{fold_id:02d}"
    run_dir.mkdir(parents = True, exist_ok = True)

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

    for epoch in range(1, config["epochs"] + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer)
        val_metrics, _ = run_epoch(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_uar": train_metrics["uar"],
            "val_uar": val_metrics["uar"]
        })

        if val_metrics["uar"] > best_uar:
            best_uar = val_metrics["uar"]
            with (run_dir / "best_model.pt").open("wb") as file:
                torch.save(model.state_dict(), file)

        print(
            f"Seed {seed} | Fold {fold_id} | Epoch {epoch} | "
            f"Train loss {train_metrics['loss']:.3f} | "
            f"Validation UAR {val_metrics['uar']:.3f}"
        )

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index = False)

    model.load_state_dict(
        torch.load(
            run_dir / "best_model.pt",
            map_location = DEVICE,
            weights_only = True
        )
    )

    test_metrics, predictions = run_epoch(model, test_loader)

    result = test[["file", "actor_id", "emotion", "label"]].copy()
    result["predicted_label"] = predictions
    result["predicted_emotion"] = [
        EMOTIONS[label] for label in predictions
    ]
    result["fold"] = fold_id
    result["seed"] = seed

    result.to_csv(run_dir / "predictions.csv", index = False)

    with (run_dir / "metrics.json").open("w") as file:
        json.dump(test_metrics, file, indent = 2)

    print(
        f"Test UAR: {test_metrics['uar']:.3f} | "
        f"Test WAR: {test_metrics['war']:.3f}"
    )

    return result

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
    output_dir.mkdir(parents = True, exist_ok = True)

    df = pd.read_csv(DATA_DIR / "index.csv")
    df["label"] = df["emotion"].map({emotion: i for i, emotion in enumerate(EMOTIONS)})

    features = extract_features(df)
    splits = load_splits(df)
    splits.to_csv(output_dir / "splits.csv", index = False)

    with (output_dir / "config.json").open("w") as file:
        json.dump(config, file, indent = 2)

    seed_results = []

    for seed in config["seeds"]:
        results = []

        for fold_id in config["folds"]:
            fold = splits[splits["fold"] == fold_id]
            results.append(run_fold(fold, features, seed, model_class, config, output_dir))

        predictions = pd.concat(results, ignore_index = True)
        predictions.to_csv(output_dir / f"predictions_seed_{seed}.csv", index = False)

        metrics = calculate_metrics(
            predictions["label"],
            predictions["predicted_label"]
        )

        seed_results.append({
            "seed": seed,
            "folds_completed": len(config["folds"]),
            **metrics
        })

        matrix = confusion_matrix(
            predictions["label"],
            predictions["predicted_label"],
            labels = range(len(EMOTIONS))
        )

        pd.DataFrame(
            matrix,
            index = EMOTIONS,
            columns = EMOTIONS,
        ).to_csv(output_dir / f"confusion_seed_{seed}.csv")

    summary = pd.DataFrame(seed_results)
    summary.to_csv(output_dir / "seed_results.csv", index = False)
    print(summary.to_string(index = False))

    if len(config["seeds"]) > 1:
        print(summary[["uar", "war"]].agg(["mean", "std"]))