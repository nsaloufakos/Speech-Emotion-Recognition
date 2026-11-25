"""Epoch execution with metrics transferred once per epoch instead of per batch."""

import torch
import run_experiments as experiments


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, labels, predictions = [], [], []
    count = 0
    with torch.set_grad_enabled(training):
        for audio, lengths, targets in loader:
            audio = audio.to(experiments.DEVICE)
            targets = targets.to(experiments.DEVICE)
            if training:
                optimizer.zero_grad()
            scores = model(audio, lengths)
            loss = torch.nn.functional.cross_entropy(scores, targets)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.detach() * len(targets))
            labels.append(targets.detach())
            predictions.append(scores.detach().argmax(dim=1))
            count += len(targets)
    labels = torch.cat(labels).cpu().tolist()
    predictions = torch.cat(predictions).cpu().tolist()
    metrics = experiments.calculate_metrics(labels, predictions)
    metrics["loss"] = torch.stack(losses).sum().item() / count
    return metrics, predictions
