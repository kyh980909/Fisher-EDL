import csv
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fisher_edl.data import one_hot
from fisher_edl.losses import edl_mse_loss, fisher_edl_mse_loss


@dataclass
class TrainConfig:
    method: str = "fisher"
    epochs: int = 120
    lr: float = 1e-3
    beta: float = 1.0
    gamma: float = 1.0
    device: str = "cpu"
    log_every: int = 10


def _batch_metrics(logits, labels):
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    sum_alpha = torch.sum(alpha, dim=1)
    probs = alpha / sum_alpha.unsqueeze(1)

    pred = torch.argmax(probs, dim=1)
    mask = labels >= 0
    acc = (pred[mask] == labels[mask]).float().mean().item() if mask.any() else float("nan")

    uncertainty = (labels == -1).float().mean().item() if labels.numel() > 0 else float("nan")
    avg_evidence = evidence.mean().item()
    avg_strength = sum_alpha.mean().item()
    return {
        "acc": acc,
        "avg_evidence": avg_evidence,
        "avg_strength": avg_strength,
        "ood_fraction": uncertainty,
    }


def _evaluate(model, loader, num_classes, device):
    model.eval()
    losses = []
    metrics = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            targets = one_hot(labels, num_classes).to(device)
            loss, _ = edl_mse_loss(logits, targets, kl_weight=0.0)
            losses.append(loss.item())
            metrics.append(_batch_metrics(logits, labels))

    mean_loss = sum(losses) / max(1, len(losses))
    avg_acc = sum(m["acc"] for m in metrics) / max(1, len(metrics))
    avg_evidence = sum(m["avg_evidence"] for m in metrics) / max(1, len(metrics))
    avg_strength = sum(m["avg_strength"] for m in metrics) / max(1, len(metrics))

    return {
        "eval_loss": mean_loss,
        "eval_acc": avg_acc,
        "eval_evidence": avg_evidence,
        "eval_strength": avg_strength,
    }


def train(model, train_loader, val_loader, num_classes, cfg: TrainConfig, run_dir=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.to(cfg.device)

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow([
            "epoch",
            "loss",
            "risk",
            "kl",
            "weight",
            "train_acc",
            "train_evidence",
            "train_strength",
            "val_acc",
            "val_evidence",
            "val_strength",
        ])
    else:
        csv_file = None
        writer = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_risk = 0.0
        epoch_kl = 0.0
        epoch_weight = 0.0
        epoch_metrics = []

        for features, labels in train_loader:
            features = features.to(cfg.device)
            labels = labels.to(cfg.device)
            targets = one_hot(labels, num_classes).to(cfg.device)

            logits = model(features)
            if cfg.method == "fisher":
                loss, stats = fisher_edl_mse_loss(logits, targets, beta=cfg.beta, gamma=cfg.gamma)
                epoch_weight += stats["weight"]
            else:
                loss, stats = edl_mse_loss(logits, targets, kl_weight=cfg.beta)
                epoch_weight += cfg.beta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_risk += stats["risk"]
            epoch_kl += stats["kl"]
            epoch_metrics.append(_batch_metrics(logits.detach(), labels))

        mean_loss = epoch_loss / max(1, len(train_loader))
        mean_risk = epoch_risk / max(1, len(train_loader))
        mean_kl = epoch_kl / max(1, len(train_loader))
        mean_weight = epoch_weight / max(1, len(train_loader))
        mean_acc = sum(m["acc"] for m in epoch_metrics) / max(1, len(epoch_metrics))
        mean_evidence = sum(m["avg_evidence"] for m in epoch_metrics) / max(1, len(epoch_metrics))
        mean_strength = sum(m["avg_strength"] for m in epoch_metrics) / max(1, len(epoch_metrics))

        val_stats = _evaluate(model, val_loader, num_classes, cfg.device)

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d} | loss={mean_loss:.4f} risk={mean_risk:.4f} "
                f"kl={mean_kl:.4f} w={mean_weight:.4f} "
                f"train_acc={mean_acc:.3f} val_acc={val_stats['eval_acc']:.3f}"
            )

        if writer:
            writer.writerow([
                epoch,
                f"{mean_loss:.6f}",
                f"{mean_risk:.6f}",
                f"{mean_kl:.6f}",
                f"{mean_weight:.6f}",
                f"{mean_acc:.6f}",
                f"{mean_evidence:.6f}",
                f"{mean_strength:.6f}",
                f"{val_stats['eval_acc']:.6f}",
                f"{val_stats['eval_evidence']:.6f}",
                f"{val_stats['eval_strength']:.6f}",
            ])

    if csv_file:
        csv_file.close()
