import csv
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from fisher_edl.cifar_data import one_hot
from fisher_edl.losses import edl_mse_loss, fisher_edl_mse_loss, fisher_weight
from fisher_edl.metrics import uncertainty_from_logits


@dataclass
class CifarTrainConfig:
    method: str = "fisher"
    epochs: int = 100
    lr: float = 1e-3
    beta: float = 1.0
    gamma: float = 1.0
    device: str = "cpu"
    log_every: int = 10
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


def _train_epoch(model, loader, num_classes, cfg, optimizer):
    model.train()
    epoch_loss = 0.0
    epoch_risk = 0.0
    epoch_kl = 0.0
    epoch_weight = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)
        targets = one_hot(labels, num_classes).to(cfg.device)

        logits = model(images)
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

        with torch.no_grad():
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    mean_loss = epoch_loss / max(1, len(loader))
    mean_risk = epoch_risk / max(1, len(loader))
    mean_kl = epoch_kl / max(1, len(loader))
    mean_weight = epoch_weight / max(1, len(loader))
    acc = correct / max(1, total)

    return {
        "loss": mean_loss,
        "risk": mean_risk,
        "kl": mean_kl,
        "weight": mean_weight,
        "acc": acc,
    }


def _eval_id(model, loader, cfg):
    model.eval()
    correct = 0
    total = 0
    uncertainties = []
    weights = []

    if loader is None:
        return float("nan"), float("nan"), float("nan"), torch.empty(0)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            logits = model(images)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            uncertainties.append(uncertainty_from_logits(logits))
            if cfg.method == "fisher":
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                weights.append(fisher_weight(alpha))

    acc = correct / max(1, total)
    mean_unc = torch.cat(uncertainties).mean().item()
    mean_weight = torch.cat(weights).mean().item() if weights else float("nan")
    return acc, mean_unc, mean_weight, torch.cat(uncertainties)


def _eval_ood(model, loader, cfg):
    model.eval()
    uncertainties = []
    weights = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(cfg.device)
            logits = model(images)
            uncertainties.append(uncertainty_from_logits(logits))
            if cfg.method == "fisher":
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                weights.append(fisher_weight(alpha))

    mean_unc = torch.cat(uncertainties).mean().item()
    mean_weight = torch.cat(weights).mean().item() if weights else float("nan")
    return mean_unc, mean_weight, torch.cat(uncertainties)


def train_cifar(
    model,
    train_loader,
    val_loader,
    num_classes,
    cfg: CifarTrainConfig,
    run_dir=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.to(cfg.device)

    best_val_acc = -1.0
    best_val_path = None

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project or "fisher-edl",
            name=cfg.wandb_run_name,
            config={
                "method": cfg.method,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "beta": cfg.beta,
                "gamma": cfg.gamma,
                "num_classes": num_classes,
            },
        )

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        last_ckpt_path = os.path.join(run_dir, "last.pt")
        best_val_path = os.path.join(run_dir, "best_val_acc.pt")
        writer.writerow([
            "epoch",
            "loss",
            "risk",
            "kl",
            "weight",
            "train_acc",
            "val_acc",
            "val_uncertainty",
            "val_fisher_weight",
        ])
    else:
        csv_file = None
        writer = None

    for epoch in range(1, cfg.epochs + 1):
        train_stats = _train_epoch(model, train_loader, num_classes, cfg, optimizer)
        val_acc, val_unc, val_weight, val_scores = _eval_id(model, val_loader, cfg)
        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d} | loss={train_stats['loss']:.4f} "
                f"risk={train_stats['risk']:.4f} kl={train_stats['kl']:.4f} "
                f"w={train_stats['weight']:.4f} train_acc={train_stats['acc']:.3f} "
                f"val_acc={val_acc:.3f}"
            )

        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "loss": train_stats["loss"],
                    "risk": train_stats["risk"],
                    "kl": train_stats["kl"],
                    "weight": train_stats["weight"],
                    "train_acc": train_stats["acc"],
                    "val_acc": val_acc,
                    "val_uncertainty": val_unc,
                    "val_fisher_weight": val_weight,
                }
            )

        if writer:
            writer.writerow([
                epoch,
                f"{train_stats['loss']:.6f}",
                f"{train_stats['risk']:.6f}",
                f"{train_stats['kl']:.6f}",
                f"{train_stats['weight']:.6f}",
                f"{train_stats['acc']:.6f}",
                f"{val_acc:.6f}",
                f"{val_unc:.6f}",
                f"{val_weight:.6f}",
            ])

        if run_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                },
                last_ckpt_path,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": cfg,
                    },
                    best_val_path,
                )

    if csv_file:
        csv_file.close()

    if wandb_run:
        wandb_run.finish()
