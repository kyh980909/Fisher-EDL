import csv
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from fisher_edl.cifar_data import one_hot
from fisher_edl.losses import edl_mse_loss, fisher_edl_mse_loss
from fisher_edl.metrics import uncertainty_from_logits, compute_auroc


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

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            logits = model(images)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            uncertainties.append(uncertainty_from_logits(logits))

    acc = correct / max(1, total)
    mean_unc = torch.cat(uncertainties).mean().item()
    return acc, mean_unc, torch.cat(uncertainties)


def _eval_ood(model, loader, cfg):
    model.eval()
    uncertainties = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(cfg.device)
            logits = model(images)
            uncertainties.append(uncertainty_from_logits(logits))

    mean_unc = torch.cat(uncertainties).mean().item()
    return mean_unc, torch.cat(uncertainties)


def train_cifar(
    model,
    train_loader,
    val_loader,
    test_loader,
    svhn_loader,
    cifar100_loader,
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
            "test_acc",
            "test_uncertainty",
            "svhn_uncertainty",
            "cifar100_uncertainty",
            "svhn_auroc_val",
            "cifar100_auroc_val",
            "svhn_auroc_test",
            "cifar100_auroc_test",
        ])
    else:
        csv_file = None
        writer = None

    for epoch in range(1, cfg.epochs + 1):
        train_stats = _train_epoch(model, train_loader, num_classes, cfg, optimizer)
        val_acc, val_unc, val_scores = _eval_id(model, val_loader, cfg)
        test_acc, test_unc, test_scores = _eval_id(model, test_loader, cfg)
        svhn_unc, svhn_scores = _eval_ood(model, svhn_loader, cfg)
        cifar100_unc, cifar100_scores = _eval_ood(model, cifar100_loader, cfg)

        svhn_labels = torch.ones_like(svhn_scores)
        val_labels = torch.zeros_like(val_scores)
        test_labels = torch.zeros_like(test_scores)
        svhn_auroc_val = compute_auroc(
            torch.cat([val_scores, svhn_scores]),
            torch.cat([val_labels, svhn_labels]),
        )
        svhn_auroc_test = compute_auroc(
            torch.cat([test_scores, svhn_scores]),
            torch.cat([test_labels, svhn_labels]),
        )

        cifar100_labels = torch.ones_like(cifar100_scores)
        cifar100_auroc_val = compute_auroc(
            torch.cat([val_scores, cifar100_scores]),
            torch.cat([val_labels, cifar100_labels]),
        )
        cifar100_auroc_test = compute_auroc(
            torch.cat([test_scores, cifar100_scores]),
            torch.cat([test_labels, cifar100_labels]),
        )

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d} | loss={train_stats['loss']:.4f} "
                f"risk={train_stats['risk']:.4f} kl={train_stats['kl']:.4f} "
                f"w={train_stats['weight']:.4f} train_acc={train_stats['acc']:.3f} "
                f"val_acc={val_acc:.3f} test_acc={test_acc:.3f} "
                f"svhn_auroc_val={svhn_auroc_val:.3f} cifar100_auroc_val={cifar100_auroc_val:.3f} "
                f"svhn_auroc_test={svhn_auroc_test:.3f} cifar100_auroc_test={cifar100_auroc_test:.3f}"
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
                    "test_acc": test_acc,
                    "test_uncertainty": test_unc,
                    "svhn_uncertainty": svhn_unc,
                    "cifar100_uncertainty": cifar100_unc,
                    "svhn_auroc_val": svhn_auroc_val,
                    "cifar100_auroc_val": cifar100_auroc_val,
                    "svhn_auroc_test": svhn_auroc_test,
                    "cifar100_auroc_test": cifar100_auroc_test,
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
                f"{test_acc:.6f}",
                f"{test_unc:.6f}",
                f"{svhn_unc:.6f}",
                f"{cifar100_unc:.6f}",
                f"{svhn_auroc_val:.6f}",
                f"{cifar100_auroc_val:.6f}",
                f"{svhn_auroc_test:.6f}",
                f"{cifar100_auroc_test:.6f}",
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
