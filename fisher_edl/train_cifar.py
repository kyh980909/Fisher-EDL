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
    anneal_kl: bool = False
    anneal_epochs: int = 10
    device: str = "cpu"
    log_every: int = 10
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


def _train_epoch(model, loader, num_classes, cfg, optimizer, epoch):
    model.train()
    epoch_loss = 0.0
    epoch_risk = 0.0
    epoch_kl = 0.0
    epoch_kl_weighted = 0.0
    epoch_weight = 0.0
    epoch_lambda_min = 0.0
    epoch_lambda_max = 0.0
    epoch_fisher_trace = 0.0
    epoch_uncertainty = 0.0
    epoch_evidence_sum = 0.0
    epoch_uncertainty_correct = 0.0
    epoch_uncertainty_wrong = 0.0
    count_correct = 0
    count_wrong = 0
    epoch_grad_norm = 0.0
    correct = 0
    total = 0

    if cfg.method == "edl" and cfg.anneal_kl:
        progress = min(1.0, epoch / max(1, cfg.anneal_epochs))
        kl_weight = cfg.beta * progress
    else:
        kl_weight = cfg.beta

    for images, labels in loader:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)
        targets = one_hot(labels, num_classes).to(cfg.device)

        logits = model(images)
        if cfg.method == "fisher":
            loss, stats = fisher_edl_mse_loss(logits, targets, beta=cfg.beta, gamma=cfg.gamma)
            epoch_weight += stats["weight"]
            epoch_lambda_min += stats["lambda_min"]
            epoch_lambda_max += stats["lambda_max"]
            epoch_fisher_trace += stats["fisher_trace"]
        else:
            loss, stats = edl_mse_loss(logits, targets, kl_weight=kl_weight)
            epoch_weight += kl_weight
            epoch_lambda_min += float("nan")
            epoch_lambda_max += float("nan")
            epoch_fisher_trace += float("nan")

        optimizer.zero_grad()
        loss.backward()
        grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq += param.grad.detach().data.norm(2).item() ** 2
        epoch_grad_norm += grad_sq ** 0.5
        optimizer.step()

        epoch_loss += loss.item()
        epoch_risk += stats["risk"]
        epoch_kl += stats["kl"]
        epoch_kl_weighted += stats["kl_weighted"]

        with torch.no_grad():
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            correct_mask = preds == labels
            wrong_mask = ~correct_mask
            correct += correct_mask.sum().item()
            total += labels.size(0)

            uncertainty = uncertainty_from_logits(logits)
            evidence = F.softplus(logits)
            alpha = evidence + 1.0
            evidence_sum = alpha.sum(dim=1) - num_classes

            epoch_uncertainty += uncertainty.mean().item()
            epoch_evidence_sum += evidence_sum.mean().item()
            if correct_mask.any():
                epoch_uncertainty_correct += uncertainty[correct_mask].sum().item()
                count_correct += correct_mask.sum().item()
            if wrong_mask.any():
                epoch_uncertainty_wrong += uncertainty[wrong_mask].sum().item()
                count_wrong += wrong_mask.sum().item()

    mean_loss = epoch_loss / max(1, len(loader))
    mean_risk = epoch_risk / max(1, len(loader))
    mean_kl = epoch_kl / max(1, len(loader))
    mean_kl_weighted = epoch_kl_weighted / max(1, len(loader))
    mean_weight = epoch_weight / max(1, len(loader))
    mean_lambda_min = epoch_lambda_min / max(1, len(loader))
    mean_lambda_max = epoch_lambda_max / max(1, len(loader))
    mean_fisher_trace = epoch_fisher_trace / max(1, len(loader))
    mean_uncertainty = epoch_uncertainty / max(1, len(loader))
    mean_evidence_sum = epoch_evidence_sum / max(1, len(loader))
    mean_uncertainty_correct = (
        epoch_uncertainty_correct / count_correct if count_correct > 0 else float("nan")
    )
    mean_uncertainty_wrong = (
        epoch_uncertainty_wrong / count_wrong if count_wrong > 0 else float("nan")
    )
    mean_grad_norm = epoch_grad_norm / max(1, len(loader))
    acc = correct / max(1, total)

    return {
        "loss": mean_loss,
        "risk": mean_risk,
        "kl": mean_kl,
        "kl_weighted": mean_kl_weighted,
        "weight": mean_weight,
        "kl_weight": kl_weight,
        "lambda_min": mean_lambda_min,
        "lambda_max": mean_lambda_max,
        "fisher_trace": mean_fisher_trace,
        "uncertainty_mean": mean_uncertainty,
        "evidence_sum": mean_evidence_sum,
        "uncertainty_correct": mean_uncertainty_correct,
        "uncertainty_wrong": mean_uncertainty_wrong,
        "grad_norm": mean_grad_norm,
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
        cfg_path = os.path.join(run_dir, "config.txt")
        with open(cfg_path, "w") as f:
            for key, value in cfg.__dict__.items():
                f.write(f"{key}: {value}\n")
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
        train_stats = _train_epoch(model, train_loader, num_classes, cfg, optimizer, epoch)
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
                    "Loss/Total": train_stats["loss"],
                    "Loss/Risk": train_stats["risk"],
                    "Loss/KL_raw": train_stats["kl"],
                    "Loss/KL_weighted": train_stats["kl_weighted"],
                    "Metric/Fisher_Trace": train_stats["fisher_trace"],
                    "Metric/Lambda_Mean": train_stats["weight"],
                    "Metric/KL_Weight": train_stats["kl_weight"],
                    "Metric/Lambda_Min": train_stats["lambda_min"],
                    "Metric/Lambda_Max": train_stats["lambda_max"],
                    "Uncertainty/Train_Mean": train_stats["uncertainty_mean"],
                    "Uncertainty/Correct": train_stats["uncertainty_correct"],
                    "Uncertainty/Wrong": train_stats["uncertainty_wrong"],
                    "Evidence/Total_Sum": train_stats["evidence_sum"],
                    "System/Gradient_Norm": train_stats["grad_norm"],
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
