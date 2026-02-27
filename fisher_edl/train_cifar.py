import csv
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from fisher_edl.cifar_data import one_hot
from fisher_edl.losses import edl_mse_loss, fisher_edl_mse_loss, fisher_weight
from fisher_edl.metrics import uncertainty_from_logits
from fisher_edl.wandb_loader import import_wandb
from fisher_edl.wandb_utils import import_wandb


@dataclass
class CifarTrainConfig:
    method: str = "fisher"  # ce | mcdropout | edl | fisher
    epochs: int = 100
    lr: float = 1e-3
    beta: float = 1.0
    gamma: float = 1.0
    anneal_kl: bool = False
    anneal_epochs: int = 10
    optimizer: str = "adam"
    weight_decay: float = 0.0
    scheduler: str = "none"
    warmup_epochs: int = 0
    device: str = "cpu"
    log_every: int = 10
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Fisher-ablation controls
    info_type: str = "fisher"  # fisher | evidence | entropy
    gate_type: str = "exp"  # exp | inverse | sigmoid
    detach_weight: bool = False
    objective: str = "risk_plus_kl"  # risk_plus_kl | kl_only


def _safe_mean(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _train_epoch(model, loader, num_classes, cfg, optimizer, epoch):
    model.train()

    if cfg.method == "edl" and cfg.anneal_kl:
        progress = min(1.0, epoch / max(1, cfg.anneal_epochs))
        kl_weight = cfg.beta * progress
    else:
        kl_weight = cfg.beta

    losses = []
    risks = []
    kls = []
    kls_weighted = []
    weights = []
    lambda_min = []
    lambda_max = []
    lambda_std = []
    infos = []
    info_std = []
    fisher_traces = []
    grad_norms = []
    uncertainties = []
    evidence_sums = []
    uncertainty_correct_sum = 0.0
    uncertainty_wrong_sum = 0.0
    count_correct = 0
    count_wrong = 0

    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False)
    for images, labels in progress_bar:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)

        logits = model(images)

        if cfg.method in {"ce", "mcdropout"}:
            loss = F.cross_entropy(logits, labels)
            stats = {
                "risk": loss.item(),
                "kl": 0.0,
                "kl_weighted": 0.0,
                "weight": float("nan"),
                "weight_std": float("nan"),
                "lambda_min": float("nan"),
                "lambda_max": float("nan"),
                "info": float("nan"),
                "info_std": float("nan"),
                "fisher_trace": float("nan"),
            }
        elif cfg.method == "fisher":
            targets = one_hot(labels, num_classes).to(cfg.device)
            loss, stats = fisher_edl_mse_loss(
                logits,
                targets,
                beta=cfg.beta,
                gamma=cfg.gamma,
                info_type=cfg.info_type,
                gate_type=cfg.gate_type,
                detach_weight=cfg.detach_weight,
                objective=cfg.objective,
            )
        else:
            targets = one_hot(labels, num_classes).to(cfg.device)
            loss, stats = edl_mse_loss(logits, targets, kl_weight=kl_weight)
            stats.update(
                {
                    "weight": kl_weight,
                    "weight_std": 0.0,
                    "lambda_min": kl_weight,
                    "lambda_max": kl_weight,
                    "info": float("nan"),
                    "info_std": float("nan"),
                    "fisher_trace": float("nan"),
                }
            )

        optimizer.zero_grad()
        loss.backward()

        grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq += param.grad.detach().data.norm(2).item() ** 2
        grad_norms.append(grad_sq ** 0.5)

        optimizer.step()

        losses.append(float(loss.item()))
        risks.append(float(stats["risk"]))
        kls.append(float(stats["kl"]))
        kls_weighted.append(float(stats["kl_weighted"]))
        weights.append(float(stats["weight"]))
        lambda_min.append(float(stats["lambda_min"]))
        lambda_max.append(float(stats["lambda_max"]))
        lambda_std.append(float(stats["weight_std"]))
        infos.append(float(stats["info"]))
        info_std.append(float(stats["info_std"]))
        fisher_traces.append(float(stats["fisher_trace"]))

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            correct_mask = preds == labels
            wrong_mask = ~correct_mask

            correct += correct_mask.sum().item()
            total += labels.size(0)

            uncertainty = uncertainty_from_logits(logits)
            evidence = F.softplus(logits)
            alpha = evidence + 1.0
            evidence_sum = alpha.sum(dim=1) - num_classes

            uncertainties.append(uncertainty.mean().item())
            evidence_sums.append(evidence_sum.mean().item())

            if correct_mask.any():
                uncertainty_correct_sum += uncertainty[correct_mask].sum().item()
                count_correct += correct_mask.sum().item()
            if wrong_mask.any():
                uncertainty_wrong_sum += uncertainty[wrong_mask].sum().item()
                count_wrong += wrong_mask.sum().item()

        progress_bar.set_postfix(loss=loss.item(), acc=correct / max(1, total))

    return {
        "loss": _safe_mean(losses),
        "risk": _safe_mean(risks),
        "kl": _safe_mean(kls),
        "kl_weighted": _safe_mean(kls_weighted),
        "weight": _safe_mean(weights),
        "kl_weight": kl_weight,
        "lambda_min": _safe_mean(lambda_min),
        "lambda_max": _safe_mean(lambda_max),
        "lambda_std": _safe_mean(lambda_std),
        "info": _safe_mean(infos),
        "info_std": _safe_mean(info_std),
        "fisher_trace": _safe_mean(fisher_traces),
        "uncertainty_mean": _safe_mean(uncertainties),
        "evidence_sum": _safe_mean(evidence_sums),
        "uncertainty_correct": uncertainty_correct_sum / count_correct if count_correct > 0 else float("nan"),
        "uncertainty_wrong": uncertainty_wrong_sum / count_wrong if count_wrong > 0 else float("nan"),
        "grad_norm": _safe_mean(grad_norms),
        "acc": correct / max(1, total),
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
                weights.append(
                    fisher_weight(
                        alpha,
                        beta=cfg.beta,
                        gamma=cfg.gamma,
                        info_type=cfg.info_type,
                        gate_type=cfg.gate_type,
                    )
                )

    all_unc = torch.cat(uncertainties) if uncertainties else torch.empty(0)
    acc = correct / max(1, total)
    mean_unc = all_unc.mean().item() if all_unc.numel() > 0 else float("nan")
    mean_weight = torch.cat(weights).mean().item() if weights else float("nan")
    return acc, mean_unc, mean_weight, all_unc


def train_cifar(
    model,
    train_loader,
    val_loader,
    num_classes,
    cfg: CifarTrainConfig,
    run_dir=None,
):
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler == "cosine":

        def lr_lambda(epoch):
            if epoch < cfg.warmup_epochs:
                return float(epoch + 1) / max(1, cfg.warmup_epochs)
            progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    model.to(cfg.device)

    best_score = float("-inf")
    best_val_path = None

    wandb_run = None
    if cfg.use_wandb:
        wandb = import_wandb(".")

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
                "info_type": cfg.info_type,
                "gate_type": cfg.gate_type,
                "detach_weight": cfg.detach_weight,
                "objective": cfg.objective,
            },
        )

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        cfg_path = os.path.join(run_dir, "config.txt")
        with open(cfg_path, "w", encoding="utf-8") as f:
            for key, value in cfg.__dict__.items():
                f.write(f"{key}: {value}\n")

        csv_path = os.path.join(run_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        last_ckpt_path = os.path.join(run_dir, "last.pt")
        best_val_path = os.path.join(run_dir, "best_val_acc.pt")
        writer.writerow(
            [
                "epoch",
                "loss",
                "risk",
                "kl",
                "kl_weighted",
                "weight",
                "weight_std",
                "lambda_min",
                "lambda_max",
                "info",
                "fisher_trace",
                "train_acc",
                "val_acc",
                "val_uncertainty",
                "val_fisher_weight",
                "grad_norm",
                "epoch_time_sec",
                "elapsed_time_sec",
            ]
        )
    else:
        csv_file = None
        writer = None
        last_ckpt_path = None

    wall_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = _train_epoch(model, train_loader, num_classes, cfg, optimizer, epoch)
        val_acc, val_unc, val_weight, _ = _eval_id(model, val_loader, cfg)

        if scheduler:
            scheduler.step()

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d} | loss={train_stats['loss']:.4f} "
                f"risk={train_stats['risk']:.4f} kl={train_stats['kl']:.4f} "
                f"w={train_stats['weight']:.4f} train_acc={train_stats['acc']:.3f} "
                f"val_acc={val_acc:.3f}"
            )

        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - wall_start

        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "Loss/Total": train_stats["loss"],
                    "Loss/Risk": train_stats["risk"],
                    "Loss/KL_raw": train_stats["kl"],
                    "Loss/KL_weighted": train_stats["kl_weighted"],
                    "Metric/Info": train_stats["info"],
                    "Metric/Info_Std": train_stats["info_std"],
                    "Metric/Fisher_Trace": train_stats["fisher_trace"],
                    "Metric/Lambda_Mean": train_stats["weight"],
                    "Metric/Lambda_Std": train_stats["lambda_std"],
                    "Metric/Lambda_Min": train_stats["lambda_min"],
                    "Metric/Lambda_Max": train_stats["lambda_max"],
                    "Uncertainty/Train_Mean": train_stats["uncertainty_mean"],
                    "Uncertainty/Correct": train_stats["uncertainty_correct"],
                    "Uncertainty/Wrong": train_stats["uncertainty_wrong"],
                    "Evidence/Total_Sum": train_stats["evidence_sum"],
                    "System/Gradient_Norm": train_stats["grad_norm"],
                    "System/Epoch_Time_sec": epoch_time,
                    "train_acc": train_stats["acc"],
                    "val_acc": val_acc,
                    "val_uncertainty": val_unc,
                    "val_fisher_weight": val_weight,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if writer:
            writer.writerow(
                [
                    epoch,
                    f"{train_stats['loss']:.6f}",
                    f"{train_stats['risk']:.6f}",
                    f"{train_stats['kl']:.6f}",
                    f"{train_stats['kl_weighted']:.6f}",
                    f"{train_stats['weight']:.6f}",
                    f"{train_stats['lambda_std']:.6f}",
                    f"{train_stats['lambda_min']:.6f}",
                    f"{train_stats['lambda_max']:.6f}",
                    f"{train_stats['info']:.6f}",
                    f"{train_stats['fisher_trace']:.6f}",
                    f"{train_stats['acc']:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_unc:.6f}",
                    f"{val_weight:.6f}",
                    f"{train_stats['grad_norm']:.6f}",
                    f"{epoch_time:.6f}",
                    f"{elapsed:.6f}",
                ]
            )

        if run_dir:
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "meta": {
                    "train_time_sec": elapsed,
                    "epoch_time_sec": epoch_time,
                    "method": cfg.method,
                },
            }
            torch.save(state, last_ckpt_path)

            score = val_acc
            if not torch.isfinite(torch.tensor(score)):
                score = train_stats["acc"]
            if score > best_score:
                best_score = score
                torch.save(state, best_val_path)

    total_train_time = time.perf_counter() - wall_start

    if csv_file:
        csv_file.close()

    if wandb_run:
        wandb_run.finish()

    return {
        "best_score": best_score,
        "best_ckpt": best_val_path,
        "train_time_sec": total_train_time,
        "method": cfg.method,
    }
