import math

import torch
import torch.nn.functional as F


def uncertainty_from_logits(logits):
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    strength = torch.sum(alpha, dim=1)
    num_classes = alpha.shape[1]
    return num_classes / strength


def probs_from_logits(logits):
    return torch.softmax(logits, dim=1)


def confidence_from_logits(logits):
    probs = probs_from_logits(logits)
    return torch.max(probs, dim=1).values


def energy_from_logits(logits, temperature=1.0):
    temp = max(float(temperature), 1e-6)
    # Larger energy score means more OOD.
    return -temp * torch.logsumexp(logits / temp, dim=1)


def ood_score_from_logits(logits, score_type="uncertainty", temperature=1.0):
    score = score_type.lower()
    if score == "uncertainty":
        return uncertainty_from_logits(logits)
    if score == "energy":
        return energy_from_logits(logits, temperature=temperature)
    if score == "msp":
        return 1.0 - confidence_from_logits(logits)
    raise ValueError(f"Unknown score_type: {score_type}")


def _sort_scores_labels(scores, labels):
    scores = scores.detach().float().cpu().view(-1)
    labels = labels.detach().float().cpu().view(-1)
    sorted_idx = torch.argsort(scores, descending=True)
    return scores[sorted_idx], labels[sorted_idx]


def _binary_pr_curve(scores, labels):
    _, labels = _sort_scores_labels(scores, labels)
    pos = labels.sum().item()
    if pos == 0:
        return None, None
    tps = torch.cumsum(labels, dim=0)
    fps = torch.cumsum(1 - labels, dim=0)
    precision = tps / torch.clamp(tps + fps, min=1.0)
    recall = tps / pos
    return precision, recall


def compute_auroc(scores, labels):
    # labels: 1 for positive, 0 for negative. scores higher => positive.
    _, labels = _sort_scores_labels(scores, labels)
    pos = labels.sum().item()
    neg = labels.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tps = torch.cumsum(labels, dim=0)
    fps = torch.cumsum(1 - labels, dim=0)
    tpr = tps / pos
    fpr = fps / neg
    return torch.trapz(tpr, fpr).item()


def compute_aupr(scores, labels):
    precision, recall = _binary_pr_curve(scores, labels)
    if precision is None:
        return float("nan")
    precision = torch.cat(
        [torch.tensor([1.0], dtype=precision.dtype), precision], dim=0
    )
    recall = torch.cat([torch.tensor([0.0], dtype=recall.dtype), recall], dim=0)
    # PR integration over recall axis
    return torch.trapz(precision, recall).item()


def compute_fpr_at_tpr(scores, labels, tpr_target=0.95):
    _, labels = _sort_scores_labels(scores, labels)
    pos = labels.sum().item()
    neg = labels.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tps = torch.cumsum(labels, dim=0)
    fps = torch.cumsum(1 - labels, dim=0)
    tpr = tps / pos
    fpr = fps / neg
    idx = torch.searchsorted(tpr, torch.tensor(float(tpr_target)), right=False)
    if idx >= tpr.numel():
        idx = tpr.numel() - 1
    return fpr[idx].item()


def compute_nll(probs, labels, eps=1e-12):
    probs = probs.detach().float()
    labels = labels.detach().long()
    mask = labels >= 0
    if not mask.any():
        return float("nan")
    probs = probs[mask]
    labels = labels[mask]
    p = probs[torch.arange(probs.shape[0]), labels].clamp_min(eps)
    return (-torch.log(p)).mean().item()


def compute_brier(probs, labels, num_classes=None):
    probs = probs.detach().float()
    labels = labels.detach().long()
    mask = labels >= 0
    if not mask.any():
        return float("nan")
    probs = probs[mask]
    labels = labels[mask]
    if num_classes is None:
        num_classes = probs.shape[1]
    targets = torch.zeros_like(probs)
    targets[torch.arange(labels.shape[0]), labels] = 1.0
    return torch.mean(torch.sum((probs - targets) ** 2, dim=1)).item()


def reliability_bins(probs, labels, n_bins=15):
    probs = probs.detach().float()
    labels = labels.detach().long()
    mask = labels >= 0
    if not mask.any():
        return []
    probs = probs[mask]
    labels = labels[mask]

    conf, preds = torch.max(probs, dim=1)
    correct = (preds == labels).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo = bins[i].item()
        hi = bins[i + 1].item()
        if i == n_bins - 1:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)
        count = in_bin.sum().item()
        if count == 0:
            rows.append(
                {
                    "bin": i,
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": 0,
                    "confidence": float("nan"),
                    "accuracy": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue
        mean_conf = conf[in_bin].mean().item()
        mean_acc = correct[in_bin].mean().item()
        rows.append(
            {
                "bin": i,
                "bin_left": lo,
                "bin_right": hi,
                "count": int(count),
                "confidence": mean_conf,
                "accuracy": mean_acc,
                "gap": abs(mean_conf - mean_acc),
            }
        )
    return rows


def compute_ece(probs, labels, n_bins=15):
    rows = reliability_bins(probs, labels, n_bins=n_bins)
    if not rows:
        return float("nan")
    total = sum(r["count"] for r in rows)
    if total == 0:
        return float("nan")
    ece = 0.0
    for row in rows:
        if row["count"] == 0:
            continue
        ece += (row["count"] / total) * row["gap"]
    return ece


def compute_classwise_ece(probs, labels, n_bins=15):
    probs = probs.detach().float()
    labels = labels.detach().long()
    mask = labels >= 0
    if not mask.any():
        return float("nan")
    probs = probs[mask]
    labels = labels[mask]
    num_classes = probs.shape[1]
    class_eces = []
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    for cls in range(num_classes):
        cls_prob = probs[:, cls]
        cls_true = (labels == cls).float()
        total = cls_prob.numel()
        if total == 0:
            continue
        cls_ece = 0.0
        for i in range(n_bins):
            lo = bins[i]
            hi = bins[i + 1]
            if i == n_bins - 1:
                in_bin = (cls_prob >= lo) & (cls_prob <= hi)
            else:
                in_bin = (cls_prob >= lo) & (cls_prob < hi)
            count = in_bin.sum().item()
            if count == 0:
                continue
            conf = cls_prob[in_bin].mean().item()
            acc = cls_true[in_bin].mean().item()
            cls_ece += (count / total) * abs(conf - acc)
        class_eces.append(cls_ece)
    if not class_eces:
        return float("nan")
    return float(sum(class_eces) / len(class_eces))


def compute_misclassification_auroc(logits, labels):
    logits = logits.detach().float()
    labels = labels.detach().long()
    mask = labels >= 0
    if not mask.any():
        return float("nan")
    logits = logits[mask]
    labels = labels[mask]
    preds = torch.argmax(logits, dim=1)
    errors = (preds != labels).float()
    score = ood_score_from_logits(logits, score_type="msp")
    return compute_auroc(score, errors)


def compute_aurc(logits, labels):
    logits = logits.detach().float()
    labels = labels.detach().long()
    mask = labels >= 0
    if not mask.any():
        return float("nan")
    logits = logits[mask]
    labels = labels[mask]
    conf = confidence_from_logits(logits)
    preds = torch.argmax(logits, dim=1)
    error = (preds != labels).float()
    order = torch.argsort(conf, descending=True)
    error = error[order]
    cumulative_error = torch.cumsum(error, dim=0)
    n = error.numel()
    coverage = torch.arange(1, n + 1, dtype=torch.float32) / n
    risk = cumulative_error / torch.arange(1, n + 1, dtype=torch.float32)
    return torch.trapz(risk, coverage).item()


def fit_temperature(logits, labels, max_iter=50):
    logits = logits.detach().float()
    labels = labels.detach().long()
    mask = labels >= 0
    if not mask.any():
        return 1.0
    logits = logits[mask]
    labels = labels[mask]
    log_temp = torch.zeros(1, requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(1e-3, 1e3)
        loss = F.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp).clamp(1e-3, 1e3).item())


def summarize_id_metrics(logits, labels, n_bins=15):
    probs = probs_from_logits(logits)
    preds = torch.argmax(probs, dim=1)
    mask = labels >= 0
    if mask.any():
        acc = (preds[mask] == labels[mask]).float().mean().item()
    else:
        acc = float("nan")
    return {
        "acc": acc,
        "nll": compute_nll(probs, labels),
        "brier": compute_brier(probs, labels),
        "ece": compute_ece(probs, labels, n_bins=n_bins),
        "classwise_ece": compute_classwise_ece(probs, labels, n_bins=n_bins),
        "miscls_auroc": compute_misclassification_auroc(logits, labels),
        "aurc": compute_aurc(logits, labels),
    }


def compute_ci95(values):
    vals = [float(v) for v in values if not math.isnan(float(v))]
    n = len(vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    return 1.96 * std / math.sqrt(n)
