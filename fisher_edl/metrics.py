import torch


def uncertainty_from_logits(logits):
    evidence = torch.nn.functional.softplus(logits)
    alpha = evidence + 1.0
    strength = torch.sum(alpha, dim=1)
    num_classes = alpha.shape[1]
    uncertainty = num_classes / strength
    return uncertainty


def compute_auroc(scores, labels):
    # labels: 1 for OOD, 0 for ID. scores higher => more OOD.
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()
    sorted_idx = torch.argsort(scores, descending=True)
    labels = labels[sorted_idx]

    pos = labels.sum().item()
    neg = labels.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")

    tps = torch.cumsum(labels, dim=0)
    fps = torch.cumsum(1 - labels, dim=0)

    tpr = tps / pos
    fpr = fps / neg

    # Trapezoidal integration
    auroc = torch.trapz(tpr, fpr).item()
    return auroc
