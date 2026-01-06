import math
import torch
import torch.nn.functional as F


def dirichlet_kl(alpha, num_classes):
    """KL(Dir(alpha) || Dir(1)) for uniform prior."""
    device = alpha.device
    one = torch.ones((1, num_classes), device=device, dtype=alpha.dtype)

    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_one = torch.sum(one, dim=1, keepdim=True)

    ln_b_alpha = (
        torch.lgamma(sum_alpha)
        - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    )
    ln_b_one = (
        torch.lgamma(sum_one)
        - torch.sum(torch.lgamma(one), dim=1, keepdim=True)
    )

    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)

    kl = torch.sum(
        (alpha - one) * (digamma_alpha - digamma_sum), dim=1, keepdim=True
    ) + ln_b_alpha - ln_b_one
    return kl


def edl_mse_loss(logits, targets, kl_weight=1.0):
    """
    Evidential MSE loss from EDL literature.
    targets is one-hot.
    """
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / sum_alpha

    mse = torch.sum((targets - probs) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (sum_alpha - alpha) / (sum_alpha * sum_alpha * (sum_alpha + 1.0)), dim=1, keepdim=True)
    risk = mse + var

    kl = dirichlet_kl(alpha, targets.shape[1])
    total = risk + kl_weight * kl
    return total.mean(), {
        "risk": risk.mean().item(),
        "kl": kl.mean().item(),
        "evidence": evidence.mean().item(),
    }


def fisher_info_trace(alpha):
    """Trace of Dirichlet Fisher Information Matrix."""
    num_classes = alpha.shape[1]
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    trigamma_alpha = torch.polygamma(1, alpha)
    trigamma_sum = torch.polygamma(1, sum_alpha)
    trace = torch.sum(trigamma_alpha, dim=1, keepdim=True) - num_classes * trigamma_sum
    return trace


def fisher_weight(alpha, beta=1.0, gamma=1.0, eps=1e-6):
    info = fisher_info_trace(alpha)
    # Larger info -> smaller weight
    return beta * torch.exp(-gamma * info).clamp_min(eps)


def fisher_edl_mse_loss(logits, targets, beta=1.0, gamma=1.0):
    evidence = F.softplus(logits)
    alpha = evidence + 1.0

    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / sum_alpha
    mse = torch.sum((targets - probs) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (sum_alpha - alpha) / (sum_alpha * sum_alpha * (sum_alpha + 1.0)), dim=1, keepdim=True)
    risk = mse + var

    kl = dirichlet_kl(alpha, targets.shape[1])
    weight = fisher_weight(alpha, beta=beta, gamma=gamma)
    total = risk + weight * kl

    return total.mean(), {
        "risk": risk.mean().item(),
        "kl": kl.mean().item(),
        "weight": weight.mean().item(),
        "evidence": evidence.mean().item(),
    }
