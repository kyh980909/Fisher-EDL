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
    kl_weighted = kl_weight * kl
    total = risk + kl_weighted
    return total.mean(), {
        "risk": risk.mean().item(),
        "kl": kl.mean().item(),
        "kl_weighted": kl_weighted.mean().item(),
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


def evidence_sum(alpha):
    return torch.sum(alpha - 1.0, dim=1, keepdim=True)


def predictive_entropy(alpha, eps=1e-12):
    probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
    return -torch.sum(probs * torch.log(probs.clamp_min(eps)), dim=1, keepdim=True)


def info_measure(alpha, info_type="fisher"):
    info_type = info_type.lower()
    if info_type == "fisher":
        return fisher_info_trace(alpha)
    if info_type == "evidence":
        # More evidence -> more information proxy.
        return evidence_sum(alpha)
    if info_type == "entropy":
        # Lower entropy -> more information; flip sign to keep monotonicity.
        return -predictive_entropy(alpha)
    raise ValueError(f"Unknown info_type: {info_type}")


def gate_weight(info, beta=1.0, gamma=1.0, gate_type="exp", eps=1e-6):
    gate_type = gate_type.lower()
    if gate_type == "exp":
        weight = beta * torch.exp(-gamma * info)
    elif gate_type == "inverse":
        weight = beta / (1.0 + gamma * torch.relu(info))
    elif gate_type == "sigmoid":
        weight = beta * torch.sigmoid(-gamma * info)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")
    return weight.clamp_min(eps)


def fisher_weight(
    alpha,
    beta=1.0,
    gamma=1.0,
    eps=1e-6,
    info_type="fisher",
    gate_type="exp",
):
    info = info_measure(alpha, info_type=info_type)
    return gate_weight(info, beta=beta, gamma=gamma, gate_type=gate_type, eps=eps)


def fisher_edl_mse_loss(
    logits,
    targets,
    beta=1.0,
    gamma=1.0,
    info_type="fisher",
    gate_type="exp",
    detach_weight=False,
    objective="risk_plus_kl",
):
    evidence = F.softplus(logits)
    alpha = evidence + 1.0

    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / sum_alpha
    mse = torch.sum((targets - probs) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (sum_alpha - alpha) / (sum_alpha * sum_alpha * (sum_alpha + 1.0)), dim=1, keepdim=True)
    risk = mse + var

    kl = dirichlet_kl(alpha, targets.shape[1])
    info = info_measure(alpha, info_type=info_type)
    weight = gate_weight(info, beta=beta, gamma=gamma, gate_type=gate_type)
    if detach_weight:
        weight = weight.detach()
    kl_weighted = weight * kl
    if objective == "risk_plus_kl":
        total = risk + kl_weighted
    elif objective == "kl_only":
        total = kl_weighted
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return total.mean(), {
        "risk": risk.mean().item(),
        "kl": kl.mean().item(),
        "kl_weighted": kl_weighted.mean().item(),
        "weight": weight.mean().item(),
        "weight_std": weight.std(unbiased=False).item(),
        "lambda_min": weight.min().item(),
        "lambda_max": weight.max().item(),
        "info": info.mean().item(),
        "info_std": info.std(unbiased=False).item(),
        "fisher_trace": fisher_info_trace(alpha).mean().item(),
        "evidence": evidence.mean().item(),
    }
