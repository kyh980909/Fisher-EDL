import math
import torch
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def _make_gaussian_cluster(center, n, std, generator):
    return center + std * torch.randn((n, 2), generator=generator)


def make_toy_split(
    num_classes=3,
    train_per_class=600,
    val_per_class=200,
    ood_train=0,
    ood_val=200,
    hard_overlap=0.35,
    seed=1234,
):
    """Creates a 2D toy dataset with optional hard negatives and OOD points."""
    gen = torch.Generator().manual_seed(seed)
    angle_step = 2.0 * math.pi / num_classes
    radius = 3.0

    centers = []
    for i in range(num_classes):
        angle = i * angle_step
        centers.append(torch.tensor([math.cos(angle) * radius, math.sin(angle) * radius]))

    train_features = []
    train_labels = []
    val_features = []
    val_labels = []

    for k, center in enumerate(centers):
        train_features.append(_make_gaussian_cluster(center, train_per_class, std=0.6, generator=gen))
        val_features.append(_make_gaussian_cluster(center, val_per_class, std=0.6, generator=gen))
        train_labels.append(torch.full((train_per_class,), k, dtype=torch.long))
        val_labels.append(torch.full((val_per_class,), k, dtype=torch.long))

    # Hard negatives: overlapping clusters between class 0 and 1.
    if num_classes >= 2 and hard_overlap > 0:
        overlap_n = int(train_per_class * hard_overlap)
        mid = (centers[0] + centers[1]) / 2.0
        train_features.append(_make_gaussian_cluster(mid, overlap_n, std=0.4, generator=gen))
        train_labels.append(torch.full((overlap_n,), 0, dtype=torch.long))

    if ood_train > 0:
        ood = (torch.rand((ood_train, 2), generator=gen) - 0.5) * 12.0
        train_features.append(ood)
        train_labels.append(torch.full((ood_train,), -1, dtype=torch.long))

    if ood_val > 0:
        ood = (torch.rand((ood_val, 2), generator=gen) - 0.5) * 12.0
        val_features.append(ood)
        val_labels.append(torch.full((ood_val,), -1, dtype=torch.long))

    train_x = torch.cat(train_features, dim=0)
    train_y = torch.cat(train_labels, dim=0)
    val_x = torch.cat(val_features, dim=0)
    val_y = torch.cat(val_labels, dim=0)

    return train_x, train_y, val_x, val_y


def one_hot(labels, num_classes):
    targets = torch.zeros((labels.shape[0], num_classes), dtype=torch.float32)
    mask = labels >= 0
    targets[mask, labels[mask]] = 1.0
    return targets


def build_loaders(
    batch_size=128,
    num_classes=3,
    train_per_class=600,
    val_per_class=200,
    ood_train=0,
    ood_val=200,
    hard_overlap=0.35,
    seed=1234,
):
    train_x, train_y, val_x, val_y = make_toy_split(
        num_classes=num_classes,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        ood_train=ood_train,
        ood_val=ood_val,
        hard_overlap=hard_overlap,
        seed=seed,
    )

    train_ds = ToyDataset(train_x, train_y)
    val_ds = ToyDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
