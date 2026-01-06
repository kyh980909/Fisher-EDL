import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class OODWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, -1


def one_hot(labels, num_classes):
    targets = torch.zeros((labels.shape[0], num_classes), dtype=torch.float32)
    mask = labels >= 0
    targets[mask, labels[mask]] = 1.0
    return targets


def build_cifar10_loaders(
    batch_size=128,
    data_root="./data",
    num_workers=2,
    val_split=0.1,
    seed=1234,
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    full_train_set = datasets.CIFAR10(
        root=data_root, train=True, transform=train_transform, download=True
    )
    test_set = datasets.CIFAR10(
        root=data_root, train=False, transform=test_transform, download=True
    )

    if val_split > 0:
        val_size = int(len(full_train_set) * val_split)
        train_size = len(full_train_set) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = torch.utils.data.random_split(
            full_train_set, [train_size, val_size], generator=generator
        )
    else:
        train_set = full_train_set
        val_set = None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    return train_loader, val_loader, test_loader, test_transform


def build_svhn_loader(batch_size=128, data_root="./data", num_workers=2, transform=None):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    svhn = datasets.SVHN(
        root=data_root, split="test", transform=transform, download=True
    )
    svhn = OODWrapper(svhn)
    loader = DataLoader(svhn, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


def build_cifar100_loader(batch_size=128, data_root="./data", num_workers=2, transform=None):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    cifar100 = datasets.CIFAR100(
        root=data_root, train=False, transform=transform, download=True
    )
    cifar100 = OODWrapper(cifar100)
    loader = DataLoader(
        cifar100, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return loader
