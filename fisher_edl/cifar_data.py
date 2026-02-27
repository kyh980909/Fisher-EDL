import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_OOD_GROUPS = {
    "svhn": "far",
    "cifar100": "near",
    "texture": "far",
    "dtd": "far",
    "lsun-crop": "near",
    "lsun-resize": "near",
    "isun": "near",
    "places365": "near",
}


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


def parse_ood_datasets(spec):
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [str(x).strip().lower() for x in spec if str(x).strip()]
    return [x.strip().lower() for x in str(spec).split(",") if x.strip()]


def ood_group(name):
    return DEFAULT_OOD_GROUPS.get(name.lower(), "unknown")


def _build_cifar_test_transform(image_size=32):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_cifar10_loaders(
    batch_size=128,
    data_root="./data",
    num_workers=2,
    val_split=0.1,
    seed=1234,
    image_size=32,
):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = _build_cifar_test_transform(image_size=image_size)

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


def _build_folder_ood_loader(folder_path, batch_size, num_workers, transform):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"OOD folder not found: {folder_path}. "
            "Prepare folder-structured images under this path."
        )
    ds = datasets.ImageFolder(root=folder_path, transform=transform)
    ds = OODWrapper(ds)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def build_ood_loader(name, batch_size=128, data_root="./data", num_workers=2, transform=None):
    name = name.lower().strip()
    if transform is None:
        transform = _build_cifar_test_transform(image_size=32)

    if name == "svhn":
        return build_svhn_loader(
            batch_size=batch_size,
            data_root=data_root,
            num_workers=num_workers,
            transform=transform,
        )
    if name == "cifar100":
        return build_cifar100_loader(
            batch_size=batch_size,
            data_root=data_root,
            num_workers=num_workers,
            transform=transform,
        )
    if name in {"texture", "dtd"}:
        dtd = datasets.DTD(root=data_root, split="test", transform=transform, download=True)
        return DataLoader(OODWrapper(dtd), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if name == "places365":
        places_path = os.path.join(data_root, "places365")
        return _build_folder_ood_loader(places_path, batch_size, num_workers, transform)
    if name in {"lsun-crop", "lsun-resize", "isun"}:
        folder = os.path.join(data_root, "ood", name)
        return _build_folder_ood_loader(folder, batch_size, num_workers, transform)

    folder = os.path.join(data_root, "ood", name)
    return _build_folder_ood_loader(folder, batch_size, num_workers, transform)


def build_ood_loaders(
    ood_datasets,
    batch_size=128,
    data_root="./data",
    num_workers=2,
    transform=None,
    skip_missing=False,
):
    names = parse_ood_datasets(ood_datasets)
    loaders = {}
    for name in names:
        try:
            loaders[name] = build_ood_loader(
                name=name,
                batch_size=batch_size,
                data_root=data_root,
                num_workers=num_workers,
                transform=transform,
            )
        except FileNotFoundError:
            if not skip_missing:
                raise
            print(f"[WARN] Skip missing OOD dataset: {name}")
    return loaders
