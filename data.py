from __future__ import annotations

import torch
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset


def make_mnist_loaders(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device_type: str,
    root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    # Keep torchvision optional for non-MNIST experiments.
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device_type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device_type == "cuda",
        drop_last=False,
    )

    return train_loader, test_loader


class NoisyXORDataset(Dataset):
    """Noisy XOR samples on {0, 1}^2."""

    def __init__(
        self,
        n: int,
        *,
        noise_std: float = 0.1,
        seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        x = torch.randint(0, 2, size=(n, 2), generator=g).to(dtype=dtype)
        y = x[:, 0].to(torch.int64) ^ x[:, 1].to(torch.int64)

        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x, generator=g)

        if device is not None:
            x = x.to(device)
            y = y.to(device)

        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def make_xor_loaders(
    *,
    train_size: int = 10_000,
    test_size: int = 2_000,
    batch_size: int = 256,
    noise_std: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
    device_type: str = "cpu",
) -> tuple[DataLoader, DataLoader]:
    train_ds = NoisyXORDataset(
        train_size,
        noise_std=noise_std,
        seed=seed,
        device=None,
        dtype=torch.float32,
    )
    test_ds = NoisyXORDataset(
        test_size,
        noise_std=noise_std,
        seed=seed + 1,
        device=None,
        dtype=torch.float32,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device_type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device_type == "cuda",
        drop_last=False,
    )
    return train_loader, test_loader


def _make_tabular_loaders(
    x: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device_type: str,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(x["train"], y["train"])
    test_ds = TensorDataset(x["test"], y["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device_type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device_type == "cuda",
        drop_last=False,
    )
    return train_loader, test_loader


def _prepare_tabular_classification_data(
    x,
    y,
    *,
    test_size: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_tensors = {
        "train": torch.tensor(x_train, dtype=torch.float32),
        "test": torch.tensor(x_test, dtype=torch.float32),
    }
    y_tensors = {
        "train": torch.tensor(y_train, dtype=torch.int64),
        "test": torch.tensor(y_test, dtype=torch.int64),
    }
    return x_tensors, y_tensors


def make_wine_loaders(
    *,
    batch_size: int = 128,
    test_size: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
    device_type: str = "cpu",
) -> tuple[DataLoader, DataLoader]:
    data = load_wine()
    x_tensors, y_tensors = _prepare_tabular_classification_data(
        data.data,
        data.target,
        test_size=test_size,
        seed=seed,
    )
    return _make_tabular_loaders(
        x_tensors,
        y_tensors,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device_type=device_type,
    )


def make_breast_cancer_loaders(
    *,
    batch_size: int = 128,
    test_size: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
    device_type: str = "cpu",
) -> tuple[DataLoader, DataLoader]:
    data = load_breast_cancer()
    x_tensors, y_tensors = _prepare_tabular_classification_data(
        data.data,
        data.target,
        test_size=test_size,
        seed=seed,
    )
    return _make_tabular_loaders(
        x_tensors,
        y_tensors,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device_type=device_type,
    )


def make_data_loaders(cfg, device: torch.device) -> tuple[DataLoader, DataLoader]:
    if cfg.dataset == "mnist":
        return make_mnist_loaders(**cfg.dataset_kwargs(device_type=device.type))
    if cfg.dataset == "xor":
        return make_xor_loaders(**cfg.dataset_kwargs(device_type=device.type))
    if cfg.dataset == "wine":
        return make_wine_loaders(**cfg.dataset_kwargs(device_type=device.type))
    if cfg.dataset == "breast_cancer":
        return make_breast_cancer_loaders(**cfg.dataset_kwargs(device_type=device.type))
    raise ValueError(f"Unknown dataset: {cfg.dataset}")
