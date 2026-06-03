from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def _save(out_dir: str | Path, filename: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_loss(
    train_losses: Sequence[float],
    test_losses: Sequence[float],
    *,
    out_dir: str | Path,
    filename: str = "loss.png",
) -> Path:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, test_losses, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save(out_dir, filename)


def plot_accuracy(
    train_accs: Sequence[float],
    test_accs: Sequence[float],
    *,
    out_dir: str | Path,
    filename: str = "accuracy.png",
) -> Path:
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="train")
    plt.plot(epochs, test_accs, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save(out_dir, filename)


def plot_astro_leakage_metrics(
    method_metrics: list[dict],
    *,
    out_dir: str | Path,
    prefix: str = "",
) -> list[Path]:
    """Compatibility hook for train.py."""
    return []
