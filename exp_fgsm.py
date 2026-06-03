"""
FGSM adversarial evaluation for already trained MNIST models.

This script loads final checkpoints from an existing static-corrupted-MNIST
experiment directory and evaluates them under untargeted FGSM attacks.

It does not train or resume models. It expects run directories such as

    <trained_dir>/runs/pc_K0__seed0/
    <trained_dir>/runs/passive_astro_pc__seed0/
    <trained_dir>/runs/full_astro_pc__seed0/

or, as a fallback,

    <trained_dir>/pc_K0__seed0/

The attack is applied in raw pixel space with epsilon in [0, 1], then images
are clamped to [0, 1] and normalized before evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from checkpointing import load_training_checkpoint
from config import Config
from train import build_model


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

VARIANT_LABELS = {
    "pc_K0": "Predictive coding",
    "passive_astro_pc": "Passive Astro-PC null",
    "full_astro_pc": "Full Astro-PC",
}

VARIANT_ORDER = {
    "pc_K0": 0,
    "passive_astro_pc": 1,
    "full_astro_pc": 2,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_csv_ints(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return vals


def parse_csv_floats(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def parse_csv_strings(text: str) -> list[str]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one string value.")
    return vals


def torch_dtype_from_string(value: str) -> torch.dtype:
    mapping = {
        "torch.float32": torch.float32,
        "float32": torch.float32,
        "torch.float64": torch.float64,
        "float64": torch.float64,
        "torch.float16": torch.float16,
        "float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(str(value), torch.float32)


def load_cfg_from_run(run_dir: Path, *, device: str | None = None) -> Config:
    """
    Load Config from run_dir/config.json when available.

    This is safer than reconstructing from current defaults, because the local
    config.py may have changed after the model was trained.
    """
    cfg = Config()
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        # Fallback for old runs.
        if device is not None:
            cfg = replace(cfg, dataset="mnist", method="pc", device=device)
        else:
            cfg = replace(cfg, dataset="mnist", method="pc")
        cfg.validate()
        return cfg

    with cfg_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    valid = {f.name for f in fields(Config)}
    updates: dict[str, Any] = {}

    for key, value in payload.items():
        if key not in valid:
            continue
        if key == "dtype":
            updates[key] = torch_dtype_from_string(value)
        elif key == "hidden_dims" and isinstance(value, list):
            updates[key] = tuple(value)
        else:
            updates[key] = value

    if device is not None:
        updates["device"] = device
    elif not torch.cuda.is_available():
        updates["device"] = "cpu"

    cfg = replace(cfg, **updates)
    cfg.validate()
    return cfg


def find_run_dir(trained_dir: Path, variant: str, seed: int) -> Path:
    candidates = [
        trained_dir / "runs" / f"{variant}__seed{seed}",
        trained_dir / f"{variant}__seed{seed}",
    ]
    if seed == 0:
        candidates.extend(
            [
                trained_dir / "runs" / variant,
                trained_dir / variant,
            ]
        )

    for candidate in candidates:
        if (candidate / "checkpoints" / "checkpoint_last.pt").exists():
            return candidate

    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find checkpoint for variant={variant!r}, seed={seed}.\n"
        f"Searched:\n{searched}"
    )


def load_model_for_run(run_dir: Path, cfg: Config) -> torch.nn.Module:
    device = torch.device(cfg.device)
    model = build_model(cfg, device)
    ckpt = load_training_checkpoint(
        run_dir,
        model=model,
        bp_optimizer=None,
        device=device,
    )
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {run_dir / 'checkpoints'}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_mnist_test_loader(
    *,
    root: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device_type: str,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device_type == "cuda",
        drop_last=False,
    )


def clamp_normalized_mnist(x_norm: torch.Tensor) -> torch.Tensor:
    lo = (0.0 - MNIST_MEAN) / MNIST_STD
    hi = (1.0 - MNIST_MEAN) / MNIST_STD
    return x_norm.clamp(lo, hi)


def fgsm_batch(
    model: torch.nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """
    Untargeted FGSM in raw pixel coordinates.

    Since x_norm = (x_raw - mean) / std, an epsilon step in raw space
    corresponds to epsilon / std in normalized space.
    """
    if epsilon <= 0.0:
        return x_norm.detach()

    x_adv = x_norm.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

    # sign(grad_raw) == sign(grad_norm) because std > 0.
    x_adv = x_adv.detach() + (float(epsilon) / MNIST_STD) * grad.sign()
    x_adv = clamp_normalized_mnist(x_adv)
    return x_adv.detach()


@torch.no_grad()
def evaluate_clean(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += int(y.numel())

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


def evaluate_fgsm(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    epsilon: float,
) -> tuple[float, float]:
    if epsilon <= 0.0:
        return evaluate_clean(model, loader, device)

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    model.eval()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_adv = fgsm_batch(model, x, y, epsilon=epsilon)

        with torch.no_grad():
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y, reduction="sum")

        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += int(y.numel())

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


def make_summary_tables(raw_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_dir / "fgsm_mnist_raw_results.csv", index=False)

    summary = (
        raw_df.groupby(["model_name", "model_label", "epsilon"], as_index=False)
        .agg(
            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),
            mean_loss=("test_loss", "mean"),
            std_loss=("test_loss", "std"),
            num_seeds=("seed", "nunique"),
        )
        .fillna({"std_acc": 0.0, "std_loss": 0.0})
    )
    summary["mean_acc_pct"] = 100.0 * summary["mean_acc"]
    summary["std_acc_pct"] = 100.0 * summary["std_acc"]
    summary["mean_loss"] = summary["mean_loss"].astype(float)
    summary["std_loss"] = summary["std_loss"].astype(float)
    summary["model_order"] = summary["model_name"].map(VARIANT_ORDER)
    summary = summary.sort_values(["model_order", "epsilon"])
    summary.to_csv(out_dir / "fgsm_mnist_summary_long.csv", index=False)

    # Machine-readable wide accuracy table.
    wide_rows: list[dict[str, Any]] = []
    for (model_name, model_label), sub in summary.groupby(
        ["model_name", "model_label"], sort=False
    ):
        row: dict[str, Any] = {
            "model_name": model_name,
            "model_label": model_label,
        }
        for _, rec in sub.iterrows():
            eps_key = f"eps_{float(rec['epsilon']):.3f}".replace(".", "p")
            row[f"{eps_key}_mean_pct"] = float(rec["mean_acc_pct"])
            row[f"{eps_key}_std_pct"] = float(rec["std_acc_pct"])
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    wide.to_csv(out_dir / "fgsm_mnist_accuracy_table.csv", index=False)

    # Paper-facing LaTeX table.
    eps_values = sorted(raw_df["epsilon"].unique())

    def fmt(mean_pct: float, std_pct: float) -> str:
        return f"{mean_pct:.2f} $\\pm$ {std_pct:.2f}"

    lines: list[str] = []
    lines.append(r"\begin{table}[!htb]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{FGSM adversarial accuracy on MNIST over seeds. Values are percentages reported as mean \(\pm\) standard deviation. The attack budget \(\epsilon\) is measured in raw pixel units in \([0,1]\).}"
    )
    lines.append(r"\resizebox{\linewidth}{!}{")
    colspec = "l" + "c" * len(eps_values)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    header = (
        "Model & " + " & ".join([rf"$\epsilon={eps:g}$" for eps in eps_values]) + r" \\"
    )
    lines.append(header)
    lines.append(r"\midrule")

    for _, row in wide.iterrows():
        vals: list[str] = []
        for eps in eps_values:
            eps_key = f"eps_{float(eps):.3f}".replace(".", "p")
            vals.append(
                fmt(float(row[f"{eps_key}_mean_pct"]), float(row[f"{eps_key}_std_pct"]))
            )
        lines.append(f"{row['model_label']} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\label{tab:fgsm_mnist_accuracy}")
    lines.append(r"\end{table}")
    (out_dir / "fgsm_mnist_accuracy_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FGSM adversarial evaluation for already trained MNIST models."
    )
    parser.add_argument(
        "--trained_dir",
        default="plots/mnist_static_corrupted_5seed_v1",
        help="Directory containing trained runs, usually the static corrupted-MNIST experiment directory.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to <trained_dir>/fgsm_eval.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds.")
    parser.add_argument(
        "--variants",
        default="pc_K0,passive_astro_pc,full_astro_pc",
        help="Comma-separated variants to evaluate.",
    )
    parser.add_argument(
        "--epsilons",
        default="0,0.025,0.05,0.1,0.15,0.2,0.3",
        help="Comma-separated FGSM epsilons in raw pixel units [0,1].",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Override device, e.g. 'cuda' or 'cpu'. Defaults to saved config unless CUDA is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trained_dir = Path(args.trained_dir)
    out_dir = (
        Path(args.out_dir) if args.out_dir is not None else trained_dir / "fgsm_eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_csv_ints(args.seeds)
    variants = parse_csv_strings(args.variants)
    epsilons = parse_csv_floats(args.epsilons)

    unknown = [v for v in variants if v not in VARIANT_LABELS]
    if unknown:
        raise ValueError(
            f"Unknown variants {unknown}. Valid options: {list(VARIANT_LABELS)}"
        )

    manifest = {
        "description": "FGSM adversarial evaluation of already trained MNIST models.",
        "trained_dir": str(trained_dir),
        "seeds": seeds,
        "variants": variants,
        "epsilons_raw_pixel_units": epsilons,
        "mnist_mean": MNIST_MEAN,
        "mnist_std": MNIST_STD,
        "note": "FGSM is applied in raw pixel space and projected to [0,1] before normalization.",
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)

    all_rows: list[dict[str, Any]] = []

    # Reuse one MNIST test loader; all evaluated runs share the same input space.
    first_run = find_run_dir(trained_dir, variants[0], seeds[0])
    first_cfg = load_cfg_from_run(first_run, device=args.device)
    if args.num_workers is not None:
        first_cfg = replace(first_cfg, num_workers=int(args.num_workers))

    loader = make_mnist_test_loader(
        root=first_cfg.mnist_root,
        batch_size=args.batch_size,
        num_workers=first_cfg.num_workers,
        pin_memory=first_cfg.pin_memory,
        device_type=torch.device(first_cfg.device).type,
    )

    for seed in seeds:
        set_seed(seed)
        for variant in variants:
            run_dir = find_run_dir(trained_dir, variant, seed)
            cfg = load_cfg_from_run(run_dir, device=args.device)
            if args.num_workers is not None:
                cfg = replace(cfg, num_workers=int(args.num_workers))

            print("\n" + "=" * 80)
            print(f"Evaluating {VARIANT_LABELS[variant]} | seed={seed}")
            print(f"Run dir: {run_dir}")
            print("=" * 80)

            model = load_model_for_run(run_dir, cfg)
            device = torch.device(cfg.device)

            for eps in epsilons:
                loss, acc = evaluate_fgsm(model, loader, device, epsilon=float(eps))
                row = {
                    "seed": int(seed),
                    "model_name": variant,
                    "model_label": VARIANT_LABELS[variant],
                    "epsilon": float(eps),
                    "test_loss": float(loss),
                    "test_acc": float(acc),
                }
                all_rows.append(row)
                print(f"  eps={eps:g}: loss={loss:.4f}, acc={acc:.4f}")

            # Persist intermediate rows for interrupted evaluations.
            pd.DataFrame(all_rows).to_csv(
                out_dir / "fgsm_mnist_raw_results.csv", index=False
            )

    raw_df = pd.DataFrame(all_rows)
    make_summary_tables(raw_df, out_dir)

    print("\nSaved outputs:")
    print(f"  {out_dir / 'fgsm_mnist_raw_results.csv'}")
    print(f"  {out_dir / 'fgsm_mnist_summary_long.csv'}")
    print(f"  {out_dir / 'fgsm_mnist_accuracy_table.csv'}")
    print(f"  {out_dir / 'fgsm_mnist_accuracy_table.tex'}")


if __name__ == "__main__":
    main()
