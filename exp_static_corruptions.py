"""
Static clean-training / corrupted-test MNIST experiment.

This script trains each model variant on clean MNIST and evaluates the final
checkpoint on clean and corrupted MNIST test sets. It is intended for the paper
table reporting final clean and corrupted accuracy averaged over seeds. The
default variant set includes the additive component ablations used for the
Astro-PC robustness table.

Compared with run_corruption_context_mnist_multicontext.py, this script does
NOT do context switching and does NOT append context cues. It trains on clean
MNIST only, then evaluates under deterministic test-time corruptions.

Default corruption parameters match the multi-context script defaults/command:
    gaussian_std = 0.275
    motion_kernel = 11
    pixel_size = 5
    brightness_delta = 0.25

Outputs:
    <out_dir>/corrupted_mnist_raw_results.csv
    <out_dir>/corrupted_mnist_summary_long.csv
    <out_dir>/corrupted_mnist_accuracy_table.csv
    <out_dir>/corrupted_mnist_accuracy_table.tex
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from checkpointing import load_training_checkpoint
from config import Config
from engine import evaluate
from train import build_model, main as train_main


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

CORRUPTION_LABELS = {
    "clean": "Clean",
    "gaussian_noise": "Gaussian",
    "motion_blur": "Motion blur",
    "brightness": "Brightness",
    "pixelation": "Pixelation",
}

CORRUPTION_ORDER = [
    "clean",
    "gaussian_noise",
    "motion_blur",
    "brightness",
    "pixelation",
]
CORRUPTED_ONLY = ["gaussian_noise", "motion_blur", "brightness", "pixelation"]

VARIANT_LABELS = {
    "pc_K0": "PC",
    "pc_memory_K2": "PC + memory",
    "astro_gain_only": "Astro gain only",
    "astro_gain_state_coupling": "Astro gain + state coupling",
    "astro_gain_state_coupling_leakage": "Astro gain + state coupling + leakage",
    "passive_astro_pc": "Passive Astro-PC null",
    "full_astro_pc": "Full Astro-PC",
}

DEFAULT_VARIANTS = tuple(VARIANT_LABELS)

VARIANT_ORDER = {name: idx for idx, name in enumerate(DEFAULT_VARIANTS)}


def parse_csv_ints(text: str) -> list[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def parse_csv_strings(text: str) -> list[str]:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one string value is required.")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_corruption(
    x: torch.Tensor,
    *,
    corruption: str,
    idx: int,
    noise_seed: int,
    gaussian_std: float,
    brightness_delta: float,
    motion_kernel: int,
    pixel_size: int,
) -> torch.Tensor:
    """Apply a deterministic corruption to a [1, 28, 28] tensor in [0, 1]."""
    if corruption == "clean":
        return x

    if corruption == "gaussian_noise":
        g = torch.Generator().manual_seed(int(noise_seed) + int(idx))
        noise = torch.randn(x.shape, generator=g, dtype=x.dtype) * float(gaussian_std)
        return (x + noise).clamp(0.0, 1.0)

    if corruption == "brightness":
        return (x + float(brightness_delta)).clamp(0.0, 1.0)

    if corruption == "motion_blur":
        k = int(motion_kernel)
        if k <= 1:
            return x
        if k % 2 == 0:
            k += 1
        kernel = torch.ones((1, 1, 1, k), dtype=x.dtype) / float(k)
        x4 = x.unsqueeze(0)
        x4 = F.pad(x4, pad=(k // 2, k // 2, 0, 0), mode="reflect")
        return F.conv2d(x4, kernel).squeeze(0).clamp(0.0, 1.0)

    if corruption == "pixelation":
        size = int(pixel_size)
        if size <= 0 or size >= 28:
            return x
        x4 = x.unsqueeze(0)
        low = F.interpolate(x4, size=(size, size), mode="nearest")
        high = F.interpolate(low, size=(28, 28), mode="nearest")
        return high.squeeze(0).clamp(0.0, 1.0)

    raise ValueError(f"Unknown corruption: {corruption}")


class StaticCorruptedMNIST(Dataset):
    """
    MNIST test set with deterministic test-time corruption.

    Returned samples are standard MNIST tensors with no context cue, so this
    dataset is compatible with the clean-MNIST classifier architecture.
    """

    def __init__(
        self,
        root: str,
        *,
        train: bool,
        corruption: str,
        noise_seed: int,
        gaussian_std: float,
        brightness_delta: float,
        motion_kernel: int,
        pixel_size: int,
    ) -> None:
        super().__init__()
        self.base = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.corruption = corruption
        self.noise_seed = int(noise_seed)
        self.gaussian_std = float(gaussian_std)
        self.brightness_delta = float(brightness_delta)
        self.motion_kernel = int(motion_kernel)
        self.pixel_size = int(pixel_size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.base[idx]
        x = apply_corruption(
            x,
            corruption=self.corruption,
            idx=idx,
            noise_seed=self.noise_seed,
            gaussian_std=self.gaussian_std,
            brightness_delta=self.brightness_delta,
            motion_kernel=self.motion_kernel,
            pixel_size=self.pixel_size,
        )
        x = (x - MNIST_MEAN) / MNIST_STD
        return x, int(y)


def make_eval_loaders(
    *,
    cfg: Config,
    batch_size: int,
    num_workers: int,
    corruption_seed: int,
    gaussian_std: float,
    brightness_delta: float,
    motion_kernel: int,
    pixel_size: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}

    for corruption in CORRUPTION_ORDER:
        ds = StaticCorruptedMNIST(
            root=cfg.mnist_root,
            train=False,
            corruption=corruption,
            noise_seed=corruption_seed,
            gaussian_std=gaussian_std,
            brightness_delta=brightness_delta,
            motion_kernel=motion_kernel,
            pixel_size=pixel_size,
        )
        loaders[corruption] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=cfg.pin_memory and str(cfg.device).startswith("cuda"),
            drop_last=False,
        )

    return loaders



def make_variant_config(base_cfg: Config, seed: int, variant: str) -> Config:
    common = dict(
        seed=seed,
        dataset="mnist",
        method="pc",
    )

    if variant == "pc_K0":
        return replace(
            base_cfg,
            **common,
            astro_controller_enabled=False,
            astro_controller_leak_enabled=False,
            astro_controller_state_coupling_enabled=False,
            phago_enabled=False,
            pc_frac_memory_len=0,
        )

    if variant == "pc_memory_K2":
        return replace(
            base_cfg,
            **common,
            astro_controller_enabled=False,
            astro_controller_leak_enabled=False,
            astro_controller_state_coupling_enabled=False,
            phago_enabled=False,
            pc_frac_memory_len=2,
        )

    if variant == "astro_gain_only":
        return replace(
            base_cfg,
            **common,
            astro_controller_enabled=True,
            astro_controller_dynamics="astro_state",
            astro_controller_state_coupling_enabled=False,
            astro_controller_leak_enabled=False,
            phago_enabled=False,
            pc_frac_memory_len=2,
        )

    if variant == "astro_gain_state_coupling":
        return replace(
            base_cfg,
            **common,
            astro_controller_enabled=True,
            astro_controller_dynamics="astro_state",
            astro_controller_state_coupling_enabled=True,
            astro_controller_leak_enabled=False,
            phago_enabled=False,
            pc_frac_memory_len=2,
        )

    if variant == "astro_gain_state_coupling_leakage":
        return replace(
            base_cfg,
            **common,
            astro_controller_enabled=True,
            astro_controller_dynamics="astro_state",
            astro_controller_state_coupling_enabled=True,
            astro_controller_leak_enabled=True,
            phago_enabled=False,
            pc_frac_memory_len=2,
        )

    if variant == "passive_astro_pc":
        return replace(
            base_cfg,
            **common,
            astro_controller_enabled=True,
            astro_controller_dynamics="passive_slow_gain",
        )

    if variant == "full_astro_pc":
        return replace(
            base_cfg,
            **common,
            astro_controller_enabled=True,
            astro_controller_dynamics="astro_state",
            astro_controller_state_coupling_enabled=True,
            astro_controller_leak_enabled=True,
            phago_enabled=True,
            pc_frac_memory_len=2,
        )

    raise ValueError(
        f"Unknown variant: {variant}. Expected one of {list(VARIANT_LABELS)}."
    )


def variant_configs(
    base_cfg: Config,
    seed: int,
    variants: list[str] | tuple[str, ...] = DEFAULT_VARIANTS,
) -> list[tuple[str, str, Config]]:
    return [
        (variant, VARIANT_LABELS[variant], make_variant_config(base_cfg, seed, variant))
        for variant in variants
    ]

def train_or_resume_variant(
    *,
    cfg: Config,
    run_name: str,
    runs_dir: Path,
    save_train_plots: bool,
) -> Path:
    """
    Train a variant, or resume/skip it if the run directory already exists.

    train.main handles checkpoint resume when run_name points to an existing
    interrupted run.
    """
    train_main(
        cfg=cfg,
        run_name=run_name,
        output_root=str(runs_dir),
        save_run_plots=save_train_plots,
    )
    return runs_dir / run_name


def load_model_from_run(cfg: Config, run_dir: Path) -> torch.nn.Module:
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
    return model


def evaluate_variant(
    *,
    cfg: Config,
    run_dir: Path,
    model_name: str,
    model_label: str,
    seed: int,
    eval_loaders: dict[str, DataLoader],
) -> list[dict[str, Any]]:
    device = torch.device(cfg.device)
    model = load_model_from_run(cfg, run_dir)

    rows: list[dict[str, Any]] = []
    for corruption, loader in eval_loaders.items():
        loss, acc = evaluate(model, loader, device)
        rows.append(
            {
                "seed": int(seed),
                "model_name": model_name,
                "model_label": model_label,
                "corruption": corruption,
                "corruption_label": CORRUPTION_LABELS[corruption],
                "test_loss": float(loss),
                "test_acc": float(acc),
            }
        )

    return rows


def make_summary_tables(raw_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(out_dir / "corrupted_mnist_raw_results.csv", index=False)

    # Long-form summary by model and corruption.
    summary_long = (
        raw_df.groupby(
            ["model_name", "model_label", "corruption", "corruption_label"],
            as_index=False,
        )
        .agg(
            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),
            mean_loss=("test_loss", "mean"),
            std_loss=("test_loss", "std"),
            num_seeds=("seed", "nunique"),
        )
        .fillna({"std_acc": 0.0, "std_loss": 0.0})
    )
    summary_long.to_csv(out_dir / "corrupted_mnist_summary_long.csv", index=False)

    # Aggregate corruption robustness per seed before summarizing over seeds.
    corr_df = raw_df[raw_df["corruption"].isin(CORRUPTED_ONLY)]
    mean_corr_per_seed = corr_df.groupby(
        ["seed", "model_name", "model_label"], as_index=False
    ).agg(test_acc=("test_acc", "mean"))
    mean_corr_per_seed["corruption"] = "mean_corrupted"
    mean_corr_per_seed["corruption_label"] = "Mean corr."
    mean_corr_per_seed["test_loss"] = float("nan")

    table_input = pd.concat(
        [
            raw_df[
                [
                    "seed",
                    "model_name",
                    "model_label",
                    "corruption",
                    "corruption_label",
                    "test_acc",
                    "test_loss",
                ]
            ],
            mean_corr_per_seed[
                [
                    "seed",
                    "model_name",
                    "model_label",
                    "corruption",
                    "corruption_label",
                    "test_acc",
                    "test_loss",
                ]
            ],
        ],
        ignore_index=True,
    )

    table_summary = (
        table_input.groupby(
            ["model_name", "model_label", "corruption", "corruption_label"],
            as_index=False,
        )
        .agg(
            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),
            num_seeds=("seed", "nunique"),
        )
        .fillna({"std_acc": 0.0})
    )

    order = {
        "clean": 0,
        "gaussian_noise": 1,
        "motion_blur": 2,
        "brightness": 3,
        "pixelation": 4,
        "mean_corrupted": 5,
    }
    table_summary["corruption_order"] = table_summary["corruption"].map(order)
    table_summary["model_order"] = table_summary["model_name"].map(VARIANT_ORDER)
    table_summary = table_summary.sort_values(["model_order", "corruption_order"])

    # Machine-readable wide table with percentage mean/std columns.
    wide_rows: list[dict[str, Any]] = []
    for (model_name, model_label), sub in table_summary.groupby(
        ["model_name", "model_label"], sort=False
    ):
        row: dict[str, Any] = {
            "model_name": model_name,
            "model_label": model_label,
        }
        for _, rec in sub.iterrows():
            key = str(rec["corruption"])
            row[f"{key}_mean_pct"] = 100.0 * float(rec["mean_acc"])
            row[f"{key}_std_pct"] = 100.0 * float(rec["std_acc"])
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    wide.to_csv(out_dir / "corrupted_mnist_accuracy_table.csv", index=False)

    # Paper-facing LaTeX table in mean ± std format.
    def fmt(mean_pct: float, std_pct: float) -> str:
        return f"{mean_pct:.2f} $\\pm$ {std_pct:.2f}"

    latex_cols = [
        ("clean", "Clean"),
        ("gaussian_noise", "Gaussian"),
        ("motion_blur", "Motion blur"),
        ("brightness", "Brightness"),
        ("pixelation", "Pixelation"),
        ("mean_corrupted", "Mean corr."),
    ]

    lines: list[str] = []
    lines.append(r"\begin{table}[!htb]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Final test accuracy on clean and corrupted MNIST variants over seeds. Values are percentages reported as mean \(\pm\) standard deviation. The mean corrupted accuracy is computed per seed over Gaussian noise, motion blur, brightness, and pixelation, and then averaged over seeds.}"
    )
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & Clean & Gaussian & Motion blur & Brightness & Pixelation & Mean corr. \\"
    )
    lines.append(r"\midrule")

    for _, row in wide.iterrows():
        vals = []
        for key, _ in latex_cols:
            vals.append(
                fmt(float(row[f"{key}_mean_pct"]), float(row[f"{key}_std_pct"]))
            )
        lines.append(f"{row['model_label']} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:corrupted_mnist_accuracy}")
    lines.append(r"\end{table}")

    (out_dir / "corrupted_mnist_accuracy_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static corrupted-MNIST final-checkpoint evaluation over seeds."
    )
    parser.add_argument("--out_dir", default="plots/mnist_static_corrupted_5seed_v1")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds.")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants to train/evaluate.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override Config.epochs if provided."
    )
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--corruption_seed",
        type=int,
        default=12345,
        help="Fixed seed for deterministic test corruptions.",
    )

    # Corruption parameters. Defaults match the multicontext experiment command/defaults.
    parser.add_argument("--gaussian_std", type=float, default=0.275)
    parser.add_argument("--brightness_delta", type=float, default=0.25)
    parser.add_argument("--motion_kernel", type=int, default=11)
    parser.add_argument("--pixel_size", type=int, default=5)

    parser.add_argument(
        "--save_train_plots",
        action="store_true",
        help="Also save training diagnostic plots for each run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_csv_ints(args.seeds)
    variants = parse_csv_strings(args.variants)

    unknown = [v for v in variants if v not in VARIANT_LABELS]
    if unknown:
        raise ValueError(
            f"Unknown variants {unknown}. Valid options: {list(VARIANT_LABELS)}"
        )

    base_cfg = Config()
    updates: dict[str, Any] = {
        "dataset": "mnist",
        "method": "pc",
        "batch_size": args.batch_size,
    }
    if args.epochs is not None:
        updates["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        updates["num_workers"] = int(args.num_workers)

    base_cfg = replace(base_cfg, **updates)
    base_cfg.validate()

    manifest = {
        "description": "Static clean-training / corrupted-test MNIST evaluation over seeds.",
        "seeds": seeds,
        "variants": variants,
        "variant_labels": {variant: VARIANT_LABELS[variant] for variant in variants},
        "corruptions": CORRUPTION_ORDER,
        "corruption_parameters": {
            "gaussian_std": args.gaussian_std,
            "brightness_delta": args.brightness_delta,
            "motion_kernel": args.motion_kernel,
            "pixel_size": args.pixel_size,
            "corruption_seed": args.corruption_seed,
        },
        "base_config": asdict(base_cfg),
        "note": "Models are trained on clean MNIST only. Corruptions are applied at test time.",
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)

    all_rows: list[dict[str, Any]] = []

    eval_loaders = make_eval_loaders(
        cfg=base_cfg,
        batch_size=args.batch_size,
        num_workers=base_cfg.num_workers,
        corruption_seed=args.corruption_seed,
        gaussian_std=args.gaussian_std,
        brightness_delta=args.brightness_delta,
        motion_kernel=args.motion_kernel,
        pixel_size=args.pixel_size,
    )

    for seed in seeds:
        set_seed(seed)
        for model_name, model_label, cfg in variant_configs(base_cfg, seed, variants):
            run_name = f"{model_name}__seed{seed}"
            print("\n" + "=" * 80)
            print(f"Training/resuming variant: {model_label} | seed={seed}")
            print("=" * 80)

            run_dir = train_or_resume_variant(
                cfg=cfg,
                run_name=run_name,
                runs_dir=runs_dir,
                save_train_plots=args.save_train_plots,
            )

            rows = evaluate_variant(
                cfg=cfg,
                run_dir=run_dir,
                model_name=model_name,
                model_label=model_label,
                seed=seed,
                eval_loaders=eval_loaders,
            )
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(
                out_dir / "corrupted_mnist_raw_results.csv", index=False
            )

    raw_df = pd.DataFrame(all_rows)
    make_summary_tables(raw_df, out_dir)

    print("\nSaved outputs:")
    print(f"  {out_dir / 'corrupted_mnist_raw_results.csv'}")
    print(f"  {out_dir / 'corrupted_mnist_summary_long.csv'}")
    print(f"  {out_dir / 'corrupted_mnist_accuracy_table.csv'}")
    print(f"  {out_dir / 'corrupted_mnist_accuracy_table.tex'}")


if __name__ == "__main__":
    main()
