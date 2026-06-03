"""
Controller-timescale ablation for the Full Astro-PC model.

This script reproduces the focused timescale experiment reported in the paper.
It trains Full Astro-PC on MNIST for several controller update periods,

    T_a in {50, 100, 250}

and writes both a summary CSV and a ready-to-use test-accuracy overlay plot.

Example:
    python timescale_ablation.py \
        --out_dir results/mnist_timescale_ablation \
        --seeds 0 \
        --epochs 50 \
        --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

from config import Config
from train import main as train_main


def parse_csv_ints(text: str) -> list[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_cfg_checked(base: Config, **updates: Any) -> Config:
    valid = {f.name for f in fields(Config)}
    unknown = sorted(set(updates) - valid)
    if unknown:
        raise ValueError(f"Unknown Config fields: {unknown}")
    cfg = replace(base, **updates)
    cfg.validate()
    return cfg


def make_full_astro_cfg(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    controller_period: int,
) -> Config:
    """
    Build the Full Astro-PC configuration for one value of T_a.

    The script uses the current Config defaults for all reported Astro-PC
    hyperparameters, and only overrides the dataset/method flags and the
    controller update period. This keeps the ablation synchronized with the
    reported configuration in config.py.
    """
    base = Config()
    return make_cfg_checked(
        base,
        dataset="mnist",
        method="pc",
        seed=int(seed),
        epochs=int(epochs),
        batch_size=int(batch_size),
        astro_controller_enabled=True,
        astro_controller_dynamics="astro_state",
        astro_controller_leak_enabled=True,
        phago_enabled=True,
        astro_controller_update_period_steps=int(controller_period),
    )


def read_run_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")
    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError(f"Empty metrics.csv: {metrics_path}")
    return df


def summarize_run(
    *,
    run_dir: Path,
    seed: int,
    controller_period: int,
) -> dict[str, Any]:
    df = read_run_metrics(run_dir)

    test_acc = pd.to_numeric(df["test_acc"], errors="coerce")
    test_loss = pd.to_numeric(df["test_loss"], errors="coerce")
    train_acc = pd.to_numeric(df["train_acc"], errors="coerce")
    train_loss = pd.to_numeric(df["train_loss"], errors="coerce")

    final = df.iloc[-1]

    return {
        "seed": int(seed),
        "astro_controller_update_period_steps": int(controller_period),
        "run_dir": str(run_dir),
        "final_epoch": int(final["epoch"]),
        "final_train_acc": float(train_acc.iloc[-1]),
        "final_test_acc": float(test_acc.iloc[-1]),
        "best_test_acc": float(test_acc.max()),
        "final_train_loss": float(train_loss.iloc[-1]),
        "final_test_loss": float(test_loss.iloc[-1]),
        "best_test_loss": float(test_loss.min()),
        "failed": int(
            pd.to_numeric(df.get("failed", pd.Series([0])), errors="coerce")
            .fillna(0)
            .max()
        ),
    }


def collect_epoch_curves(
    *,
    runs_root: Path,
    periods: list[int],
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for period in periods:
        for seed in seeds:
            run_name = f"full_astro_pc_Ta{period}__seed{seed}"
            run_dir = runs_root / run_name
            df = read_run_metrics(run_dir).copy()
            df["seed"] = int(seed)
            df["astro_controller_update_period_steps"] = int(period)
            df["run_name"] = run_name
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def plot_timescale_accuracy(
    curves_df: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    if curves_df.empty:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    df = curves_df.copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")
    df = df.dropna(subset=["epoch", "test_acc"])

    summary = (
        df.groupby(["astro_controller_update_period_steps", "epoch"], as_index=False)
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            num_seeds=("seed", "nunique"),
        )
        .fillna({"std_test_acc": 0.0})
        .sort_values(["astro_controller_update_period_steps", "epoch"])
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    for period, sub in summary.groupby("astro_controller_update_period_steps"):
        x = sub["epoch"].to_numpy(dtype=float)
        mean = 100.0 * sub["mean_test_acc"].to_numpy(dtype=float)
        std = 100.0 * sub["std_test_acc"].to_numpy(dtype=float)

        ax.plot(
            x,
            mean,
            linewidth=2.0,
            label=rf"$T_a={int(period)}$",
        )

        if int(sub["num_seeds"].max()) > 1:
            ax.fill_between(x, mean - std, mean + std, alpha=0.18)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Controller-timescale ablation")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    # Keep the historical filename used by existing analysis scripts.
    png = out_dir / "timescale_training_accuracy_overlay.png"
    pdf = out_dir / "timescale_training_accuracy_overlay.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return [png, pdf]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Full Astro-PC controller-timescale ablation."
    )
    parser.add_argument(
        "--out_dir",
        default="results/mnist_timescale_ablation",
        help="Output directory for the timescale-ablation study.",
    )
    parser.add_argument(
        "--periods",
        default="50,100,250",
        help="Comma-separated controller update periods T_a.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--save_run_plots",
        action="store_true",
        help="Also generate per-run diagnostics. Disabled by default to save time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    runs_root = out_dir / "runs"
    comparison_dir = out_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    periods = parse_csv_ints(args.periods)
    seeds = parse_csv_ints(args.seeds)

    manifest = {
        "description": "Full Astro-PC controller-timescale ablation.",
        "controller_update_periods": periods,
        "seeds": seeds,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "base_config_note": (
            "All Full Astro-PC hyperparameters are taken from config.py except "
            "astro_controller_update_period_steps, which is swept."
        ),
    }

    with (out_dir / "timescale_ablation_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)

    summary_rows: list[dict[str, Any]] = []

    for period in periods:
        for seed in seeds:
            set_seed(seed)
            cfg = make_full_astro_cfg(
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                controller_period=period,
            )
            run_name = f"full_astro_pc_Ta{period}__seed{seed}"

            print("\n" + "=" * 88)
            print(f"Running Full Astro-PC with T_a={period}, seed={seed}")
            print(f"Run name: {run_name}")
            print("=" * 88 + "\n")

            train_main(
                cfg=cfg,
                run_name=run_name,
                output_root=str(runs_root),
                save_run_plots=bool(args.save_run_plots),
            )

            run_dir = runs_root / run_name
            summary_rows.append(
                summarize_run(
                    run_dir=run_dir,
                    seed=seed,
                    controller_period=period,
                )
            )

            # Save incrementally so interrupted studies leave usable summaries.
            pd.DataFrame(summary_rows).to_csv(
                out_dir / "timescale_summary.csv", index=False
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "timescale_summary.csv", index=False)

    curves_df = collect_epoch_curves(
        runs_root=runs_root,
        periods=periods,
        seeds=seeds,
    )
    curves_df.to_csv(out_dir / "timescale_epoch_curves.csv", index=False)

    plot_paths = plot_timescale_accuracy(curves_df, comparison_dir)

    print("\nSaved outputs:")
    print(f"  {out_dir / 'timescale_summary.csv'}")
    print(f"  {out_dir / 'timescale_epoch_curves.csv'}")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
