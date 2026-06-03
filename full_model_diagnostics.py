from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAYER_METRICS = {
    "astro_controller_state_values": "astro_state",
    "astro_controller_effective_gains": "effective_gain",
    "astro_controller_received_leak_layers": "received_leakage",
    "astro_controller_phago_permissive_layers": "phago_permissive_signal",
    "pc_G_rms_layers": "instantaneous_drive_rms",
    "pc_M_rms_layers": "memory_drive_rms",
    "dW_rms_layers": "unscaled_update_drive_rms",
    "weight_rms_layers": "weight_rms",
    "applied_weight_delta_rms_layers": "applied_weight_delta_rms",
    "phago_num_newly_pruned_layers": "hard_pruned_events",
    "phago_num_candidates_layers": "pruning_candidates",
    "phago_num_selected_layers": "pruning_selected",
    "phago_alive_fraction_layers": "alive_fraction",
    "phago_hard_pruned_fraction_layers": "hard_pruned_fraction",
    "phago_mean_pruning_pressure_layers": "mean_pruning_pressure",
    "phago_max_pruning_pressure_layers": "max_pruning_pressure",
    "phago_mean_permissive_signal_layers": "permissive_signal",
}


STREAM_CANDIDATES = (
    "diagnostics/layer_internal_diagnostics_stream.csv",
    "layer_internal_diagnostics_stream.csv",
)


def _load_method_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    path = run_dir / "method_metrics.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Could not find method metrics at {path}")

    with path.open("rb") as f:
        records = pickle.load(f)

    if not isinstance(records, list):
        raise TypeError(
            "method_metrics.pkl must contain a list of metric dictionaries."
        )

    return records


def _find_stream_csv(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    for rel in STREAM_CANDIDATES:
        candidate = run_dir / rel
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _epoch_axis(df: pd.DataFrame) -> pd.Series:
    """Convert epoch and step columns to a continuous epoch axis."""
    if df.empty or "epoch" not in df.columns or "step" not in df.columns:
        return pd.Series([], dtype=float)

    epoch = pd.to_numeric(df["epoch"], errors="coerce").fillna(0).astype(int)
    step = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(int)
    max_step = step.groupby(epoch).transform("max").clip(lower=1)

    return (epoch.astype(float) - 1.0) + step.astype(float) / max_step.astype(float)


def _add_batch_index_and_axis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce").fillna(0).astype(int)
    out["step"] = pd.to_numeric(out["step"], errors="coerce").fillna(0).astype(int)
    out["layer"] = pd.to_numeric(out["layer"], errors="coerce").fillna(0).astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[pd.notna(out["value"])].copy()

    batch_keys = (
        out[["epoch", "step"]]
        .drop_duplicates()
        .sort_values(["epoch", "step"])
        .reset_index(drop=True)
    )
    batch_keys["batch_index"] = range(len(batch_keys))

    out = out.merge(batch_keys, on=["epoch", "step"], how="left")
    out["epoch_axis"] = _epoch_axis(out)
    return out.sort_values(["batch_index", "metric", "layer"]).reset_index(drop=True)


def flatten_layer_metrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert in-memory layer metrics to a long table."""
    rows: list[dict[str, Any]] = []

    for rec in records:
        epoch = int(rec.get("epoch", 0))
        step = int(rec.get("step", 0))

        for source_key, metric_name in LAYER_METRICS.items():
            values = rec.get(source_key)
            if not isinstance(values, (list, tuple)):
                continue

            for layer_idx, value in enumerate(values, start=1):
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(value_f):
                    continue

                rows.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "source_key": source_key,
                        "metric": metric_name,
                        "layer": layer_idx,
                        "value": value_f,
                    }
                )

    return _add_batch_index_and_axis(pd.DataFrame(rows))


def load_layer_diagnostics(run_dir: str | Path) -> tuple[pd.DataFrame, str]:
    """Load layer diagnostics from CSV or method_metrics.pkl."""
    stream_path = _find_stream_csv(run_dir)
    if stream_path is not None:
        df = pd.read_csv(stream_path)
        required = {"epoch", "step", "metric", "layer", "value"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Stream diagnostics CSV is missing columns: {sorted(missing)}"
            )
        return _add_batch_index_and_axis(df), f"stream_csv:{stream_path}"

    records = _load_method_metrics(run_dir)
    return flatten_layer_metrics(records), "method_metrics.pkl"


def _add_effective_gain_deviation(layer_df: pd.DataFrame) -> pd.DataFrame:
    r"""Add the effective-gain deviation diagnostic."""
    if layer_df.empty or "metric" not in layer_df.columns:
        return layer_df

    gain = layer_df[layer_df["metric"] == "effective_gain"].copy()
    if gain.empty:
        return layer_df

    gain["metric"] = "effective_gain_deviation"
    gain["value"] = pd.to_numeric(gain["value"], errors="coerce") - 1.0
    if "source_key" in gain.columns:
        gain["source_key"] = gain["source_key"].astype(str) + "_minus_one"

    return pd.concat([layer_df, gain], ignore_index=True)


def _safe_log_values(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values.where(values > 0.0, np.nan)


def _smooth_by_batch(y: pd.Series, window: int) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    if window <= 1 or len(y) < 3:
        return y
    window = min(int(window), len(y))
    return y.rolling(window=window, min_periods=1, center=True).mean()


def _plot_metric_to_file(
    layer_df: pd.DataFrame,
    metric: str,
    *,
    out_dir: Path,
    filename_stem: str,
    ylabel: str,
    title: str,
    yscale: str | None = None,
    smoothing_window: int = 25,
) -> list[Path]:
    """Plot one layer-wise metric."""
    sub = layer_df[layer_df["metric"] == metric].copy()
    if sub.empty:
        return []

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for layer, layer_metric_df in sub.groupby("layer"):
        layer_metric_df = layer_metric_df.sort_values(["batch_index"])
        x = pd.to_numeric(layer_metric_df["epoch_axis"], errors="coerce")
        y = pd.to_numeric(layer_metric_df["value"], errors="coerce")
        y = _smooth_by_batch(y, smoothing_window)

        if yscale == "log":
            y = _safe_log_values(y)

        ax.plot(x, y, linewidth=1.8, label=f"Layer {layer}")

    ax.set_xlabel("Training epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if yscale is not None:
        ax.set_yscale(yscale)

    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    png = out_dir / f"{filename_stem}.png"
    pdf = out_dir / f"{filename_stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_separate_internal_diagnostics(
    layer_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    smoothing_window: int = 25,
) -> list[Path]:
    """Create separate layer-wise diagnostic figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "astro_state",
            "astro_state_trajectories",
            r"Astrocyte-like state $a_l^{(s)}$",
            "Astrocyte-like state trajectories",
            None,
        ),
        (
            "effective_gain_deviation",
            "effective_gain_trajectories",
            r"Effective gain deviation $\tilde{g}_l^{(t)} - 1$",
            r"Effective plasticity-gain deviation trajectories",
            None,
        ),
        (
            "memory_drive_rms",
            "memory_drive_rms_trajectories",
            r"RMS memory-filtered drive",
            r"Memory-filtered drive norm $\|H_l^{(t)}\|$",
            "log",
        ),
        (
            "received_leakage",
            "received_leakage_trajectories",
            r"Received leakage $\Delta g_l^{\mathrm{leak},(t)}$",
            "Received gain leakage",
            None,
        ),
        (
            "weight_rms",
            "weight_rms_trajectories",
            "Weight RMS",
            "Weight magnitude over training",
            None,
        ),
        (
            "applied_weight_delta_rms",
            "applied_weight_delta_rms_trajectories",
            r"RMS actual weight change",
            "Actual weight change per batch",
            "log",
        ),
        (
            "unscaled_update_drive_rms",
            "unscaled_update_drive_rms_trajectories",
            "RMS unscaled update drive",
            "Unscaled update drive",
            "log",
        ),
        (
            "alive_fraction",
            "alive_fraction_trajectories",
            "Alive synapse fraction",
            "Alive synapse fraction",
            None,
        ),
        (
            "mean_pruning_pressure",
            "mean_pruning_pressure_trajectories",
            "Mean pruning pressure",
            "Mean pruning pressure",
            "log",
        ),
        (
            "permissive_signal",
            "permissive_signal_trajectories",
            "Permissive signal",
            "Phagocytosis-permissive signal",
            None,
        ),
    ]

    paths: list[Path] = []
    for metric, filename_stem, ylabel, title, yscale in specs:
        paths.extend(
            _plot_metric_to_file(
                layer_df,
                metric,
                out_dir=out_dir,
                filename_stem=filename_stem,
                ylabel=ylabel,
                title=title,
                yscale=yscale,
                smoothing_window=smoothing_window,
            )
        )

    return paths


def plot_metric_mean_across_layers(
    layer_df: pd.DataFrame,
    out_dir: Path,
    *,
    metric: str,
    filename: str,
    title: str,
    ylabel: str,
    yscale: str | None = None,
    smoothing_window: int = 25,
) -> list[Path]:
    """Plot the mean trajectory across layers."""
    sub = layer_df[layer_df["metric"] == metric].copy()
    if sub.empty:
        return []

    summary = (
        sub.groupby("batch_index", as_index=False)
        .agg(
            epoch_axis=("epoch_axis", "first"),
            mean_value=("value", "mean"),
            min_value=("value", "min"),
            max_value=("value", "max"),
        )
        .sort_values("batch_index")
    )

    if summary.empty:
        return []

    x = summary["epoch_axis"].to_numpy(dtype=float)
    mean = _smooth_by_batch(summary["mean_value"], smoothing_window).to_numpy(
        dtype=float
    )
    lo = _smooth_by_batch(summary["min_value"], smoothing_window).to_numpy(dtype=float)
    hi = _smooth_by_batch(summary["max_value"], smoothing_window).to_numpy(dtype=float)

    if yscale == "log":
        mean = np.where(mean > 0.0, mean, np.nan)
        lo = np.where(lo > 0.0, lo, np.nan)
        hi = np.where(hi > 0.0, hi, np.nan)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(x, mean, linewidth=2.2, label="Mean across layers")
    ax.fill_between(x, lo, hi, alpha=0.20, label="Layer range")

    ax.set_xlabel("Training epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    if yscale is not None and np.isfinite(mean).any():
        ax.set_yscale(yscale)

    fig.tight_layout()

    png = out_dir / f"{filename}.png"
    pdf = out_dir / f"{filename}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_mean_across_layer_summaries(
    layer_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    smoothing_window: int = 25,
) -> list[Path]:
    """Create mean-across-layers summary plots."""
    out_dir = Path(out_dir)
    paths: list[Path] = []

    specs = [
        (
            "astro_state",
            "mean_astro_state_across_layers",
            "Mean astrocyte-like state across layers",
            r"Mean $a_l^{(s)}$",
            None,
        ),
        (
            "effective_gain_deviation",
            "mean_effective_gain_across_layers",
            r"Mean effective-gain deviation across layers",
            r"Mean $(\tilde{g}_l^{(t)} - 1)$",
            None,
        ),
        (
            "memory_drive_rms",
            "mean_memory_drive_rms_across_layers",
            r"Mean memory-filtered drive across layers",
            r"Mean RMS $\|H_l^{(t)}\|$",
            "log",
        ),
        (
            "received_leakage",
            "mean_received_leakage_across_layers",
            r"Mean received leakage across layers",
            r"Mean $\Delta g_l^{\mathrm{leak},(t)}$",
            None,
        ),
        (
            "applied_weight_delta_rms",
            "mean_applied_weight_delta_rms_across_layers",
            r"Mean applied weight-update RMS across layers",
            r"Mean RMS of applied $\Delta W_l$",
            "log",
        ),
        (
            "unscaled_update_drive_rms",
            "mean_update_drive_rms_across_layers",
            r"Mean update-drive RMS across layers",
            r"Mean RMS update drive",
            "log",
        ),
    ]

    for metric, filename, title, ylabel, yscale in specs:
        paths.extend(
            plot_metric_mean_across_layers(
                layer_df,
                out_dir,
                metric=metric,
                filename=filename,
                title=title,
                ylabel=ylabel,
                yscale=yscale,
                smoothing_window=smoothing_window,
            )
        )

    return paths


def plot_pruning_events_per_epoch(
    layer_df: pd.DataFrame, out_dir: str | Path
) -> list[Path]:
    """Plot hard-pruning events per epoch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = layer_df[layer_df["metric"] == "hard_pruned_events"].copy()
    ylabel = "Hard-pruned synapses per epoch"
    title = "Hard-pruning events per epoch"

    if sub.empty:
        cumulative = layer_df[layer_df["metric"] == "alive_fraction"].copy()
        if cumulative.empty:
            return []

        epoch_layer = (
            cumulative.sort_values(["epoch", "step"])
            .groupby(["epoch", "layer"], as_index=False)
            .tail(1)
            .sort_values(["layer", "epoch"])
        )
        epoch_layer["value"] = (
            -epoch_layer.groupby("layer")["value"].diff().fillna(0.0)
        ).clip(lower=0.0)
        sub = epoch_layer
        ylabel = "Decrease in alive fraction per epoch"
        title = "Pruning-rate proxy per epoch"

    per_epoch = (
        sub.groupby(["epoch", "layer"], as_index=False)["value"]
        .sum()
        .sort_values(["layer", "epoch"])
    )

    csv_path = out_dir / "hard_pruning_events_per_epoch.csv"
    per_epoch.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for layer, layer_metric_df in per_epoch.groupby("layer"):
        ax.plot(
            layer_metric_df["epoch"],
            layer_metric_df["value"],
            marker="o",
            linewidth=1.8,
            label=f"Layer {layer}",
        )

    total = per_epoch.groupby("epoch", as_index=False)["value"].sum()
    ax.plot(
        total["epoch"],
        total["value"],
        marker="s",
        linewidth=2.2,
        linestyle="--",
        label="Total",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    png = out_dir / "hard_pruning_events_per_epoch.png"
    pdf = out_dir / "hard_pruning_events_per_epoch.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [csv_path, png, pdf]


def save_layer_diagnostics_tables(
    layer_df: pd.DataFrame, out_dir: str | Path
) -> list[Path]:
    """Save long-format and epoch-summary diagnostic tables."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    long_path = out_dir / "layer_internal_diagnostics_long.csv"
    layer_df.to_csv(long_path, index=False)
    paths.append(long_path)

    if not layer_df.empty:
        continuous = layer_df[layer_df["metric"] != "hard_pruned_events"]
        events = layer_df[layer_df["metric"] == "hard_pruned_events"]

        pieces: list[pd.DataFrame] = []
        if not continuous.empty:
            pieces.append(
                continuous.groupby(["epoch", "layer", "metric"], as_index=False).agg(
                    value=("value", "mean")
                )
            )
        if not events.empty:
            pieces.append(
                events.groupby(["epoch", "layer", "metric"], as_index=False).agg(
                    value=("value", "sum")
                )
            )

        if pieces:
            summary = pd.concat(pieces, ignore_index=True)
            summary_path = out_dir / "layer_internal_diagnostics_epoch_summary.csv"
            summary.to_csv(summary_path, index=False)
            paths.append(summary_path)

    return paths


def generate_full_model_diagnostics(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    verbose: bool = True,
    smoothing_window: int = 25,
) -> list[Path]:
    """Generate internal diagnostics from a training run directory."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_df, source = load_layer_diagnostics(run_dir)
    if layer_df.empty:
        raise ValueError("No layer-wise diagnostics were found.")

    layer_df = _add_effective_gain_deviation(layer_df)

    generated: list[Path] = []
    generated.extend(save_layer_diagnostics_tables(layer_df, out_dir))
    generated.extend(
        plot_separate_internal_diagnostics(
            layer_df,
            out_dir,
            smoothing_window=smoothing_window,
        )
    )
    generated.extend(
        plot_mean_across_layer_summaries(
            layer_df,
            out_dir,
            smoothing_window=smoothing_window,
        )
    )

    if (layer_df["metric"].isin(["hard_pruned_events", "alive_fraction"])).any():
        generated.extend(plot_pruning_events_per_epoch(layer_df, out_dir))

    if verbose:
        print(f"Generated diagnostics from {source}:")
        for path in generated:
            print(f"  {path}")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Full Astro-PC internal diagnostic plots."
    )
    parser.add_argument(
        "run_dir",
        help="Run directory containing diagnostics CSV or method_metrics.pkl.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to run_dir/diagnostics.",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=25,
        help="Rolling window for visualization only.",
    )
    args = parser.parse_args()

    generate_full_model_diagnostics(
        args.run_dir,
        args.out_dir,
        smoothing_window=args.smoothing_window,
    )


if __name__ == "__main__":
    main()
