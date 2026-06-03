from __future__ import annotations

import csv
import math
import sys
from numbers import Number
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from controller_configs import PCConfig
from pc import PCTrainState, pc_train_batch


_USE_COLOR = sys.stdout.isatty()


LAYER_DIAGNOSTIC_METRICS = {
    "astro_controller_state_values": "astro_state",
    "astro_controller_effective_gains": "effective_gain",
    "astro_controller_received_leak_layers": "received_leakage",
    "astro_controller_phago_permissive_layers": "phago_permissive_signal",
    "pc_G_rms_layers": "instantaneous_drive_rms",
    "pc_M_rms_layers": "memory_drive_rms",
    "dW_rms_layers": "unscaled_update_drive_rms",
    "weight_rms_layers": "weight_rms",
    "weight_fro_layers": "weight_fro",
    "applied_weight_delta_rms_layers": "applied_weight_delta_rms",
    "applied_weight_delta_fro_layers": "applied_weight_delta_fro",
    "relative_weight_delta_fro_layers": "relative_weight_delta_fro",
    "phago_num_newly_pruned_layers": "hard_pruned_events",
    "phago_num_candidates_layers": "pruning_candidates",
    "phago_num_selected_layers": "pruning_selected",
    "phago_alive_fraction_layers": "alive_fraction",
    "phago_hard_pruned_fraction_layers": "hard_pruned_fraction",
    "phago_mean_pruning_pressure_layers": "mean_pruning_pressure",
    "phago_max_pruning_pressure_layers": "max_pruning_pressure",
    "phago_mean_permissive_signal_layers": "permissive_signal",
}


def _is_number_like(value: object) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def _mean_numeric_list(values: object) -> float | None:
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        return None
    xs = []
    for value in values:
        if _is_number_like(value):
            xs.append(float(value))
    if not xs:
        return None
    return sum(xs) / len(xs)


def _sum_numeric_list(values: object) -> float | None:
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        return None
    xs = []
    for value in values:
        if _is_number_like(value):
            xs.append(float(value))
    if not xs:
        return None
    return sum(xs)


def _compact_batch_metrics_for_memory(batch_metrics: dict) -> dict:
    """Keep scalar diagnostics and compact summaries in memory."""
    compact: dict = {}
    for key, value in batch_metrics.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            compact[key] = value
            continue

        if isinstance(value, (list, tuple)):
            mean_value = _mean_numeric_list(value)
            if mean_value is not None:
                compact[f"{key}_mean"] = mean_value

            if key in {
                "phago_num_newly_pruned_layers",
                "phago_num_candidates_layers",
                "phago_num_selected_layers",
            }:
                sum_value = _sum_numeric_list(value)
                if sum_value is not None:
                    compact[f"{key}_sum"] = sum_value

    return compact


def _append_layer_diagnostics_csv(
    csv_path: str | Path | None,
    *,
    epoch: int,
    step: int,
    batch_metrics: dict,
) -> None:
    """Append layer-wise batch diagnostics to CSV."""
    if csv_path is None:
        return

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    fieldnames = ["epoch", "step", "source_key", "metric", "layer", "value"]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for source_key, metric_name in LAYER_DIAGNOSTIC_METRICS.items():
            values = batch_metrics.get(source_key)
            if not isinstance(values, (list, tuple)):
                continue

            for layer_idx, value in enumerate(values, start=1):
                if not _is_number_like(value):
                    continue
                writer.writerow(
                    {
                        "epoch": int(epoch),
                        "step": int(step),
                        "source_key": source_key,
                        "metric": metric_name,
                        "layer": int(layer_idx),
                        "value": float(value),
                    }
                )


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"


def _color(text: object, *styles: str) -> str:
    text = str(text)
    if not _USE_COLOR:
        return text
    return "".join(styles) + text + _C.RESET


def _fmt_num(x: Number, precision: int = 4) -> str:
    x = float(x)

    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"

    if x != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:.{precision}e}"

    return f"{x:.{precision}f}"


def _fmt_list(xs, precision: int = 4) -> str:
    return "[" + ", ".join(_fmt_num(x, precision) for x in xs) + "]"


def _print_pc_step_summary(epoch: int, step: int, batch_metrics: dict) -> None:
    if step != 1:
        return

    msg = (
        f"{_color(f'[PC ep{epoch}]', _C.BOLD, _C.CYAN)} "
        f"Loss | before={batch_metrics['ce_before']:.4f} | "
        f"after={batch_metrics['ce_after']:.4f} | "
        f"delta={batch_metrics['delta_ce']:+.4f}"
    )

    if "pc_G_rms_layers" in batch_metrics:
        msg += (
            f"\n  Grad | "
            f"G_rms={_fmt_list(batch_metrics['pc_G_rms_layers'], 4)} | "
            f"M_rms={_fmt_list(batch_metrics['pc_M_rms_layers'], 4)} | "
            f"M/G={_fmt_list(batch_metrics['pc_M_over_G_layers'], 4)} | "
            f"cos(G,M)={_fmt_list(batch_metrics['pc_cos_G_M_layers'], 4)}"
        )

    if "astro_controller_gain_mean" in batch_metrics:
        msg += (
            f"\n  Astro | "
            f"gain_mean={_fmt_num(batch_metrics['astro_controller_gain_mean'], 4)} | "
            f"state_mean={_fmt_num(batch_metrics.get('astro_controller_state_mean', 0.0), 4)} | "
            f"state_abs={_fmt_num(batch_metrics.get('astro_controller_state_abs_mean', 0.0), 4)} | "
            f"couple_abs={_fmt_num(batch_metrics.get('astro_controller_state_coupling_abs_mean', 0.0), 4)} | "
            f"leak_mean={_fmt_num(batch_metrics.get('astro_controller_leak_increment_mean', 0.0), 4)} | "
            f"leak_max={_fmt_num(batch_metrics.get('astro_controller_leak_increment_max', 0.0), 4)} | "
            f"sources={batch_metrics.get('astro_controller_num_active_sources', 0)} | "
            f"receivers={batch_metrics.get('astro_controller_num_receiving_layers', 0)} | "
            f"edges={batch_metrics.get('astro_controller_num_active_edges', 0)}"
        )

    if "astro_controller_leak_baselines" in batch_metrics:
        msg += (
            f" | base={_fmt_list(batch_metrics['astro_controller_leak_baselines'], 4)}"
        )

    if "phago_alive_fraction_layers" in batch_metrics:
        msg += (
            f"\n  Phagocytosis | "
            f"alive={_fmt_list(batch_metrics['phago_alive_fraction_layers'], 4)} | "
            f"hard={_fmt_list(batch_metrics['phago_hard_pruned_fraction_layers'], 4)} | "
            f"useful={_fmt_list(batch_metrics['phago_mean_usefulness_layers'], 4)} | "
            f"weak={_fmt_list(batch_metrics['phago_mean_weakness_layers'], 4)} | "
            f"gate={_fmt_list(batch_metrics.get('phago_mean_permissive_signal_layers', batch_metrics['phago_mean_leak_signal_layers']), 4)} | "
            f"press={_fmt_list(batch_metrics['phago_mean_pruning_pressure_layers'], 4)}"
        )

    print(msg, flush=True)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(1, total)
    avg_acc = correct / max(1, total)
    return avg_loss, avg_acc


def train_one_epoch_bp(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 200,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="mean")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_total += y.size(0)

        if step % log_every == 0:
            avg_loss = running_loss / max(1, running_total)
            avg_acc = running_correct / max(1, running_total)
            print(
                f"[BP ep{epoch} step{step}] "
                f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f}",
                flush=True,
            )

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    return avg_loss, avg_acc


def train_one_epoch_pc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    cfg: PCConfig,
    state: PCTrainState,
    *,
    metrics_store: list[dict] | None = None,
    diagnostics_csv_path: str | Path | None = None,
    keep_full_batch_metrics_in_memory: bool = False,
) -> tuple[float, float, PCTrainState]:
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        loss_before, batch_correct, batch_metrics, state = pc_train_batch(
            model=model,
            x=x,
            y=y,
            cfg=cfg,
            state=state,
        )

        running_loss += loss_before * y.size(0)
        running_correct += batch_correct
        running_total += y.size(0)

        _append_layer_diagnostics_csv(
            diagnostics_csv_path,
            epoch=epoch,
            step=step,
            batch_metrics=batch_metrics,
        )

        if metrics_store is not None:
            if keep_full_batch_metrics_in_memory:
                metrics_record = {
                    "epoch": epoch,
                    "step": step,
                    **batch_metrics,
                }
            else:
                metrics_record = {
                    "epoch": epoch,
                    "step": step,
                    **_compact_batch_metrics_for_memory(batch_metrics),
                }
            metrics_store.append(metrics_record)

        _print_pc_step_summary(epoch, step, batch_metrics)

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    return avg_loss, avg_acc, state
