from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any


def save_metrics_csv(metrics: list[dict[str, Any]], out_path: str | Path) -> None:
    """Save metric dictionaries to CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not metrics:
        raise ValueError("No metrics to save.")

    fieldnames: list[str] = []
    seen: set[str] = set()
    for rec in metrics:
        for key in rec.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in metrics:
            writer.writerow({key: rec.get(key, None) for key in fieldnames})


def save_metrics_pickle(obj: Any, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def mean_list_value(x: Any) -> float | None:
    if isinstance(x, list) and len(x) > 0:
        return float(sum(float(v) for v in x) / len(x))
    return None


def last_record_with_key(records: list[dict], key: str) -> dict | None:
    for rec in reversed(records):
        if key in rec:
            return rec
    return None


def last_scalar_metric(records: list[dict], key: str) -> float | None:
    rec = last_record_with_key(records, key)
    if rec is None:
        return None
    value = rec.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def last_list_mean_metric(records: list[dict], key: str) -> float | None:
    rec = last_record_with_key(records, key)
    if rec is None:
        return None
    return mean_list_value(rec.get(key))


def compute_epoch_mean_dw(method_metrics: list[dict], epoch: int) -> float | None:
    epoch_recs = [rec for rec in method_metrics if int(rec.get("epoch", -1)) == epoch]
    if not epoch_recs:
        return None

    vals: list[float] = []
    for rec in epoch_recs:
        layers = rec.get("dW_rms_layers", None)
        if isinstance(layers, list):
            vals.extend(float(x) for x in layers)
            continue

        mean_value = rec.get("dW_rms_layers_mean", None)
        if isinstance(mean_value, (int, float)):
            vals.append(float(mean_value))

    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_final_method_metrics(
    method_metrics: list[dict],
) -> dict[str, float | None]:
    """Summarize final method diagnostics."""
    scalar_keys = {
        "final_astro_gain_mean": "astro_controller_gain_mean",
        "final_astro_base_gain_mean": "astro_controller_base_gain_mean",
        "final_astro_state_mean": "astro_controller_state_mean",
        "final_astro_state_abs_mean": "astro_controller_state_abs_mean",
        "final_astro_state_max_abs": "astro_controller_state_max_abs",
        "final_astro_mismatch_signal_mean": "astro_controller_mismatch_signal_mean",
        "final_astro_plasticity_signal_mean": "astro_controller_plasticity_signal_mean",
        "final_astro_state_input_mean": "astro_controller_state_input_mean",
        "final_state_coupling_abs_mean": "astro_controller_state_coupling_abs_mean",
        "final_state_coupling_max_abs": "astro_controller_state_coupling_max_abs",
        "final_leak_increment_mean": "astro_controller_leak_increment_mean",
        "final_leak_increment_max": "astro_controller_leak_increment_max",
        "final_num_active_leak_sources": "astro_controller_num_active_sources",
        "final_num_receiving_layers": "astro_controller_num_receiving_layers",
    }
    list_keys = {
        "final_mean_alive_fraction": "phago_alive_fraction_layers",
        "final_mean_hard_pruned_fraction": "phago_hard_pruned_fraction_layers",
        "final_mean_phago_weakness": "phago_mean_weakness_layers",
        "final_mean_pruning_pressure": "phago_mean_pruning_pressure_layers",
    }

    out: dict[str, float | None] = {}
    for out_key, metric_key in scalar_keys.items():
        out[out_key] = last_scalar_metric(method_metrics, metric_key)
    for out_key, metric_key in list_keys.items():
        out[out_key] = last_list_mean_metric(method_metrics, metric_key)
    return out
