"""
Multi-context MNIST corruption-switching experiment with resume support.

This standalone script uses the existing predictive-coding training code, but
creates a sequential context-switching protocol with multiple input regimes.

Default corruption sequence:
    clean -> gaussian_noise -> motion_blur -> pixelation -> brightness

Labels stay unchanged across contexts. Only the input distribution changes.
Weights, PC state, fractional-memory state, astro-controller state, leakage
baselines, and phagocytosis masks are NOT reset between blocks.

The PC baseline uses K=0 by default so it is plain predictive coding without
fractional memory. The ablation variants add memory, active astrocyte-like gain
control, state coupling, gain leakage, and pruning one step at a time. The
passive Astro-PC variant remains available as an auxiliary null.

Resume behavior:
    Each variant/seed writes:
        <out_dir>/<model>__seed<seed>/checkpoint_last.pt
        <out_dir>/<model>__seed<seed>/status.json
    Completed runs are skipped. Interrupted runs resume from the last completed
    training block.

Example:
    python run_corruption_context_mnist_multicontext.py \
        --out_dir plots/mnist_corruption_context_multi_pcK0_v1 \
        --context_sequence clean,gaussian_noise,motion_blur,pixelation,brightness \
        --gaussian_std 0.275 \
        --motion_kernel 11 \
        --pixel_size 5 \
        --seeds 0,1,2 \
        --block_epochs 2 \
        --cycles 3 \
        --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from config import Config
from model import MLP
from pc import initialize_pc_train_state
from engine import evaluate, train_one_epoch_pc
from astro_phagocytosis import attach_phagocytosis_from_cfg, print_phagocytosis_summary
from full_model_diagnostics import generate_full_model_diagnostics

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

CORRUPTION_LABELS = {
    "clean": "Clean",
    "gaussian_noise": "Gaussian noise",
    "motion_blur": "Motion blur",
    "brightness": "Brightness",
    "pixelation": "Pixelation",
}
VALID_CORRUPTIONS = tuple(CORRUPTION_LABELS.keys())

VARIANT_LABELS = {
    "pc_K0": "PC, K=0",
    "pc_memory_K2": "PC + memory",
    "astro_gain_only": "Astro gain only",
    "astro_gain_state_coupling": "Astro gain + state coupling",
    "astro_gain_state_coupling_leakage": "Astro gain + state coupling + leakage",
    "passive_astro_pc": "Passive Astro-PC null",
    "full_astro_pc": "Full Astro-PC",
}

DEFAULT_VARIANTS = tuple(VARIANT_LABELS)
VARIANT_ORDER = {name: idx for idx, name in enumerate(DEFAULT_VARIANTS)}


# -----------------------------------------------------------------------------
# Reproducibility and config helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_cfg(base: Config, **updates: Any) -> Config:
    valid = {f.name for f in fields(Config)}
    cleaned = {k: v for k, v in updates.items() if k in valid}
    return replace(base, **cleaned)


def cfg_to_jsonable(cfg: Config) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["dtype"] = str(payload.get("dtype"))
    return payload


def parse_seeds(text: str | None, fallback_seed: int) -> list[int]:
    if text is None or not str(text).strip():
        return [int(fallback_seed)]
    seeds = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds



def parse_variants(text: str | None) -> list[str]:
    if text is None or not str(text).strip() or str(text).strip().lower() == "all":
        return list(DEFAULT_VARIANTS)

    requested = [name.strip() for name in str(text).split(",") if name.strip()]
    unknown = [name for name in requested if name not in VARIANT_LABELS]
    if unknown:
        raise ValueError(
            f"Unknown variants {unknown}. Expected one of {list(VARIANT_LABELS)} or 'all'."
        )
    return unique_preserve_order(requested)


def parse_context_sequence(text: str | None, *, fallback_corruption: str) -> list[str]:
    if text is None or not text.strip():
        sequence = ["clean", fallback_corruption]
    else:
        sequence = [part.strip() for part in text.split(",") if part.strip()]

    if len(sequence) < 2:
        raise ValueError("The context sequence must contain at least two contexts.")

    unknown = [c for c in sequence if c not in VALID_CORRUPTIONS]
    if unknown:
        raise ValueError(
            f"Unknown corruption contexts {unknown}. Expected one of {list(VALID_CORRUPTIONS)}."
        )
    return sequence


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------


def status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint_last.pt"


def load_status(run_dir: Path) -> dict[str, Any] | None:
    path = status_path(run_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_status(
    run_dir: Path,
    *,
    status: str,
    model_name: str,
    model_label: str,
    seed: int,
    completed_block: int,
    total_blocks: int,
    global_epoch: int,
    stop_reason: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "model_name": model_name,
        "model_label": model_label,
        "seed": int(seed),
        "completed_block": int(completed_block),
        "total_blocks": int(total_blocks),
        "global_epoch": int(global_epoch),
        "stop_reason": stop_reason,
    }
    with status_path(run_dir).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _torch_load(path: Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def collect_phago_state(model: MLP) -> dict[str, Any] | None:
    phago = getattr(model, "phago", None)
    if not phago:
        return None
    return {str(layer_idx): ph.state_dict() for layer_idx, ph in phago.items()}


def restore_phago_state(model: MLP, phago_state: dict[str, Any] | None) -> None:
    if phago_state is None:
        return
    phago = getattr(model, "phago", None)
    if not phago:
        return
    for layer_idx, state in phago_state.items():
        idx = int(layer_idx)
        if idx in phago:
            phago[idx].load_state_dict(state)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except pd.errors.EmptyDataError:
        return []


def save_checkpoint(
    *,
    run_dir: Path,
    model: MLP,
    pc_state: Any,
    cfg: Config,
    spec: dict[str, Any],
    seed: int,
    completed_block: int,
    global_epoch: int,
    eval_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    method_metrics: list[dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eval_rows).to_csv(run_dir / "context_eval_metrics.csv", index=False)
    pd.DataFrame(train_rows).to_csv(run_dir / "train_metrics.csv", index=False)
    with (run_dir / "method_metrics.pkl").open("wb") as f:
        pickle.dump(method_metrics, f)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "phago_state": collect_phago_state(model),
            "pc_state": pc_state,
            "cfg": cfg_to_jsonable(cfg),
            "variant": spec,
            "seed": int(seed),
            "completed_block": int(completed_block),
            "global_epoch": int(global_epoch),
            "eval_rows": eval_rows,
            "train_rows": train_rows,
            "method_metrics": method_metrics,
        },
        checkpoint_path(run_dir),
    )


LAYER_DIAGNOSTIC_KEYS = {
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


def compact_metric_record(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Remove list-valued layer diagnostics after streaming them to CSV.

    This keeps method_metrics.pkl smaller during long context-switching runs.
    """
    return {
        key: value for key, value in rec.items() if key not in LAYER_DIAGNOSTIC_KEYS
    }


def append_layer_diagnostics_for_context(
    records: list[dict[str, Any]],
    out_path: Path,
    *,
    seed: int,
    model_name: str,
    model_label: str,
    block_index: int,
    local_epoch: int,
    trained_context: str,
    trained_context_label: str,
) -> None:
    """
    Append batch-level layer diagnostics to a long-format CSV file.

    The diagnostics are written at batch resolution, with context metadata
    attached to each row.
    """
    rows: list[dict[str, Any]] = []

    for rec in records:
        epoch = int(rec.get("epoch", 0))
        step = int(rec.get("step", 0))

        for source_key, metric_name in LAYER_DIAGNOSTIC_KEYS.items():
            values = rec.get(source_key)
            if not isinstance(values, (list, tuple)):
                continue

            for layer_idx, value in enumerate(values, start=1):
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue

                if not pd.notna(value_f):
                    continue

                rows.append(
                    {
                        "seed": int(seed),
                        "model_name": model_name,
                        "model_label": model_label,
                        "block_index": int(block_index),
                        "local_epoch": int(local_epoch),
                        "trained_context": trained_context,
                        "trained_context_label": trained_context_label,
                        "epoch": epoch,
                        "step": step,
                        "source_key": source_key,
                        "metric": metric_name,
                        "layer": int(layer_idx),
                        "value": value_f,
                    }
                )

    if not rows:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    pd.DataFrame(rows).to_csv(out_path, mode="a", header=write_header, index=False)


def trim_layer_diagnostics_csv(csv_path: Path, *, max_global_epoch: int) -> None:
    """
    Remove rows after the last completed epoch when resuming a run.

    This prevents duplicated diagnostics if a run was interrupted mid-block.
    """
    if not csv_path.exists():
        return

    if max_global_epoch < 1:
        csv_path.unlink()
        return

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        csv_path.unlink()
        return

    df = df[pd.to_numeric(df["epoch"], errors="coerce") <= max_global_epoch]
    df.to_csv(csv_path, index=False)


def write_multicontext_metrics_csv(run_dir: Path) -> None:
    """
    Build a metrics.csv file compatible with full_model_diagnostics.py.

    Test metrics are averaged across all evaluated contexts at each global epoch.
    """
    train_path = run_dir / "train_metrics.csv"
    eval_path = run_dir / "context_eval_metrics.csv"

    if not train_path.exists():
        return

    train_df = pd.read_csv(train_path)
    if train_df.empty:
        return

    train_summary = train_df.groupby("global_epoch", as_index=False).agg(
        epoch=("global_epoch", "first"),
        train_loss=("train_loss", "mean"),
        train_acc=("train_acc", "mean"),
    )

    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
    else:
        eval_df = pd.DataFrame()

    if not eval_df.empty:
        eval_summary = eval_df.groupby("global_epoch", as_index=False).agg(
            test_loss=("test_loss", "mean"),
            test_acc=("test_acc", "mean"),
        )
        metrics_df = train_summary.merge(eval_summary, on="global_epoch", how="left")
    else:
        metrics_df = train_summary
        metrics_df["test_loss"] = float("nan")
        metrics_df["test_acc"] = float("nan")

    metrics_df = metrics_df.sort_values("epoch")
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)


def maybe_generate_run_diagnostics(run_dir: Path, *, verbose: bool = False) -> None:
    """
    Generate internal diagnostics for one model/seed run directory.

    New runs should contain:
        diagnostics/layer_internal_diagnostics_stream.csv

    Older runs may still contain layer-wise diagnostics inside:
        method_metrics.pkl

    The streamed CSV is preferred by full_model_diagnostics.py when available.
    """
    try:
        write_multicontext_metrics_csv(run_dir)

        metrics_path = run_dir / "metrics.csv"
        stream_path = run_dir / "diagnostics" / "layer_internal_diagnostics_stream.csv"
        method_metrics_path = run_dir / "method_metrics.pkl"

        if not metrics_path.exists():
            print(f"[warning] No metrics.csv found for {run_dir}")
            return

        if not stream_path.exists() and not method_metrics_path.exists():
            print(f"[warning] No diagnostics source found for {run_dir}")
            return

        generate_full_model_diagnostics(
            run_dir=run_dir,
            out_dir=run_dir / "diagnostics",
            verbose=verbose,
        )

    except Exception as exc:
        print(f"[warning] Could not generate diagnostics for {run_dir}: {exc}")


# -----------------------------------------------------------------------------
# Context MNIST dataset
# -----------------------------------------------------------------------------


def _apply_corruption(
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


class ContextMNIST(Dataset):
    """
    MNIST with an appended one-hot context cue.

    All contexts use normal labels in corruption mode. Returned x has shape
    [784 + num_contexts].
    """

    def __init__(
        self,
        root: str,
        *,
        train: bool,
        context_id: int,
        num_contexts: int,
        context_label: str,
        label_perm: torch.Tensor,
        corruption: str = "clean",
        context_scale: float = 1.0,
        subset_size: int | None = None,
        subset_seed: int = 0,
        gaussian_std: float = 0.275,
        brightness_delta: float = 0.25,
        motion_kernel: int = 11,
        pixel_size: int = 5,
    ) -> None:
        super().__init__()
        if not (0 <= int(context_id) < int(num_contexts)):
            raise ValueError("context_id must be in [0, num_contexts).")

        base = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

        if subset_size is not None and subset_size > 0 and subset_size < len(base):
            generator = torch.Generator().manual_seed(subset_seed)
            indices = torch.randperm(len(base), generator=generator)[
                :subset_size
            ].tolist()
            base = Subset(base, indices)

        self.base = base
        self.context_id = int(context_id)
        self.num_contexts = int(num_contexts)
        self.context_label = str(context_label)
        self.context_scale = float(context_scale)
        self.label_perm = label_perm.to(dtype=torch.long).clone()
        self.corruption = str(corruption)
        self.noise_seed = int(subset_seed) + 17_000 * (1 + int(context_id))
        self.gaussian_std = float(gaussian_std)
        self.brightness_delta = float(brightness_delta)
        self.motion_kernel = int(motion_kernel)
        self.pixel_size = int(pixel_size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        x = _apply_corruption(
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
        x = x.view(-1).to(dtype=torch.float32)

        cue = torch.zeros(self.num_contexts, dtype=torch.float32)
        cue[self.context_id] = self.context_scale

        y = int(y)
        y_out = int(self.label_perm[y])
        return torch.cat([x, cue], dim=0), y_out


def make_context_loaders(
    *,
    root: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    context_task: str,
    context_sequence: list[str],
    context_scale: float,
    train_subset: int | None,
    test_subset: int | None,
    seed: int,
    gaussian_std: float,
    brightness_delta: float,
    motion_kernel: int,
    pixel_size: int,
) -> tuple[
    dict[str, DataLoader], dict[str, DataLoader], dict[str, Any], list[str], int
]:
    """Build train/test loaders for all contexts used in the schedule."""
    identity = torch.arange(10, dtype=torch.long)
    perm_list = [3, 8, 1, 6, 0, 9, 2, 5, 7, 4]
    perm = torch.tensor(perm_list, dtype=torch.long)

    if context_task == "corruption":
        contexts = unique_preserve_order(context_sequence)
        context_info: dict[str, Any] = {}
        label_perms: dict[str, torch.Tensor] = {}
        corruptions: dict[str, str] = {}
        for idx, corruption in enumerate(contexts):
            context_info[corruption] = {
                "id": idx,
                "label": CORRUPTION_LABELS.get(corruption, corruption),
                "corruption": corruption,
                "label_perm": identity.tolist(),
            }
            label_perms[corruption] = identity
            corruptions[corruption] = corruption
        permutation_B = None

    elif context_task == "permuted_labels":
        contexts = ["A", "B"]
        context_info = {
            "A": {
                "id": 0,
                "label": "Normal labels",
                "corruption": "clean",
                "label_perm": identity.tolist(),
            },
            "B": {
                "id": 1,
                "label": "Permuted labels",
                "corruption": "clean",
                "label_perm": perm_list,
            },
        }
        label_perms = {"A": identity, "B": perm}
        corruptions = {"A": "clean", "B": "clean"}
        permutation_B = perm_list
    else:
        raise ValueError(f"Unknown context_task: {context_task}")

    num_contexts = len(contexts)
    common_kwargs = dict(
        root=root,
        num_contexts=num_contexts,
        context_scale=context_scale,
        gaussian_std=gaussian_std,
        brightness_delta=brightness_delta,
        motion_kernel=motion_kernel,
        pixel_size=pixel_size,
    )

    train_loaders: dict[str, DataLoader] = {}
    test_loaders: dict[str, DataLoader] = {}
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    for context_key in contexts:
        info = context_info[context_key]
        train_dataset = ContextMNIST(
            **common_kwargs,
            train=True,
            context_id=int(info["id"]),
            context_label=str(info["label"]),
            label_perm=label_perms[context_key],
            corruption=corruptions[context_key],
            subset_size=train_subset,
            subset_seed=seed,
        )
        test_dataset = ContextMNIST(
            **common_kwargs,
            train=False,
            context_id=int(info["id"]),
            context_label=str(info["label"]),
            label_perm=label_perms[context_key],
            corruption=corruptions[context_key],
            subset_size=test_subset,
            subset_seed=seed + 1000,
        )
        train_loaders[context_key] = DataLoader(
            train_dataset, shuffle=True, **loader_kwargs
        )
        test_loaders[context_key] = DataLoader(
            test_dataset, shuffle=False, **loader_kwargs
        )

    context_info["permutation_B"] = permutation_B
    return train_loaders, test_loaders, context_info, contexts, num_contexts


# -----------------------------------------------------------------------------
# Model variants
# -----------------------------------------------------------------------------


def variant_specs(
    base_cfg: Config,
    variants: list[str] | tuple[str, ...] = DEFAULT_VARIANTS,
) -> list[dict[str, Any]]:
    k = base_cfg.pc_frac_memory_len
    rho = base_cfg.pc_frac_rho

    overrides_by_variant: dict[str, dict[str, Any]] = {
        "pc_K0": {
            "astro_controller_enabled": False,
            "astro_controller_leak_enabled": False,
            "astro_controller_state_coupling_enabled": False,
            "phago_enabled": False,
            "pc_frac_memory_len": 0,
            "pc_frac_rho": rho,
        },
        "pc_memory_K2": {
            "astro_controller_enabled": False,
            "astro_controller_leak_enabled": False,
            "astro_controller_state_coupling_enabled": False,
            "phago_enabled": False,
            "pc_frac_memory_len": k,
            "pc_frac_rho": rho,
        },
        "astro_gain_only": {
            "astro_controller_enabled": True,
            "astro_controller_dynamics": "astro_state",
            "astro_controller_leak_enabled": False,
            "astro_controller_state_coupling_enabled": False,
            "phago_enabled": False,
            "pc_frac_memory_len": k,
            "pc_frac_rho": rho,
        },
        "astro_gain_state_coupling": {
            "astro_controller_enabled": True,
            "astro_controller_dynamics": "astro_state",
            "astro_controller_leak_enabled": False,
            "astro_controller_state_coupling_enabled": True,
            "astro_controller_state_coupling_strength": 0.1,
            "phago_enabled": False,
            "pc_frac_memory_len": k,
            "pc_frac_rho": rho,
        },
        "astro_gain_state_coupling_leakage": {
            "astro_controller_enabled": True,
            "astro_controller_dynamics": "astro_state",
            "astro_controller_leak_enabled": True,
            "astro_controller_state_coupling_enabled": True,
            "astro_controller_state_coupling_strength": 0.1,
            "phago_enabled": False,
            "pc_frac_memory_len": k,
            "pc_frac_rho": rho,
        },
        "passive_astro_pc": {
            "astro_controller_enabled": True,
            "astro_controller_dynamics": "passive_slow_gain",
            "astro_controller_passive_state_drive": 0.5,
            "astro_controller_leak_enabled": True,
            "astro_controller_state_coupling_enabled": True,
            "astro_controller_state_coupling_strength": 0.1,
            "phago_enabled": True,
            "pc_frac_memory_len": k,
            "pc_frac_rho": rho,
        },
        "full_astro_pc": {
            "astro_controller_enabled": True,
            "astro_controller_dynamics": "astro_state",
            "astro_controller_leak_enabled": True,
            "astro_controller_state_coupling_enabled": True,
            "astro_controller_state_coupling_strength": 0.1,
            "phago_enabled": True,
            "pc_frac_memory_len": k,
            "pc_frac_rho": rho,
        },
    }

    specs: list[dict[str, Any]] = []
    for name in variants:
        if name not in overrides_by_variant:
            raise ValueError(
                f"Unknown variant: {name}. Expected one of {list(VARIANT_LABELS)}."
            )
        specs.append(
            {
                "name": name,
                "label": VARIANT_LABELS[name],
                "overrides": dict(overrides_by_variant[name]),
            }
        )
    return specs



def build_context_model(cfg: Config, device: torch.device, *, context_dim: int) -> MLP:
    model = MLP(
        in_dim=784 + context_dim,
        hidden_dims=cfg.get_hidden_dims(),
        out_dim=10,
        activation_function=cfg.activation_function,
    ).to(device=device, dtype=cfg.dtype)
    attach_phagocytosis_from_cfg(model, cfg)
    return model



def sort_by_variant_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model_name" not in df.columns:
        return df

    out = df.copy()
    out["_variant_order"] = out["model_name"].map(VARIANT_ORDER).fillna(999).astype(int)
    sort_cols = ["_variant_order"]
    if "seed" in out.columns:
        sort_cols.append("seed")
    out = out.sort_values(sort_cols).drop(columns=["_variant_order"])
    return out


# -----------------------------------------------------------------------------
# Metrics and summaries
# -----------------------------------------------------------------------------


def compute_forgetting(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if eval_df.empty:
        return pd.DataFrame()

    context_keys = list(eval_df["eval_context"].drop_duplicates())
    group_cols = ["seed", "model_name"] if "seed" in eval_df.columns else ["model_name"]

    for group_key, model_df in eval_df.groupby(group_cols):
        if isinstance(group_key, tuple):
            seed = int(group_key[0])
            model_name = str(group_key[1])
        else:
            seed = int(model_df["seed"].iloc[0]) if "seed" in model_df.columns else 0
            model_name = str(group_key)

        model_df = model_df.sort_values("block_index")
        for eval_context in context_keys:
            best_so_far = None
            context_rows = model_df[model_df["eval_context"] == eval_context]
            for _, rec in context_rows.iterrows():
                acc = float(rec["test_acc"])
                trained_context = str(rec["trained_context"])
                if best_so_far is not None and trained_context != eval_context:
                    rows.append(
                        {
                            "seed": seed,
                            "model_name": model_name,
                            "model_label": rec["model_label"],
                            "block_index": int(rec["block_index"]),
                            "trained_context": trained_context,
                            "trained_context_label": rec.get(
                                "trained_context_label", trained_context
                            ),
                            "eval_context": eval_context,
                            "eval_context_label": rec.get(
                                "eval_context_label", eval_context
                            ),
                            "best_previous_acc": best_so_far,
                            "current_acc": acc,
                            "forgetting": max(0.0, best_so_far - acc),
                        }
                    )
                best_so_far = acc if best_so_far is None else max(best_so_far, acc)
    return pd.DataFrame(rows)


def summarize_forgetting_per_seed(forgetting_df: pd.DataFrame) -> pd.DataFrame:
    if forgetting_df.empty:
        return pd.DataFrame()
    per_seed = forgetting_df.groupby(
        ["seed", "model_name", "model_label"], as_index=False
    ).agg(
        mean_forgetting=("forgetting", "mean"),
        max_forgetting=("forgetting", "max"),
        num_events=("forgetting", "count"),
    )
    return sort_by_variant_order(per_seed)


def aggregate_forgetting_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame()
    summary = (
        per_seed.groupby(["model_name", "model_label"], as_index=False)
        .agg(
            mean_forgetting=("mean_forgetting", "mean"),
            std_forgetting=("mean_forgetting", "std"),
            max_forgetting=("max_forgetting", "max"),
            num_seeds=("seed", "nunique"),
        )
        .fillna({"std_forgetting": 0.0})
    )
    return sort_by_variant_order(summary)


def summarize_context_switching_per_seed(
    eval_df: pd.DataFrame, forgetting_per_seed: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if eval_df.empty:
        return pd.DataFrame()

    group_cols = ["seed", "model_name"] if "seed" in eval_df.columns else ["model_name"]
    for group_key, model_df in eval_df.groupby(group_cols):
        if isinstance(group_key, tuple):
            seed = int(group_key[0])
            model_name = str(group_key[1])
        else:
            seed = int(model_df["seed"].iloc[0]) if "seed" in model_df.columns else 0
            model_name = str(group_key)
        model_label = str(model_df["model_label"].iloc[0])

        current_vals = [
            float(r["test_acc"])
            for _, r in model_df.iterrows()
            if r["eval_context"] == r["trained_context"]
        ]
        non_current_vals = [
            float(r["test_acc"])
            for _, r in model_df.iterrows()
            if r["eval_context"] != r["trained_context"]
        ]
        final_block = int(model_df["block_index"].max())
        final_df = model_df[model_df["block_index"] == final_block]

        fs = (
            forgetting_per_seed[
                (forgetting_per_seed["seed"] == seed)
                & (forgetting_per_seed["model_name"] == model_name)
            ]
            if not forgetting_per_seed.empty
            else pd.DataFrame()
        )
        mean_forgetting = float(fs["mean_forgetting"].iloc[0]) if not fs.empty else 0.0
        max_forgetting = float(fs["max_forgetting"].iloc[0]) if not fs.empty else 0.0

        rows.append(
            {
                "seed": seed,
                "model_name": model_name,
                "model_label": model_label,
                "mean_current_acc": sum(current_vals) / max(1, len(current_vals)),
                "mean_non_current_acc": sum(non_current_vals)
                / max(1, len(non_current_vals)),
                "final_mean_acc_all_contexts": float(final_df["test_acc"].mean()),
                "final_min_acc_all_contexts": float(final_df["test_acc"].min()),
                "mean_forgetting": mean_forgetting,
                "max_forgetting": max_forgetting,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "seed"])


def aggregate_context_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame()
    metrics = [
        "mean_current_acc",
        "mean_non_current_acc",
        "final_mean_acc_all_contexts",
        "final_min_acc_all_contexts",
        "mean_forgetting",
        "max_forgetting",
    ]
    agg = per_seed.groupby(["model_name", "model_label"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        num_seeds=("seed", "nunique"),
    )
    for col in agg.columns:
        if col.endswith("_std"):
            agg[col] = agg[col].fillna(0.0)
    return sort_by_variant_order(agg)


def make_final_context_accuracy_table_per_seed(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if eval_df.empty:
        return pd.DataFrame()
    for (seed, model_name), model_df in eval_df.groupby(["seed", "model_name"]):
        final_block = int(model_df["block_index"].max())
        final_df = model_df[model_df["block_index"] == final_block].copy()
        final_df["seed"] = seed
        rows.append(final_df)
    final_df = pd.concat(rows, ignore_index=True)
    return final_df.pivot_table(
        index=["seed", "model_name", "model_label"],
        columns="eval_context_label",
        values="test_acc",
        aggfunc="mean",
    ).reset_index()


def aggregate_final_context_table(per_seed_table: pd.DataFrame) -> pd.DataFrame:
    if per_seed_table.empty:
        return pd.DataFrame()
    context_cols = [
        c
        for c in per_seed_table.columns
        if c not in {"seed", "model_name", "model_label"}
    ]
    table = per_seed_table.groupby(["model_name", "model_label"], as_index=False).agg(
        **{c: (c, "mean") for c in context_cols}
    )
    return sort_by_variant_order(table)


def save_latex_table(df: pd.DataFrame, path: Path, *, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pretty = df.copy()
    for col in pretty.columns:
        if col not in {"seed", "model_name", "model_label", "num_seeds"}:
            pretty[col] = pd.to_numeric(pretty[col], errors="ignore")
            if pd.api.types.is_numeric_dtype(pretty[col]):
                pretty[col] = (100.0 * pretty[col]).map(lambda x: f"{x:.2f}")
    with path.open("w", encoding="utf-8") as f:
        f.write(
            pretty.to_latex(index=False, caption=caption, label=label, escape=False)
        )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_context_accuracy(eval_df: pd.DataFrame, out_dir: Path) -> None:
    if eval_df.empty:
        return
    for model_name, model_df in eval_df.groupby("model_name"):
        label = str(model_df["model_label"].iloc[0])
        plt.figure(figsize=(9, 5))
        for eval_context, sub in model_df.groupby("eval_context"):
            context_label = str(sub["eval_context_label"].iloc[0])
            agg = sub.groupby("block_index", as_index=False).agg(
                test_acc=("test_acc", "mean")
            )
            plt.plot(
                agg["block_index"],
                100.0 * agg["test_acc"],
                marker="o",
                linewidth=1.5,
                label=f"eval {context_label}",
            )
        for _, rec in model_df.drop_duplicates("block_index").iterrows():
            x = int(rec["block_index"])
            trained = rec.get("trained_context_label", rec["trained_context"])
            plt.axvline(x, alpha=0.10)
            plt.text(
                x, 5.0, str(trained), ha="center", va="bottom", fontsize=7, rotation=90
            )
        plt.xlabel("Training block")
        plt.ylabel("Test accuracy (%)")
        plt.ylim(0, 105)
        plt.title(f"Context-switching accuracy: {label}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        _save_plot(out_dir / f"context_accuracy_{model_name}.png")


def plot_current_and_non_current_accuracy(eval_df: pd.DataFrame, out_dir: Path) -> None:
    if eval_df.empty:
        return
    rows: list[dict[str, Any]] = []
    for (seed, model_name, block_index), block_df in eval_df.groupby(
        ["seed", "model_name", "block_index"]
    ):
        trained = str(block_df["trained_context"].iloc[0])
        current = block_df[block_df["eval_context"] == trained]
        non_current = block_df[block_df["eval_context"] != trained]
        rows.append(
            {
                "seed": seed,
                "model_name": model_name,
                "model_label": block_df["model_label"].iloc[0],
                "block_index": block_index,
                "current_acc": (
                    float(current["test_acc"].iloc[0])
                    if not current.empty
                    else float("nan")
                ),
                "mean_non_current_acc": (
                    float(non_current["test_acc"].mean())
                    if not non_current.empty
                    else float("nan")
                ),
            }
        )
    df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 6))
    for model_name, model_df in df.groupby("model_name"):
        label = str(model_df["model_label"].iloc[0])
        agg = model_df.groupby("block_index", as_index=False).agg(
            current_acc=("current_acc", "mean"),
            mean_non_current_acc=("mean_non_current_acc", "mean"),
        )
        plt.plot(
            agg["block_index"],
            100.0 * agg["current_acc"],
            marker="o",
            label=f"{label}: trained context",
        )
        plt.plot(
            agg["block_index"],
            100.0 * agg["mean_non_current_acc"],
            marker="x",
            linestyle="--",
            label=f"{label}: mean non-current contexts",
        )
    plt.xlabel("Training block")
    plt.ylabel("Test accuracy (%)")
    plt.ylim(0, 105)
    plt.title("Trained-context vs non-current-context accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    _save_plot(out_dir / "current_vs_non_current_context_accuracy.png")


def plot_forgetting(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return
    labels = summary_df["model_label"].tolist()
    vals = 100.0 * summary_df["mean_forgetting"].astype(float).to_numpy()
    errs = (
        100.0
        * summary_df.get("std_forgetting", pd.Series([0.0] * len(summary_df)))
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        range(len(labels)),
        vals,
        yerr=errs if any(e > 0 for e in errs) else None,
        capsize=3,
    )
    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("Mean forgetting (percentage points)")
    plt.title("Context-switching forgetting")
    plt.grid(True, axis="y", alpha=0.3)
    _save_plot(out_dir / "forgetting_summary.png")


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------


def run_variant(
    *,
    spec: dict[str, Any],
    base_cfg: Config,
    train_loaders: dict[str, DataLoader],
    test_loaders: dict[str, DataLoader],
    context_info: dict[str, Any],
    schedule: list[str],
    block_epochs: int,
    out_root: Path,
    seed: int,
    context_dim: int,
    num_seeds: int,
    force_restart: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = make_cfg(base_cfg, seed=seed, **spec["overrides"])
    cfg.validate()
    pc_cfg = cfg.make_method_cfg(log_every=10_000)
    device = torch.device(cfg.device)

    run_name = f"{spec['name']}__seed{seed}" if num_seeds > 1 else spec["name"]
    run_dir = out_root / run_name
    if force_restart and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    layer_diagnostics_csv = diagnostics_dir / "layer_internal_diagnostics_stream.csv"

    status = load_status(run_dir)
    if status is not None and status.get("status") == "completed":
        print(f"[skip] Completed run found: {run_dir}")
        maybe_generate_run_diagnostics(run_dir, verbose=False)
        return read_csv_rows(run_dir / "context_eval_metrics.csv"), read_csv_rows(
            run_dir / "train_metrics.csv"
        )

    model = build_context_model(cfg, device, context_dim=context_dim)
    state = initialize_pc_train_state(model=model, cfg=pc_cfg)

    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    method_metrics: list[dict[str, Any]] = []
    global_epoch = 0
    completed_blocks = 0

    ckpt = checkpoint_path(run_dir)
    if ckpt.exists():
        print(f"[resume] Loading checkpoint: {ckpt}")
        checkpoint = _torch_load(ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        restore_phago_state(model, checkpoint.get("phago_state"))
        state = checkpoint.get("pc_state", state)
        eval_rows = list(checkpoint.get("eval_rows", [])) or read_csv_rows(
            run_dir / "context_eval_metrics.csv"
        )
        train_rows = list(checkpoint.get("train_rows", [])) or read_csv_rows(
            run_dir / "train_metrics.csv"
        )
        method_metrics = list(checkpoint.get("method_metrics", []))
        if not method_metrics and (run_dir / "method_metrics.pkl").exists():
            with (run_dir / "method_metrics.pkl").open("rb") as f:
                method_metrics = pickle.load(f)
        global_epoch = int(checkpoint.get("global_epoch", 0))
        completed_blocks = int(
            checkpoint.get("completed_block", checkpoint.get("completed_blocks", 0))
        )

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_to_jsonable(cfg), f, indent=2, sort_keys=True, default=str)

    print("\n" + "=" * 88)
    print(f"Variant: {spec['label']} ({spec['name']}) | seed={seed}")
    print(f"Run dir: {run_dir}")
    print(f"Completed blocks before start: {completed_blocks}/{len(schedule)}")
    print("=" * 88)

    save_status(
        run_dir,
        status="running",
        model_name=spec["name"],
        model_label=spec["label"],
        seed=seed,
        completed_block=completed_blocks,
        total_blocks=len(schedule),
        global_epoch=global_epoch,
    )

    try:
        for block_index in range(completed_blocks + 1, len(schedule) + 1):
            context_name = schedule[block_index - 1]
            context_label = context_info[context_name]["label"]
            print(
                f"\n[Block {block_index}/{len(schedule)}] Train context {context_name}: {context_label}"
            )
            loader = train_loaders[context_name]

            for local_epoch in range(1, block_epochs + 1):
                global_epoch += 1

                before_metrics = len(method_metrics)

                train_loss, train_acc, state = train_one_epoch_pc(
                    model=model,
                    loader=loader,
                    device=device,
                    epoch=global_epoch,
                    cfg=pc_cfg,
                    state=state,
                    metrics_store=method_metrics,
                )

                new_records = method_metrics[before_metrics:]

                append_layer_diagnostics_for_context(
                    new_records,
                    layer_diagnostics_csv,
                    seed=seed,
                    model_name=spec["name"],
                    model_label=spec["label"],
                    block_index=block_index,
                    local_epoch=local_epoch,
                    trained_context=context_name,
                    trained_context_label=context_label,
                )

                # Keep method_metrics.pkl compact after writing layer-wise diagnostics.
                method_metrics[before_metrics:] = [
                    compact_metric_record(rec) for rec in new_records
                ]

                train_rows.append(
                    {
                        "seed": int(seed),
                        "model_name": spec["name"],
                        "model_label": spec["label"],
                        "global_epoch": global_epoch,
                        "block_index": block_index,
                        "local_epoch": local_epoch,
                        "trained_context": context_name,
                        "trained_context_label": context_label,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                    }
                )
                print(
                    f"  epoch {local_epoch}/{block_epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}"
                )

            for eval_context, test_loader in test_loaders.items():
                eval_label = context_info[eval_context]["label"]
                test_loss, test_acc = evaluate(model, test_loader, device)
                eval_rows.append(
                    {
                        "seed": int(seed),
                        "model_name": spec["name"],
                        "model_label": spec["label"],
                        "block_index": block_index,
                        "trained_context": context_name,
                        "trained_context_label": context_label,
                        "eval_context": eval_context,
                        "eval_context_label": eval_label,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "global_epoch": global_epoch,
                    }
                )
                print(
                    f"  eval {eval_context} ({eval_label}): loss={test_loss:.4f}, acc={test_acc:.4f}"
                )

            completed_blocks = block_index
            save_checkpoint(
                run_dir=run_dir,
                model=model,
                pc_state=state,
                cfg=cfg,
                spec=spec,
                seed=seed,
                completed_block=completed_blocks,
                global_epoch=global_epoch,
                eval_rows=eval_rows,
                train_rows=train_rows,
                method_metrics=method_metrics,
            )
            save_status(
                run_dir,
                status="running",
                model_name=spec["name"],
                model_label=spec["label"],
                seed=seed,
                completed_block=completed_blocks,
                total_blocks=len(schedule),
                global_epoch=global_epoch,
            )

    except KeyboardInterrupt:
        save_status(
            run_dir,
            status="interrupted",
            model_name=spec["name"],
            model_label=spec["label"],
            seed=seed,
            completed_block=completed_blocks,
            total_blocks=len(schedule),
            global_epoch=global_epoch,
            stop_reason="KeyboardInterrupt",
        )
        raise
    except Exception as exc:
        save_status(
            run_dir,
            status="failed",
            model_name=spec["name"],
            model_label=spec["label"],
            seed=seed,
            completed_block=completed_blocks,
            total_blocks=len(schedule),
            global_epoch=global_epoch,
            stop_reason=f"{type(exc).__name__}: {exc}",
        )
        raise

    if cfg.phago_enabled:
        print_phagocytosis_summary(model)

    save_status(
        run_dir,
        status="completed",
        model_name=spec["name"],
        model_label=spec["label"],
        seed=seed,
        completed_block=completed_blocks,
        total_blocks=len(schedule),
        global_epoch=global_epoch,
    )
    maybe_generate_run_diagnostics(run_dir, verbose=False)

    return eval_rows, train_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST multi-context switching experiment."
    )
    parser.add_argument("--out_dir", default="plots/mnist_corruption_context_multi_v1")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Single-seed fallback if --seeds is omitted.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds, e.g. 0,1,2. Overrides --seed.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Deprecated alias. Use --block_epochs."
    )
    parser.add_argument("--block_epochs", type=int, default=2)
    parser.add_argument(
        "--cycles", type=int, default=3, help="Number of full context-sequence cycles."
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--context_scale", type=float, default=1.0)
    parser.add_argument(
        "--train_subset", type=int, default=0, help="0 means full training set."
    )
    parser.add_argument(
        "--test_subset", type=int, default=0, help="0 means full test set."
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--no_phago",
        action="store_true",
        help="Disable phagocytosis in the full/passive variants.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help=(
            "Comma-separated variant names to run, or 'all'. "
            f"Available names: {', '.join(DEFAULT_VARIANTS)}."
        ),
    )
    parser.add_argument(
        "--force_restart",
        action="store_true",
        help="Delete existing per-variant run folders and restart.",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Rebuild aggregate CSVs/plots from existing run folders without training.",
    )
    parser.add_argument(
        "--context_task",
        choices=["corruption", "permuted_labels"],
        default="corruption",
        help="corruption: normal labels across multiple input corruptions. permuted_labels: legacy strict label flip-flop.",
    )
    parser.add_argument(
        "--context_sequence",
        default=None,
        help="Comma-separated context sequence for corruption mode. Example: clean,gaussian_noise,motion_blur,pixelation,brightness.",
    )
    parser.add_argument(
        "--corruption_B",
        choices=["gaussian_noise", "motion_blur", "brightness", "pixelation"],
        default="gaussian_noise",
    )
    parser.add_argument("--gaussian_std", type=float, default=0.275)
    parser.add_argument("--brightness_delta", type=float, default=0.25)
    parser.add_argument("--motion_kernel", type=int, default=11)
    parser.add_argument("--pixel_size", type=int, default=5)
    return parser.parse_args()


def collect_existing_rows(
    out_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        eval_rows.extend(read_csv_rows(run_dir / "context_eval_metrics.csv"))
        train_rows.extend(read_csv_rows(run_dir / "train_metrics.csv"))
    return eval_rows, train_rows


def write_aggregate_outputs(
    out_root: Path, eval_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]
) -> None:
    eval_df = pd.DataFrame(eval_rows)
    train_df = pd.DataFrame(train_rows)

    eval_df.to_csv(out_root / "context_eval_metrics.csv", index=False)
    train_df.to_csv(out_root / "train_metrics.csv", index=False)

    forgetting_df = compute_forgetting(eval_df)
    forgetting_per_seed = summarize_forgetting_per_seed(forgetting_df)
    forgetting_summary = aggregate_forgetting_summary(forgetting_per_seed)
    context_per_seed = summarize_context_switching_per_seed(
        eval_df, forgetting_per_seed
    )
    context_summary = aggregate_context_summary(context_per_seed)
    final_per_seed = make_final_context_accuracy_table_per_seed(eval_df)
    final_context_table = aggregate_final_context_table(final_per_seed)

    forgetting_df.to_csv(out_root / "forgetting_events.csv", index=False)
    forgetting_per_seed.to_csv(
        out_root / "forgetting_summary_per_seed.csv", index=False
    )
    forgetting_summary.to_csv(out_root / "forgetting_summary.csv", index=False)
    context_per_seed.to_csv(
        out_root / "context_switching_summary_per_seed.csv", index=False
    )
    context_summary.to_csv(out_root / "context_switching_summary.csv", index=False)
    final_per_seed.to_csv(
        out_root / "final_context_accuracy_table_per_seed.csv", index=False
    )
    final_context_table.to_csv(
        out_root / "final_context_accuracy_table.csv", index=False
    )

    save_latex_table(
        context_summary,
        out_root / "context_switching_summary.tex",
        caption="Summary of multi-context corruption switching performance.",
        label="tab:context_switching_summary",
    )
    save_latex_table(
        final_context_table,
        out_root / "final_context_accuracy_table.tex",
        caption="Final accuracy across all evaluated contexts after the last training block.",
        label="tab:final_context_accuracy",
    )

    plot_dir = out_root / "plots"
    plot_context_accuracy(eval_df, plot_dir)
    plot_current_and_non_current_accuracy(eval_df, plot_dir)
    plot_forgetting(forgetting_summary, plot_dir)


def main() -> None:
    args = parse_args()
    if args.epochs is not None:
        args.block_epochs = args.epochs

    seeds = parse_seeds(args.seeds, args.seed)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        eval_rows, train_rows = collect_existing_rows(out_root)
        write_aggregate_outputs(out_root, eval_rows, train_rows)
        print(f"[summary_only] Rebuilt aggregate outputs in {out_root}")
        return

    if args.context_task == "corruption":
        context_sequence = parse_context_sequence(
            args.context_sequence, fallback_corruption=args.corruption_B
        )
    else:
        context_sequence = ["A", "B"]

    all_eval_rows: list[dict[str, Any]] = []
    all_train_rows: list[dict[str, Any]] = []
    manifest_payload: dict[str, Any] | None = None

    for seed in seeds:
        set_seed(seed)
        base_cfg = make_cfg(
            Config(),
            dataset="mnist",
            method="pc",
            seed=seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_subset = None if args.train_subset <= 0 else args.train_subset
        test_subset = None if args.test_subset <= 0 else args.test_subset

        train_loaders, test_loaders, context_info, contexts, context_dim = (
            make_context_loaders(
                root=base_cfg.mnist_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=base_cfg.pin_memory and torch.cuda.is_available(),
                context_task=args.context_task,
                context_sequence=context_sequence,
                context_scale=args.context_scale,
                train_subset=train_subset,
                test_subset=test_subset,
                seed=seed,
                gaussian_std=args.gaussian_std,
                brightness_delta=args.brightness_delta,
                motion_kernel=args.motion_kernel,
                pixel_size=args.pixel_size,
            )
        )
        schedule = [ctx for _ in range(args.cycles) for ctx in contexts]

        selected_variants = parse_variants(args.variants)
        specs = variant_specs(base_cfg, selected_variants)

        if args.no_phago:
            for spec in specs:
                spec["overrides"]["phago_enabled"] = False

        if manifest_payload is None:
            manifest_payload = {
                "description": "MNIST multi-context switching experiment",
                "context_task": args.context_task,
                "seeds": seeds,
                "contexts": contexts,
                "context_dim": context_dim,
                "schedule": schedule,
                "context_info": context_info,
                "block_epochs": args.block_epochs,
                "cycles": args.cycles,
                "batch_size": args.batch_size,
                "context_scale": args.context_scale,
                "train_subset": train_subset,
                "test_subset": test_subset,
                "corruption_sequence": (
                    context_sequence if args.context_task == "corruption" else None
                ),
                "corruption_parameters": {
                    "gaussian_std": args.gaussian_std,
                    "brightness_delta": args.brightness_delta,
                    "motion_kernel": args.motion_kernel,
                    "pixel_size": args.pixel_size,
                },
                "selected_variants": selected_variants,
                "available_variants": list(DEFAULT_VARIANTS),
                "variants": specs,
                "base_config": cfg_to_jsonable(base_cfg),
                "resume_behavior": "Completed per-variant/seed runs are skipped; interrupted runs resume from the last completed block.",
            }
            with (out_root / "context_manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest_payload, f, indent=2, sort_keys=True, default=str)

        for spec in specs:
            set_seed(seed)
            eval_rows, train_rows = run_variant(
                spec=spec,
                base_cfg=base_cfg,
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                context_info=context_info,
                schedule=schedule,
                block_epochs=args.block_epochs,
                out_root=out_root,
                seed=seed,
                context_dim=context_dim,
                num_seeds=len(seeds),
                force_restart=args.force_restart,
            )
            all_eval_rows.extend(eval_rows)
            all_train_rows.extend(train_rows)
            existing_eval_rows, existing_train_rows = collect_existing_rows(out_root)
            pd.DataFrame(existing_eval_rows).to_csv(
                out_root / "context_eval_metrics.csv", index=False
            )
            pd.DataFrame(existing_train_rows).to_csv(
                out_root / "train_metrics.csv", index=False
            )

    aggregate_eval_rows, aggregate_train_rows = collect_existing_rows(out_root)
    write_aggregate_outputs(out_root, aggregate_eval_rows, aggregate_train_rows)

    print("\nSaved outputs:")
    print(f"  {out_root / 'context_eval_metrics.csv'}")
    print(f"  {out_root / 'forgetting_events.csv'}")
    print(f"  {out_root / 'forgetting_summary.csv'}")
    print(f"  {out_root / 'context_switching_summary.csv'}")
    print(f"  {out_root / 'final_context_accuracy_table.csv'}")
    print(f"  {out_root / 'plots'}")


if __name__ == "__main__":
    main()
