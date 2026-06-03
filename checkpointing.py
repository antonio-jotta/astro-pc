from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from metrics import save_metrics_csv, save_metrics_pickle


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save_config_snapshot(run_dir: str | Path, cfg: Any) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / "config.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True, default=str)


def load_run_status(run_dir: str | Path) -> dict | None:
    run_dir = Path(run_dir)
    path = run_dir / "status.json"
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_run_status(
    run_dir: str | Path,
    *,
    run_name: str,
    cfg: Any,
    status: str,
    last_completed_epoch: int,
    elapsed_time_seconds: float,
    stop_reason: str | None = None,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_name": run_name,
        "method": cfg.method,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "status": status,
        "last_completed_epoch": int(last_completed_epoch),
        "elapsed_time_seconds": float(elapsed_time_seconds),
        "stop_reason": stop_reason,
        "updated_at": _now_iso(),
    }

    with (run_dir / "status.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _capture_rng_state() -> dict:
    payload = {
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        payload["cuda_random_state_all"] = torch.cuda.get_rng_state_all()

    return payload


def _coerce_rng_state_tensor(state) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().to(dtype=torch.uint8)
    return torch.tensor(state, dtype=torch.uint8)


def _restore_rng_state(payload: dict | None) -> None:
    if payload is None:
        return

    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])

    if "torch_random_state" in payload:
        torch_state = _coerce_rng_state_tensor(payload["torch_random_state"])
        torch.set_rng_state(torch_state)

    if torch.cuda.is_available() and "cuda_random_state_all" in payload:
        cuda_states = [
            _coerce_rng_state_tensor(state)
            for state in payload["cuda_random_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _collect_phago_state(model) -> dict | None:
    phago = getattr(model, "phago", None)
    if phago is None or len(phago) == 0:
        return None

    return {int(layer_idx): module.state_dict() for layer_idx, module in phago.items()}


def _restore_phago_state(model, phago_state: dict | None) -> None:
    if phago_state is None:
        return

    phago = getattr(model, "phago", None)
    if phago is None:
        return

    for layer_idx, state in phago_state.items():
        if layer_idx in phago:
            phago[layer_idx].load_state_dict(state)


def _torch_load(path: Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_training_checkpoint(
    run_dir: str | Path,
    *,
    epoch: int,
    elapsed_time_seconds: float,
    training_finished: bool,
    stop_reason: str | None,
    model,
    bp_optimizer,
    pc_state,
    early_stop_state,
    metrics_logging: list[dict],
    method_metrics: list[dict],
) -> Path:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "elapsed_time_seconds": float(elapsed_time_seconds),
        "training_finished": bool(training_finished),
        "stop_reason": stop_reason,
        "model_state_dict": model.state_dict(),
        "phago_state": _collect_phago_state(model),
        "bp_optimizer_state_dict": (
            None if bp_optimizer is None else bp_optimizer.state_dict()
        ),
        "pc_state": pc_state,
        "early_stop_state": early_stop_state,
        "metrics_logging": metrics_logging,
        "method_metrics": method_metrics,
        "rng_state": _capture_rng_state(),
        "saved_at": _now_iso(),
    }

    out_path = ckpt_dir / "checkpoint_last.pt"
    torch.save(payload, out_path)
    return out_path


def load_training_checkpoint(
    run_dir: str | Path,
    *,
    model,
    bp_optimizer,
    device: torch.device,
) -> dict | None:
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "checkpoints" / "checkpoint_last.pt"

    if not ckpt_path.exists():
        return None

    checkpoint = _torch_load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    _restore_phago_state(model, checkpoint.get("phago_state"))

    opt_state = checkpoint.get("bp_optimizer_state_dict")
    if bp_optimizer is not None and opt_state is not None:
        bp_optimizer.load_state_dict(opt_state)

    _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint


def save_progress_artifacts(
    run_dir: str | Path,
    *,
    metrics_logging: list[dict],
    method_metrics: list[dict],
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if metrics_logging:
        save_metrics_csv(metrics_logging, out_path=str(run_dir / "metrics.csv"))
        save_metrics_pickle(metrics_logging, out_path=str(run_dir / "metrics.pkl"))

    save_metrics_pickle(method_metrics, out_path=str(run_dir / "method_metrics.pkl"))


def metric_curves_from_history(
    metrics_logging: list[dict],
) -> tuple[list[float], list[float], list[float], list[float]]:
    train_losses = [float(rec["train_loss"]) for rec in metrics_logging]
    train_accs = [float(rec["train_acc"]) for rec in metrics_logging]
    test_losses = [float(rec["test_loss"]) for rec in metrics_logging]
    test_accs = [float(rec["test_acc"]) for rec in metrics_logging]
    return train_losses, train_accs, test_losses, test_accs
