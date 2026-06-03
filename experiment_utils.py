import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from config import Config


def make_cfg(base: Config, **overrides: Any) -> Config:
    return replace(base, **overrides)


def build_run_name(cfg: Config, prefix: str | None = None) -> str:
    parts: list[str] = []

    if prefix:
        parts.append(prefix)

    parts.append(f"{cfg.method}")
    parts.append(f"{cfg.dataset}")
    parts.append(f"seed{cfg.seed}")

    if cfg.method == "pc":
        parts.append(f"K{cfg.pc_frac_memory_len}")
        parts.append(f"rho{cfg.pc_frac_rho:g}")

        ctrl_tag = "on" if cfg.astro_controller_enabled else "off"
        leak_tag = "on" if cfg.astro_controller_leak_enabled else "off"
        phago_tag = "on" if cfg.phago_enabled else "off"

        parts.append(f"ctrl-{ctrl_tag}")
        if cfg.astro_controller_enabled:
            parts.append(f"dyn-{cfg.astro_controller_dynamics}")
            parts.append(f"period{cfg.astro_controller_update_period_steps}")
            if getattr(cfg, "astro_controller_state_coupling_enabled", False):
                parts.append(
                    f"statecouple{cfg.astro_controller_state_coupling_strength:g}"
                )
        parts.append(f"leak-{leak_tag}")
        parts.append(f"phago-{phago_tag}")

    return "_".join(parts)


def build_unique_name(base_name: str, root_dir: str | Path) -> str:
    root = Path(root_dir)

    candidate = base_name
    if not (root / candidate).exists():
        return candidate

    run_idx = 1
    while True:
        candidate = f"{base_name}__run{run_idx}"
        if not (root / candidate).exists():
            return candidate
        run_idx += 1


def _is_bad_number(x: Any) -> bool:
    if x is None:
        return False

    try:
        x = float(x)
    except (TypeError, ValueError):
        return False

    return math.isnan(x) or math.isinf(x)


def _should_stop_for_numerical_failure(
    *,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
    epoch_mean_dW: float | None,
) -> tuple[bool, str | None]:
    bad_fields = []

    if _is_bad_number(train_loss):
        bad_fields.append("train_loss")
    if _is_bad_number(train_acc):
        bad_fields.append("train_acc")
    if _is_bad_number(test_loss):
        bad_fields.append("test_loss")
    if _is_bad_number(test_acc):
        bad_fields.append("test_acc")
    if _is_bad_number(epoch_mean_dW):
        bad_fields.append("epoch_mean_dW_rms")

    if not bad_fields:
        return False, None

    reason = "Numerical failure detected: " + ", ".join(bad_fields)
    return True, reason


def _last_finite_value(values: list[float], fallback: float) -> float:
    for value in reversed(values):
        if not _is_bad_number(value):
            return float(value)
    return float(fallback)


def _pad_metrics_after_failure(
    *,
    metrics_logging: list[dict],
    start_epoch: int,
    final_epoch: int,
    train_losses: list[float],
    train_accs: list[float],
    test_losses: list[float],
    test_accs: list[float],
    method: str,
    dataset: str,
    seed: int,
    cfg: Config,
    failed_test_acc: float,
    failed_train_acc: float,
    failed_test_loss: float,
    failed_train_loss: float,
    epoch_mean_dW: float | None,
    failure_reason: str,
) -> None:
    """Pad metric histories after a numerical failure."""
    if start_epoch > final_epoch:
        return

    safe_train_loss = (
        _last_finite_value(train_losses, fallback=failed_train_loss)
        if _is_bad_number(failed_train_loss)
        else float(failed_train_loss)
    )
    safe_test_loss = (
        _last_finite_value(test_losses, fallback=failed_test_loss)
        if _is_bad_number(failed_test_loss)
        else float(failed_test_loss)
    )

    safe_train_acc = (
        0.0 if _is_bad_number(failed_train_acc) else float(failed_train_acc)
    )
    safe_test_acc = 0.0 if _is_bad_number(failed_test_acc) else float(failed_test_acc)

    for epoch in range(start_epoch, final_epoch + 1):
        train_losses.append(safe_train_loss)
        train_accs.append(safe_train_acc)
        test_losses.append(safe_test_loss)
        test_accs.append(safe_test_acc)

        epoch_record = {
            "epoch": epoch,
            "train_loss": safe_train_loss,
            "train_acc": safe_train_acc,
            "test_loss": safe_test_loss,
            "test_acc": safe_test_acc,
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "failed": 1,
            "imputed_after_failure": 1,
            "failure_reason": failure_reason,
        }

        if method == "pc":
            epoch_record.update(
                {
                    "pc_frac_memory_len": cfg.pc_frac_memory_len,
                    "pc_frac_rho": cfg.pc_frac_rho,
                    "astro_controller_enabled": cfg.astro_controller_enabled,
                    "astro_controller_update_period_steps": cfg.astro_controller_update_period_steps,
                    "phago_enabled": int(cfg.phago_enabled),
                    "epoch_mean_dW_rms": (
                        None if _is_bad_number(epoch_mean_dW) else epoch_mean_dW
                    ),
                }
            )

        metrics_logging.append(epoch_record)
