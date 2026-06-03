from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EarlyStoppingState:
    best_metric: float | None = None
    metric_bad_epochs: int = 0
    weight_bad_epochs: int = 0


def _metric_improved(
    *,
    mode: str,
    current: float,
    best: float | None,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if mode == "acc":
        return current > best + min_delta
    if mode == "loss":
        return current < best - min_delta
    raise ValueError(f"Unknown early_stop_metric: {mode}")


def update_early_stopping(
    state: EarlyStoppingState,
    cfg: Any,
    *,
    epoch: int,
    test_loss: float,
    test_acc: float,
    epoch_mean_dW: float | None,
) -> tuple[EarlyStoppingState, bool, str | None]:
    """Update early-stopping counters and return the stop decision."""
    if not getattr(cfg, "early_stop_enabled", False):
        return state, False, None

    if epoch <= getattr(cfg, "early_stop_warmup_epochs", 0):
        return state, False, None

    stop_messages: list[str] = []

    if cfg.early_stop_mode in {"metric", "both"}:
        current = float(test_acc if cfg.early_stop_metric == "acc" else test_loss)
        if _metric_improved(
            mode=cfg.early_stop_metric,
            current=current,
            best=state.best_metric,
            min_delta=cfg.early_stop_min_delta,
        ):
            state.best_metric = current
            state.metric_bad_epochs = 0
        else:
            state.metric_bad_epochs += 1

        if state.metric_bad_epochs >= cfg.early_stop_patience:
            stop_messages.append(
                f"early_stopping_metric: no {cfg.early_stop_metric} improvement "
                f"for {state.metric_bad_epochs} epochs"
            )

    if cfg.early_stop_mode in {"weight", "both"} and epoch_mean_dW is not None:
        if float(epoch_mean_dW) <= cfg.early_stop_weight_threshold:
            state.weight_bad_epochs += 1
        else:
            state.weight_bad_epochs = 0

        if state.weight_bad_epochs >= cfg.early_stop_weight_patience:
            stop_messages.append(
                f"early_stopping_weight: mean dW RMS <= "
                f"{cfg.early_stop_weight_threshold:g} for "
                f"{state.weight_bad_epochs} epochs"
            )

    if stop_messages:
        return state, True, "; ".join(stop_messages)

    return state, False, None
