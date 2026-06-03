from __future__ import annotations

from dataclasses import replace

import torch

from controller import AstroCState
from controller_configs import AstroCConfig


def _stack_scalar_gains(gains: list[torch.Tensor]) -> torch.Tensor:
    if len(gains) == 0:
        raise ValueError("Cannot stack an empty gain list.")
    return torch.stack([g.reshape(()) for g in gains], dim=0)


def update_learned_leak_baselines(
    astro_state: AstroCState,
    base_gains: list[torch.Tensor],
    cfg: AstroCConfig,
    controller_updated: bool,
) -> AstroCState:
    """Update per-layer leakage baselines from early controller updates."""
    if len(base_gains) == 0:
        return astro_state
    if cfg.leak_baseline is not None:
        return astro_state
    if not controller_updated:
        return astro_state
    if cfg.leak_baseline_updates <= 0:
        raise ValueError("cfg.leak_baseline_updates must be positive.")
    if len(base_gains) != len(astro_state.layers):
        raise ValueError("base_gains must contain one tensor per controller layer.")

    new_layers = []
    for layer_state, gain in zip(astro_state.layers, base_gains):
        if layer_state.leak_baseline_frozen:
            new_layers.append(layer_state)
            continue

        gain_detached = gain.detach().clone().reshape(())
        if layer_state.leak_baseline is None:
            updated_baseline = gain_detached
        else:
            updated_baseline = torch.minimum(
                layer_state.leak_baseline.reshape(()),
                gain_detached,
            )

        updated_count = layer_state.leak_baseline_count + 1
        baseline_frozen = updated_count >= cfg.leak_baseline_updates

        new_layers.append(
            replace(
                layer_state,
                leak_baseline=updated_baseline,
                leak_baseline_count=updated_count,
                leak_baseline_frozen=baseline_frozen,
            )
        )

    return AstroCState(kernel=astro_state.kernel, layers=new_layers)


def apply_astrocytic_gain_leakage(
    base_gains: list[torch.Tensor],
    astro_state: AstroCState,
    cfg: AstroCConfig,
) -> tuple[list[torch.Tensor], dict]:
    """Apply gain leakage between neighboring layers."""
    if len(base_gains) == 0:
        return [], {}

    gains_t = _stack_scalar_gains(base_gains)
    num_layers = gains_t.shape[0]

    baselines_t = torch.stack(
        [
            (
                layer.leak_baseline.reshape(())
                if layer.leak_baseline is not None
                else torch.zeros_like(gains_t[i])
            )
            for i, layer in enumerate(astro_state.layers)
        ],
        dim=0,
    )
    baselines_frozen = all(layer.leak_baseline_frozen for layer in astro_state.layers)
    leakage_active = cfg.leak_enabled and baselines_frozen and cfg.leak_radius > 0

    if not leakage_active:
        return [g.clone() for g in base_gains], {
            "astro_controller_leak_active": 0.0,
            "astro_controller_leak_baselines": [float(x.item()) for x in baselines_t],
            "astro_controller_leak_baseline_mean": float(baselines_t.mean().item()),
            "astro_controller_leak_baseline_frozen": float(baselines_frozen),
            "astro_controller_num_active_sources": 0,
            "astro_controller_num_receiving_layers": 0,
            "astro_controller_num_active_edges": 0,
            "astro_controller_leak_increment_mean": 0.0,
            "astro_controller_leak_increment_max": 0.0,
            "astro_controller_leak_increment_sum": 0.0,
            "astro_controller_source_excess_mean": 0.0,
        }

    leak_increments = torch.zeros_like(gains_t)
    global_excess = (gains_t - baselines_t).clamp_min(0.0)
    num_active_sources = int((global_excess > 0.0).sum().item())

    num_active_edges = 0
    edge_excesses: list[torch.Tensor] = []

    for i in range(num_layers):
        neighbor_indices: list[int] = []
        neighbor_weights: list[float] = []

        left = max(0, i - cfg.leak_radius)
        right = min(num_layers - 1, i + cfg.leak_radius)
        for j in range(left, right + 1):
            if j == i:
                continue
            dist = abs(j - i)
            neighbor_indices.append(j)
            neighbor_weights.append(1.0 / float(dist))

        if not neighbor_indices:
            continue

        weights = torch.tensor(
            neighbor_weights,
            device=gains_t.device,
            dtype=gains_t.dtype,
        )
        weights = weights / weights.sum().clamp_min(1e-12)
        neighbor_excess = global_excess[neighbor_indices]

        num_active_edges += int((neighbor_excess > 0.0).sum().item())
        edge_excesses.append(neighbor_excess)
        leak_increments[i] = torch.sum(weights * neighbor_excess)

    effective_gains = gains_t + leak_increments
    num_receiving_layers = int((leak_increments > 0.0).sum().item())

    if edge_excesses:
        edge_excesses_t = torch.cat(edge_excesses)
        source_excess_mean = float(edge_excesses_t.mean().item())
    else:
        source_excess_mean = 0.0

    leak_stats = {
        "astro_controller_leak_active": 1.0,
        "astro_controller_leak_baselines": [float(x.item()) for x in baselines_t],
        "astro_controller_leak_baseline_mean": float(baselines_t.mean().item()),
        "astro_controller_leak_baseline_frozen": float(baselines_frozen),
        "astro_controller_num_active_sources": num_active_sources,
        "astro_controller_num_receiving_layers": num_receiving_layers,
        "astro_controller_num_active_edges": num_active_edges,
        "astro_controller_leak_increment_mean": float(leak_increments.mean().item()),
        "astro_controller_leak_increment_max": float(leak_increments.max().item()),
        "astro_controller_leak_increment_sum": float(leak_increments.sum().item()),
        "astro_controller_source_excess_mean": source_excess_mean,
    }

    return [effective_gains[i].clone() for i in range(num_layers)], leak_stats
