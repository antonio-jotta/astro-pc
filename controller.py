from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn

from controller_configs import AstroCConfig, PCConfig


@dataclass
class AstroLayerCState:
    """Controller state for one trainable layer."""

    astro_state: torch.Tensor
    gain: torch.Tensor
    gain_history: deque

    last_mismatch_signal: torch.Tensor
    last_state_input: torch.Tensor
    last_state_coupling: torch.Tensor

    leak_baseline: torch.Tensor | None
    leak_baseline_count: int
    leak_baseline_frozen: bool


@dataclass
class AstroCState:
    """Container for layer-local controller states."""

    kernel: torch.Tensor
    layers: list[AstroLayerCState]


def _build_power_law_kernel(
    memory_len: int,
    rho: float,
    normalize: bool,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ks = torch.arange(memory_len + 1, device=device, dtype=dtype)
    kernel = (ks + 1.0).pow(-rho)
    if normalize:
        kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


def _stack_scalar_tensors(values: list[torch.Tensor]) -> torch.Tensor:
    if len(values) == 0:
        raise ValueError("Cannot stack an empty tensor list.")
    return torch.stack([v.reshape(()) for v in values], dim=0)


def _make_scalar_like(reference: torch.Tensor, value: float) -> torch.Tensor:
    return torch.tensor(float(value), device=reference.device, dtype=reference.dtype)


def _gain_from_astro_state(
    astro_state: torch.Tensor,
    cfg: AstroCConfig,
) -> torch.Tensor:
    """Map latent state to a multiplicative gain."""
    return 1.0 + cfg.gain_scale * torch.tanh(astro_state)


def initialize_astro_controller_state(
    model: nn.Module,
    cfg: PCConfig,
) -> AstroCState:
    """Initialize one controller state per trainable layer."""
    ref_param = next(model.parameters())
    device = ref_param.device
    dtype = ref_param.dtype

    kernel = _build_power_law_kernel(
        memory_len=cfg.frac_memory_len,
        rho=cfg.frac_rho,
        normalize=cfg.frac_normalize,
        device=device,
        dtype=dtype,
    )

    layers: list[AstroLayerCState] = []
    for _ in model.layers:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        astro_state = zero.clone()
        gain = _gain_from_astro_state(astro_state, cfg.astro_controller)

        gain_history = deque(maxlen=cfg.frac_memory_len + 1)
        gain_history.appendleft(gain.detach().clone())

        if cfg.astro_controller.leak_baseline is None:
            leak_baseline = None
            leak_baseline_count = 0
            leak_baseline_frozen = False
        else:
            fixed_baseline = torch.tensor(
                float(cfg.astro_controller.leak_baseline),
                device=device,
                dtype=dtype,
            )
            leak_baseline = fixed_baseline.clone()
            leak_baseline_count = cfg.astro_controller.leak_baseline_updates
            leak_baseline_frozen = True

        layers.append(
            AstroLayerCState(
                astro_state=astro_state,
                gain=gain.detach().clone(),
                gain_history=gain_history,
                last_mismatch_signal=zero.clone(),
                last_state_input=zero.clone(),
                last_state_coupling=zero.clone(),
                leak_baseline=leak_baseline,
                leak_baseline_count=leak_baseline_count,
                leak_baseline_frozen=leak_baseline_frozen,
            )
        )

    return AstroCState(kernel=kernel, layers=layers)


def extract_local_layer_signals_for_controller(
    x_states: list[torch.Tensor],
    eps: list[torch.Tensor],
    weight_drives: list[torch.Tensor] | None = None,
    bias_drives: list[torch.Tensor] | None = None,
) -> list[dict]:
    """Extract residual and activity signals for each trainable layer."""
    del weight_drives, bias_drives

    layer_signals: list[dict] = []
    num_layers = len(x_states) - 1

    for layer in range(1, num_layers + 1):
        x_l = x_states[layer].detach()
        eps_l = eps[layer].detach()

        residual_energy = torch.mean(eps_l * eps_l)
        activity_energy = torch.mean(x_l * x_l)
        mismatch = residual_energy / (residual_energy + activity_energy + 1e-12)

        layer_signals.append(
            {
                "x": x_l,
                "eps": eps_l,
                "residual_energy": residual_energy,
                "activity_energy": activity_energy,
                "mismatch_signal": mismatch,
            }
        )

    return layer_signals


def should_update_astro_controller(
    total_neural_actions: int,
    pending_neural_actions: int,
    cfg: AstroCConfig,
) -> bool:
    """Return whether the slow controller should update."""
    if not cfg.enabled:
        return False
    if total_neural_actions <= cfg.warmup_steps:
        return False
    if cfg.update_period_steps <= 0:
        return True
    return pending_neural_actions >= cfg.update_period_steps


def _compute_state_coupling_terms(
    astro_state: AstroCState,
    cfg: AstroCConfig,
) -> list[torch.Tensor]:
    """Compute neighbor coupling terms between controller states."""
    layers = astro_state.layers
    if len(layers) == 0:
        return []

    zeros = [torch.zeros_like(layer.astro_state) for layer in layers]
    if (
        not cfg.state_coupling_enabled
        or cfg.state_coupling_radius <= 0
        or cfg.state_coupling_strength == 0.0
    ):
        return zeros

    num_layers = len(layers)
    state_values = _stack_scalar_tensors([layer.astro_state for layer in layers])
    state_values = torch.tanh(state_values)

    terms: list[torch.Tensor] = []
    for i in range(num_layers):
        neighbor_indices: list[int] = []
        neighbor_weights: list[float] = []

        left = max(0, i - cfg.state_coupling_radius)
        right = min(num_layers - 1, i + cfg.state_coupling_radius)

        for j in range(left, right + 1):
            if j == i:
                continue
            dist = abs(j - i)
            neighbor_indices.append(j)
            neighbor_weights.append(1.0 / float(dist))

        if len(neighbor_indices) == 0:
            terms.append(torch.zeros_like(layers[i].astro_state))
            continue

        weights = torch.tensor(
            neighbor_weights,
            device=state_values.device,
            dtype=state_values.dtype,
        )
        weights = weights / weights.sum().clamp_min(1e-12)
        neighbor_state = state_values[neighbor_indices]
        coupling = cfg.state_coupling_strength * torch.sum(weights * neighbor_state)
        terms.append(coupling.reshape_as(layers[i].astro_state))

    return terms


def update_layer_astro_controller_state(
    layer_state: AstroLayerCState,
    layer_signals: dict,
    kernel: torch.Tensor,
    cfg: AstroCConfig,
    state_coupling_term: torch.Tensor | None = None,
) -> AstroLayerCState:
    """Update one layer-local controller state."""
    del kernel

    mismatch_signal = layer_signals["mismatch_signal"].to(
        device=layer_state.astro_state.device,
        dtype=layer_state.astro_state.dtype,
    )

    if state_coupling_term is None:
        state_coupling_term = torch.zeros_like(layer_state.astro_state)
    else:
        state_coupling_term = state_coupling_term.to(
            device=layer_state.astro_state.device,
            dtype=layer_state.astro_state.dtype,
        )

    if cfg.dynamics == "astro_state":
        state_input = mismatch_signal
    elif cfg.dynamics == "passive_slow_gain":
        state_input = _make_scalar_like(
            layer_state.astro_state, cfg.passive_state_drive
        )
    elif cfg.dynamics == "legacy_gain":
        state_input = mismatch_signal
    else:
        raise ValueError(
            "Unknown astro-controller dynamics "
            f"'{cfg.dynamics}'. Expected 'astro_state', 'passive_slow_gain', or 'legacy_gain'."
        )

    if cfg.dynamics == "legacy_gain":
        new_astro_state = layer_state.astro_state.detach().clone()
        desired_gain = 1.0 + cfg.gain_scale * torch.tanh(state_input)
    else:
        target = state_input + state_coupling_term
        new_astro_state = (
            1.0 - cfg.state_smoothing
        ) * layer_state.astro_state + cfg.state_smoothing * target
        desired_gain = _gain_from_astro_state(new_astro_state, cfg)

    new_gain_history = deque(
        layer_state.gain_history,
        maxlen=layer_state.gain_history.maxlen,
    )
    new_gain_history.appendleft(desired_gain.detach().clone())

    return AstroLayerCState(
        astro_state=new_astro_state.detach().clone(),
        gain=desired_gain.detach().clone(),
        gain_history=new_gain_history,
        last_mismatch_signal=mismatch_signal.detach().clone(),
        last_state_input=state_input.detach().clone(),
        last_state_coupling=state_coupling_term.detach().clone(),
        leak_baseline=layer_state.leak_baseline,
        leak_baseline_count=layer_state.leak_baseline_count,
        leak_baseline_frozen=layer_state.leak_baseline_frozen,
    )


def recompute_all_astro_controller_states(
    astro_state: AstroCState,
    layer_signals: list[dict],
    cfg: AstroCConfig,
) -> AstroCState:
    """Recompute all layer-local controller states."""
    if len(astro_state.layers) != len(layer_signals):
        raise ValueError(
            "Mismatch between number of controller layers and extracted layer signals."
        )

    coupling_terms = _compute_state_coupling_terms(astro_state, cfg)

    new_layers: list[AstroLayerCState] = []
    for layer_state, signals, coupling_term in zip(
        astro_state.layers,
        layer_signals,
        coupling_terms,
    ):
        new_layers.append(
            update_layer_astro_controller_state(
                layer_state=layer_state,
                layer_signals=signals,
                kernel=astro_state.kernel,
                cfg=cfg,
                state_coupling_term=coupling_term,
            )
        )

    return AstroCState(kernel=astro_state.kernel, layers=new_layers)


def get_layer_plasticity_gains(astro_state: AstroCState) -> list[torch.Tensor]:
    """Return the current scalar gain for each trainable layer."""
    return [layer_state.gain for layer_state in astro_state.layers]


def apply_astro_controller_gain_modulation_to_weight_updates(
    base_weight_drives: list[torch.Tensor],
    base_bias_drives: list[torch.Tensor],
    gains: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Multiply each layer's already-computed plasticity drive by its gain."""
    if not (len(base_weight_drives) == len(base_bias_drives) == len(gains)):
        raise ValueError(
            "Mismatch between number of weight drives, bias drives, and controller gains."
        )

    mod_weight_drives: list[torch.Tensor] = []
    mod_bias_drives: list[torch.Tensor] = []
    for dW, db, g in zip(base_weight_drives, base_bias_drives, gains):
        mod_weight_drives.append(g * dW)
        mod_bias_drives.append(g * db)
    return mod_weight_drives, mod_bias_drives


def collect_astro_controller_metrics(
    astro_state: AstroCState,
    base_gains: list[torch.Tensor],
    effective_gains: list[torch.Tensor],
    leak_stats: dict,
) -> dict:
    """Collect compact diagnostics for the controller stage."""
    if len(base_gains) == 0:
        return {}

    base_gain_values = _stack_scalar_tensors(base_gains)
    eff_gain_values = _stack_scalar_tensors(effective_gains)
    state_values = _stack_scalar_tensors(
        [layer_state.astro_state for layer_state in astro_state.layers]
    )
    mismatch_values = _stack_scalar_tensors(
        [layer_state.last_mismatch_signal for layer_state in astro_state.layers]
    )
    state_input_values = _stack_scalar_tensors(
        [layer_state.last_state_input for layer_state in astro_state.layers]
    )
    coupling_values = _stack_scalar_tensors(
        [layer_state.last_state_coupling for layer_state in astro_state.layers]
    )

    return {
        "astro_controller_base_gains": [float(g.item()) for g in base_gain_values],
        "astro_controller_effective_gains": [float(g.item()) for g in eff_gain_values],
        "astro_controller_gain_mean": float(eff_gain_values.mean().item()),
        "astro_controller_base_gain_mean": float(base_gain_values.mean().item()),
        "astro_controller_state_values": [float(a.item()) for a in state_values],
        "astro_controller_state_mean": float(state_values.mean().item()),
        "astro_controller_state_abs_mean": float(state_values.abs().mean().item()),
        "astro_controller_state_max_abs": float(state_values.abs().max().item()),
        "astro_controller_mismatch_signal_mean": float(mismatch_values.mean().item()),
        "astro_controller_state_input_mean": float(state_input_values.mean().item()),
        "astro_controller_state_coupling_mean": float(coupling_values.mean().item()),
        "astro_controller_state_coupling_abs_mean": float(
            coupling_values.abs().mean().item()
        ),
        "astro_controller_state_coupling_max_abs": float(
            coupling_values.abs().max().item()
        ),
        **leak_stats,
    }
