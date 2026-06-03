from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from astro_leakage import (
    apply_astrocytic_gain_leakage,
    update_learned_leak_baselines,
)
from astro_phagocytosis import (
    get_effective_weight,
    use_phagocytosis,
)
from controller import (
    AstroCState,
    apply_astro_controller_gain_modulation_to_weight_updates,
    collect_astro_controller_metrics,
    extract_local_layer_signals_for_controller,
    get_layer_plasticity_gains,
    initialize_astro_controller_state,
    recompute_all_astro_controller_states,
    should_update_astro_controller,
)
from controller_configs import PCConfig


@dataclass
class FractionalLayerMemory:
    weight_history: deque
    bias_history: deque


@dataclass
class FractionalPlasticityState:
    kernel: torch.Tensor
    layers: list[FractionalLayerMemory]


@dataclass
class PCTrainState:
    plasticity_state: FractionalPlasticityState
    astro_controller_state: AstroCState | None
    total_neural_actions: int = 0
    pending_neural_actions: int = 0


def _build_fractional_kernel(
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


def _aggregate_fractional_history(
    history: deque,
    kernel: torch.Tensor,
) -> torch.Tensor:
    if len(history) == 0:
        raise ValueError("Empty fractional history.")

    coeffs = kernel[: len(history)]
    out = torch.zeros_like(history[0])

    for ck, item in zip(coeffs, history):
        out = out + ck * item

    return out


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x * x) + 1e-12).item())


def _safe_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    denom = (a_flat.norm(p=2) * b_flat.norm(p=2)).clamp_min(1e-12)
    return float(torch.dot(a_flat, b_flat).item() / denom.item())


def _phi(v: torch.Tensor, activation_function: str) -> torch.Tensor:
    if activation_function == "tanh":
        return torch.tanh(v)
    elif activation_function == "relu":
        return F.relu(v)
    elif activation_function == "sigmoid":
        return torch.sigmoid(v)
    elif activation_function == "identity":
        return v
    else:
        raise ValueError(f"Unsupported activation_function: {activation_function}")


def _phi_prime_from_v(v: torch.Tensor, activation_function: str) -> torch.Tensor:
    if activation_function == "tanh":
        r = torch.tanh(v)
        return 1.0 - r * r
    elif activation_function == "relu":
        return (v > 0).to(v.dtype)
    elif activation_function == "sigmoid":
        s = torch.sigmoid(v)
        return s * (1.0 - s)
    elif activation_function == "identity":
        return torch.ones_like(v)
    else:
        raise ValueError(f"Unsupported activation_function: {activation_function}")


def initialize_fractional_plasticity_state(
    model: nn.Module,
    cfg: PCConfig,
) -> FractionalPlasticityState:
    ref_param = next(model.parameters())
    kernel = _build_fractional_kernel(
        memory_len=cfg.frac_memory_len,
        rho=cfg.frac_rho,
        normalize=cfg.frac_normalize,
        device=ref_param.device,
        dtype=ref_param.dtype,
    )

    layers: list[FractionalLayerMemory] = []
    for _ in model.layers:
        layers.append(
            FractionalLayerMemory(
                weight_history=deque(maxlen=cfg.frac_memory_len + 1),
                bias_history=deque(maxlen=cfg.frac_memory_len + 1),
            )
        )

    return FractionalPlasticityState(kernel=kernel, layers=layers)


def initialize_pc_train_state(
    model: nn.Module,
    cfg: PCConfig,
) -> PCTrainState:
    plasticity_state = initialize_fractional_plasticity_state(
        model=model,
        cfg=cfg,
    )

    astro_controller_state = None
    if cfg.astro_controller.enabled:
        astro_controller_state = initialize_astro_controller_state(
            model=model,
            cfg=cfg,
        )

    return PCTrainState(
        plasticity_state=plasticity_state,
        astro_controller_state=astro_controller_state,
        total_neural_actions=0,
        pending_neural_actions=0,
    )


def compute_pc_output_target(
    logits_free: torch.Tensor,
    y: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return the cross-entropy nudged output target."""
    num_classes = logits_free.shape[1]
    target = F.one_hot(y, num_classes=num_classes).to(logits_free.dtype)
    probs = torch.softmax(logits_free, dim=1)
    ce_grad = probs - target
    return logits_free - beta * ce_grad


def _forward_predictions_from_states(
    model: nn.Module,
    x_states: list[torch.Tensor],
    activation_function: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute layer predictions and residuals from the current states."""
    num_layers = len(model.layers)
    mu: list[torch.Tensor] = [None] * (num_layers + 1)  # type: ignore
    eps: list[torch.Tensor] = [None] * (num_layers + 1)  # type: ignore

    for layer in range(1, num_layers + 1):
        W = get_effective_weight(model, layer - 1)
        b = model.layers[layer - 1].bias

        if layer == num_layers:
            mu_l = x_states[layer - 1] @ W.t() + b
        else:
            v_l = x_states[layer - 1] @ W.t() + b
            mu_l = _phi(v_l, activation_function)

        mu[layer] = mu_l
        eps[layer] = x_states[layer] - mu_l

    return mu, eps


def _total_pc_energy(eps: list[torch.Tensor]) -> torch.Tensor:
    """Compute the total predictive-coding energy."""
    total = torch.tensor(0.0, device=eps[1].device, dtype=eps[1].dtype)
    for e in eps[1:]:
        total = total + 0.5 * torch.sum(e * e)
    return total


def run_pc_inference(
    model: nn.Module,
    x_init: list[torch.Tensor],
    output_target: torch.Tensor,
    cfg: PCConfig,
) -> dict:
    """Run predictive-coding inference over hidden states."""
    x_states = [x.detach().clone() for x in x_init]
    x_states[0] = x_init[0].detach()
    x_states[-1] = output_target.detach()

    num_layers = len(model.layers)
    energy_trace: list[float] = []
    state_update_trace: list[float] = []

    for _ in range(cfg.infer_steps):
        mu, eps = _forward_predictions_from_states(
            model=model,
            x_states=x_states,
            activation_function=cfg.activation_function,
        )

        delta_norm_accum = 0.0

        for layer in range(1, num_layers):
            e_l = eps[layer]

            W_next = get_effective_weight(model, layer)
            b_next = model.layers[layer].bias

            if layer == num_layers - 1:
                topdown_term = eps[layer + 1] @ W_next
            else:
                v_next = x_states[layer] @ W_next.t() + b_next
                phi_prime_next = _phi_prime_from_v(v_next, cfg.activation_function)
                topdown_term = (eps[layer + 1] * phi_prime_next) @ W_next

            dx = -e_l + topdown_term
            x_states[layer] = x_states[layer] + cfg.state_lr * dx

            delta_norm_accum += float(
                torch.sqrt(torch.mean((cfg.state_lr * dx) ** 2) + 1e-12).item()
            )

        _, eps_new = _forward_predictions_from_states(
            model=model,
            x_states=x_states,
            activation_function=cfg.activation_function,
        )

        energy_trace.append(float(_total_pc_energy(eps_new).item()))
        state_update_trace.append(delta_norm_accum)

    mu_final, eps_final = _forward_predictions_from_states(
        model=model,
        x_states=x_states,
        activation_function=cfg.activation_function,
    )

    return {
        "x_states": x_states,
        "mu": mu_final,
        "eps": eps_final,
        "energy_trace": energy_trace,
        "state_update_trace": state_update_trace,
    }


@torch.no_grad()
def compute_pc_instantaneous_weight_drives(
    model: nn.Module,
    x_states: list[torch.Tensor],
    eps: list[torch.Tensor],
    cfg: PCConfig,
) -> dict:
    """Compute instantaneous local PC drives for each trainable layer."""
    num_layers = len(model.layers)

    inst_weight_drives: list[torch.Tensor] = []
    inst_bias_drives: list[torch.Tensor] = []

    G_rms_layers: list[float] = []
    gb_rms_layers: list[float] = []

    for layer in range(1, num_layers + 1):
        W = get_effective_weight(model, layer - 1)
        b = model.layers[layer - 1].bias

        pre = x_states[layer - 1]
        e = eps[layer]

        if layer == num_layers:
            local_factor = e
        else:
            v_l = pre @ W.t() + b
            phi_prime_l = _phi_prime_from_v(v_l, cfg.activation_function)
            local_factor = e * phi_prime_l

        G_t = local_factor.t() @ pre / max(1, pre.shape[0])
        gb_t = local_factor.mean(dim=0)

        if cfg.grad_clip is not None:
            G_t = G_t.clamp(min=-cfg.grad_clip, max=cfg.grad_clip)
            gb_t = gb_t.clamp(min=-cfg.grad_clip, max=cfg.grad_clip)

        inst_weight_drives.append(G_t)
        inst_bias_drives.append(gb_t)

        G_rms_layers.append(_rms(G_t))
        gb_rms_layers.append(_rms(gb_t))

    return {
        "inst_weight_drives": inst_weight_drives,
        "inst_bias_drives": inst_bias_drives,
        "pc_G_rms_layers": G_rms_layers,
        "pc_gb_rms_layers": gb_rms_layers,
    }


@torch.no_grad()
def aggregate_fractional_weight_drives(
    instantaneous_drives: dict,
    plasticity_state: FractionalPlasticityState,
) -> dict:
    """Aggregate local drives with fractional-memory state."""
    inst_weight_drives = instantaneous_drives["inst_weight_drives"]
    inst_bias_drives = instantaneous_drives["inst_bias_drives"]

    agg_weight_drives: list[torch.Tensor] = []
    agg_bias_drives: list[torch.Tensor] = []

    M_rms_layers: list[float] = []
    MG_ratio_layers: list[float] = []
    cos_GM_layers: list[float] = []
    mb_rms_layers: list[float] = []

    for layer_idx, (G_t, gb_t) in enumerate(zip(inst_weight_drives, inst_bias_drives)):
        layer_mem = plasticity_state.layers[layer_idx]

        layer_mem.weight_history.appendleft(G_t.detach().clone())
        layer_mem.bias_history.appendleft(gb_t.detach().clone())

        M_t = _aggregate_fractional_history(
            history=layer_mem.weight_history,
            kernel=plasticity_state.kernel,
        )
        mb_t = _aggregate_fractional_history(
            history=layer_mem.bias_history,
            kernel=plasticity_state.kernel,
        )

        G_rms = _rms(G_t)
        M_rms = _rms(M_t)
        ratio = M_rms / max(G_rms, 1e-12)
        cos_gm = _safe_cosine_similarity(G_t, M_t)

        agg_weight_drives.append(M_t)
        agg_bias_drives.append(mb_t)

        M_rms_layers.append(M_rms)
        MG_ratio_layers.append(ratio)
        cos_GM_layers.append(cos_gm)
        mb_rms_layers.append(_rms(mb_t))

    return {
        "base_weight_drives": agg_weight_drives,
        "base_bias_drives": agg_bias_drives,
        "pc_M_rms_layers": M_rms_layers,
        "pc_M_over_G_layers": MG_ratio_layers,
        "pc_cos_G_M_layers": cos_GM_layers,
        "pc_mb_rms_layers": mb_rms_layers,
    }


@torch.no_grad()
def compute_pc_fractional_weight_drives(
    model: nn.Module,
    x_states: list[torch.Tensor],
    eps: list[torch.Tensor],
    cfg: PCConfig,
    plasticity_state: FractionalPlasticityState,
) -> dict:
    """Compute instantaneous and fractional-memory PC drives."""
    instantaneous = compute_pc_instantaneous_weight_drives(
        model=model,
        x_states=x_states,
        eps=eps,
        cfg=cfg,
    )

    aggregated = aggregate_fractional_weight_drives(
        instantaneous_drives=instantaneous,
        plasticity_state=plasticity_state,
    )

    return {
        **aggregated,
        "pc_G_rms_layers": instantaneous["pc_G_rms_layers"],
        "pc_gb_rms_layers": instantaneous["pc_gb_rms_layers"],
    }


@torch.no_grad()
def apply_pc_weight_drives(
    model: nn.Module,
    weight_drives: list[torch.Tensor],
    bias_drives: list[torch.Tensor],
    cfg: PCConfig,
    phago_permissive_signals: list[torch.Tensor] | None = None,
) -> dict:
    """Apply local PC drives to model parameters."""
    dW_rms_layers: list[float] = []
    weight_rms_layers: list[float] = []
    applied_weight_delta_rms_layers: list[float] = []

    phago_active = use_phagocytosis(model, cfg)

    phago_metrics: dict[str, list[float]] = {
        "phago_did_prune_layers": [],
        "phago_alive_fraction_layers": [],
        "phago_hard_pruned_fraction_layers": [],
        "phago_soft_pruned_fraction_layers": [],
        "phago_mean_soft_mask_layers": [],
        "phago_mean_usefulness_layers": [],
        "phago_mean_weakness_layers": [],
        "phago_mean_leak_signal_layers": [],
        "phago_mean_permissive_signal_layers": [],
        "phago_mean_pruning_pressure_layers": [],
        "phago_max_pruning_pressure_layers": [],
        "phago_num_candidates_layers": [],
        "phago_num_selected_layers": [],
        "phago_num_newly_pruned_layers": [],
    }

    for i, layer in enumerate(model.layers):
        W = layer.weight
        b = layer.bias

        dW = weight_drives[i]
        db = bias_drives[i]

        W_before = W.detach().clone()

        if phago_active and i in model.phago:
            if phago_permissive_signals is not None:
                permissive_i = phago_permissive_signals[i]
            else:
                permissive_i = 0.0

            model.phago[i].observe(
                plasticity_drive=dW.detach(),
                permissive_signal=permissive_i,
            )
            dW_to_apply = model.phago[i].masked_update(dW)
        else:
            dW_to_apply = dW

        if cfg.weight_decay > 0.0:
            W.mul_(1.0 - cfg.lr_weights * cfg.weight_decay)

        W.add_(cfg.lr_weights * dW_to_apply)
        b.add_(cfg.lr_weights * db)

        applied_delta = W.detach() - W_before
        dW_rms_layers.append(_rms(dW_to_apply))
        weight_rms_layers.append(_rms(W.detach()))
        applied_weight_delta_rms_layers.append(_rms(applied_delta))

        if phago_active and i in model.phago:
            phago_stats = model.phago[i].maybe_prune(weight=W)

            phago_metrics["phago_did_prune_layers"].append(phago_stats["did_prune"])
            phago_metrics["phago_alive_fraction_layers"].append(
                phago_stats["alive_fraction"]
            )
            phago_metrics["phago_hard_pruned_fraction_layers"].append(
                phago_stats["hard_pruned_fraction"]
            )
            phago_metrics["phago_soft_pruned_fraction_layers"].append(
                phago_stats["soft_pruned_fraction"]
            )
            phago_metrics["phago_mean_soft_mask_layers"].append(
                phago_stats["mean_soft_mask"]
            )
            phago_metrics["phago_mean_usefulness_layers"].append(
                phago_stats["mean_usefulness"]
            )
            phago_metrics["phago_mean_weakness_layers"].append(
                phago_stats["mean_weakness"]
            )
            phago_metrics["phago_mean_leak_signal_layers"].append(
                phago_stats["mean_leak_signal"]
            )
            phago_metrics["phago_mean_permissive_signal_layers"].append(
                phago_stats.get(
                    "mean_permissive_signal",
                    phago_stats["mean_leak_signal"],
                )
            )
            phago_metrics["phago_mean_pruning_pressure_layers"].append(
                phago_stats["mean_pruning_pressure"]
            )
            phago_metrics["phago_max_pruning_pressure_layers"].append(
                phago_stats.get("max_pruning_pressure", 0.0)
            )
            phago_metrics["phago_num_candidates_layers"].append(
                phago_stats.get("num_candidates", 0.0)
            )
            phago_metrics["phago_num_selected_layers"].append(
                phago_stats.get("num_selected", 0.0)
            )
            phago_metrics["phago_num_newly_pruned_layers"].append(
                phago_stats.get("num_newly_pruned", 0.0)
            )

    out = {
        "dW_rms_layers": dW_rms_layers,
        "weight_rms_layers": weight_rms_layers,
        "applied_weight_delta_rms_layers": applied_weight_delta_rms_layers,
    }

    if phago_active:
        out.update(phago_metrics)

    return out


@torch.no_grad()
def compute_phago_permissive_signals(
    astro_controller_state: AstroCState,
    base_gains: list[torch.Tensor],
    effective_gains: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Compute local gain excess, received leakage, and the phagocytosis gate."""
    if not (
        len(astro_controller_state.layers) == len(base_gains) == len(effective_gains)
    ):
        raise ValueError(
            "Mismatch between controller layers, base gains, and effective gains."
        )

    local_gain_excesses: list[torch.Tensor] = []
    received_leak_increments: list[torch.Tensor] = []
    phago_permissive_signals: list[torch.Tensor] = []

    for layer_state, g_base, g_eff in zip(
        astro_controller_state.layers,
        base_gains,
        effective_gains,
    ):
        g_base_scalar = g_base.reshape(())
        g_eff_scalar = g_eff.reshape(())

        received_leak = (g_eff_scalar - g_base_scalar).clamp_min(0.0)

        baseline = getattr(layer_state, "leak_baseline", None)
        baseline_frozen = bool(getattr(layer_state, "leak_baseline_frozen", True))

        if baseline is None or not baseline_frozen:
            local_excess = torch.zeros_like(g_base_scalar)
        else:
            baseline_t = torch.as_tensor(
                baseline,
                device=g_base_scalar.device,
                dtype=g_base_scalar.dtype,
            ).reshape(())
            local_excess = (g_base_scalar - baseline_t).clamp_min(0.0)

        permissive = local_excess + received_leak

        local_gain_excesses.append(local_excess.detach().clone())
        received_leak_increments.append(received_leak.detach().clone())
        phago_permissive_signals.append(permissive.detach().clone())

    return (
        local_gain_excesses,
        received_leak_increments,
        phago_permissive_signals,
    )


@torch.no_grad()
def apply_astro_controller_to_pc_drives(
    x_states: list[torch.Tensor],
    eps: list[torch.Tensor],
    weight_drives: list[torch.Tensor],
    bias_drives: list[torch.Tensor],
    cfg: PCConfig,
    astro_controller_state: AstroCState | None,
    *,
    total_neural_actions: int,
    pending_neural_actions: int,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    AstroCState | None,
    int,
    dict,
    list[torch.Tensor] | None,
]:
    """Apply the optional astro-controller stage to local PC drives."""
    if not cfg.astro_controller.enabled or astro_controller_state is None:
        return (
            weight_drives,
            bias_drives,
            astro_controller_state,
            pending_neural_actions,
            {},
            None,
        )

    layer_signals = extract_local_layer_signals_for_controller(
        x_states=x_states,
        eps=eps,
        weight_drives=weight_drives,
        bias_drives=bias_drives,
    )

    controller_updated = False
    if should_update_astro_controller(
        total_neural_actions=total_neural_actions,
        pending_neural_actions=pending_neural_actions,
        cfg=cfg.astro_controller,
    ):
        astro_controller_state = recompute_all_astro_controller_states(
            astro_state=astro_controller_state,
            layer_signals=layer_signals,
            cfg=cfg.astro_controller,
        )
        controller_updated = True

        if cfg.astro_controller.update_period_steps > 0:
            pending_neural_actions = (
                pending_neural_actions % cfg.astro_controller.update_period_steps
            )
        else:
            pending_neural_actions = 0

    base_gains = get_layer_plasticity_gains(astro_controller_state)

    astro_controller_state = update_learned_leak_baselines(
        astro_state=astro_controller_state,
        base_gains=base_gains,
        cfg=cfg.astro_controller,
        controller_updated=controller_updated,
    )

    effective_gains, leak_stats = apply_astrocytic_gain_leakage(
        base_gains=base_gains,
        astro_state=astro_controller_state,
        cfg=cfg.astro_controller,
    )

    (
        local_gain_excesses,
        received_leak_increments,
        phago_permissive_signals,
    ) = compute_phago_permissive_signals(
        astro_controller_state=astro_controller_state,
        base_gains=base_gains,
        effective_gains=effective_gains,
    )

    mod_weight_drives, mod_bias_drives = (
        apply_astro_controller_gain_modulation_to_weight_updates(
            base_weight_drives=weight_drives,
            base_bias_drives=bias_drives,
            gains=effective_gains,
        )
    )

    astro_metrics = collect_astro_controller_metrics(
        astro_state=astro_controller_state,
        base_gains=base_gains,
        effective_gains=effective_gains,
        leak_stats=leak_stats,
    )

    local_excess_values = [float(x.item()) for x in local_gain_excesses]
    received_leak_values = [float(x.item()) for x in received_leak_increments]
    phago_gate_values = [float(x.item()) for x in phago_permissive_signals]

    astro_metrics.update(
        {
            "astro_controller_total_neural_actions": total_neural_actions,
            "astro_controller_pending_neural_actions": pending_neural_actions,
            "astro_controller_controller_updated": float(controller_updated),
            "astro_controller_local_gain_excess_layers": local_excess_values,
            "astro_controller_local_gain_excess_mean": (
                sum(local_excess_values) / max(1, len(local_excess_values))
            ),
            "astro_controller_received_leak_layers": received_leak_values,
            "astro_controller_received_leak_mean": (
                sum(received_leak_values) / max(1, len(received_leak_values))
            ),
            "astro_controller_phago_permissive_layers": phago_gate_values,
            "astro_controller_phago_permissive_mean": (
                sum(phago_gate_values) / max(1, len(phago_gate_values))
            ),
            "astro_controller_phago_permissive_max": (
                max(phago_gate_values) if phago_gate_values else 0.0
            ),
        }
    )

    return (
        mod_weight_drives,
        mod_bias_drives,
        astro_controller_state,
        pending_neural_actions,
        astro_metrics,
        phago_permissive_signals,
    )


def pc_train_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: PCConfig,
    state: PCTrainState,
) -> tuple[float, int, dict, PCTrainState]:
    """Run one predictive-coding batch update."""
    logits_free, cache = model(x, return_acts=True)

    loss_before = float(F.cross_entropy(logits_free, y).item())
    correct = int((logits_free.argmax(dim=1) == y).sum().item())

    x_init = [ri.detach().clone() for ri in cache["r"]]
    output_target = compute_pc_output_target(
        logits_free=logits_free,
        y=y,
        beta=cfg.beta,
    )

    infer_out = run_pc_inference(
        model=model,
        x_init=x_init,
        output_target=output_target,
        cfg=cfg,
    )

    actions_this_batch = cfg.infer_steps
    total_neural_actions = state.total_neural_actions + actions_this_batch
    pending_neural_actions = state.pending_neural_actions + actions_this_batch

    drive_stats = compute_pc_fractional_weight_drives(
        model=model,
        x_states=infer_out["x_states"],
        eps=infer_out["eps"],
        cfg=cfg,
        plasticity_state=state.plasticity_state,
    )

    weight_drives = drive_stats["base_weight_drives"]
    bias_drives = drive_stats["base_bias_drives"]

    (
        weight_drives,
        bias_drives,
        astro_controller_state,
        pending_neural_actions,
        astro_metrics,
        phago_permissive_signals,
    ) = apply_astro_controller_to_pc_drives(
        x_states=infer_out["x_states"],
        eps=infer_out["eps"],
        weight_drives=weight_drives,
        bias_drives=bias_drives,
        cfg=cfg,
        astro_controller_state=state.astro_controller_state,
        total_neural_actions=total_neural_actions,
        pending_neural_actions=pending_neural_actions,
    )

    update_stats = apply_pc_weight_drives(
        model=model,
        weight_drives=weight_drives,
        bias_drives=bias_drives,
        cfg=cfg,
        phago_permissive_signals=phago_permissive_signals,
    )

    with torch.no_grad():
        logits_after = model(x)
        loss_after = float(F.cross_entropy(logits_after, y).item())

    eps_rms_layers = [
        float(torch.sqrt(torch.mean(e * e) + 1e-12).item())
        for e in infer_out["eps"][1:]
    ]

    metrics = {
        "ce_before": loss_before,
        "ce_after": loss_after,
        "delta_ce": loss_after - loss_before,
        "pc_energy_trace": infer_out["energy_trace"],
        "pc_state_update_trace": infer_out["state_update_trace"],
        "pc_eps_rms_layers": eps_rms_layers,
        **{k: v for k, v in drive_stats.items() if not k.startswith("base_")},
        **update_stats,
        **astro_metrics,
    }

    new_state = PCTrainState(
        plasticity_state=state.plasticity_state,
        astro_controller_state=astro_controller_state,
        total_neural_actions=total_neural_actions,
        pending_neural_actions=pending_neural_actions,
    )

    return loss_before, correct, metrics, new_state
