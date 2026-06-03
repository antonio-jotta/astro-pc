from __future__ import annotations

import math
from dataclasses import asdict

import torch

from controller_configs import AstroPhagoConfig


class AstroPhagocytosis:
    """Synapse-level pruning gated by a permissive astrocyte signal."""

    def __init__(
        self,
        weight_shape,
        device=None,
        dtype=torch.float32,
        cfg: AstroPhagoConfig | None = None,
    ):
        self.cfg = cfg or AstroPhagoConfig()

        self.alive_mask = torch.ones(weight_shape, device=device, dtype=torch.bool)

        self.last_usefulness = torch.zeros(weight_shape, device=device, dtype=dtype)
        self.last_weakness = torch.zeros(weight_shape, device=device, dtype=dtype)

        self.last_leak_signal = torch.tensor(0.0, device=device, dtype=dtype)

        self.weak_counter = torch.zeros(weight_shape, device=device, dtype=torch.long)

        self.last_num_candidates = 0
        self.last_num_selected = 0
        self.last_num_newly_pruned = 0
        self.last_max_pressure = 0.0

        self.step_count = 0

    @property
    def effective_mask(self) -> torch.Tensor:
        return self.alive_mask.to(self.last_usefulness.dtype)

    @torch.no_grad()
    def observe(
        self,
        plasticity_drive: torch.Tensor,
        leak_signal=None,
        permissive_signal=None,
    ) -> None:
        """Record the current plasticity drive and pruning signal."""
        cfg = self.cfg
        self.step_count += 1

        drive = plasticity_drive.detach().abs()
        if drive.shape != self.alive_mask.shape:
            raise ValueError(
                f"plasticity_drive shape {tuple(drive.shape)} does not match "
                f"phagocytosis mask shape {tuple(self.alive_mask.shape)}"
            )

        self.last_usefulness.copy_(drive)

        layer_mean_usefulness = drive.mean().clamp_min(cfg.eps)
        weakness = torch.relu(layer_mean_usefulness - drive) / (
            layer_mean_usefulness + cfg.eps
        )
        self.last_weakness.copy_(weakness)

        if permissive_signal is None:
            permissive_signal = 0.0 if leak_signal is None else leak_signal

        permissive_signal = torch.as_tensor(
            permissive_signal,
            device=self.last_usefulness.device,
            dtype=self.last_usefulness.dtype,
        )
        self.last_leak_signal.copy_(permissive_signal.reshape(()).clamp_min(0.0))

    def _hard_pruned_count(self) -> int:
        return int(self.alive_mask.numel() - int(self.alive_mask.sum().item()))

    def _max_total_hard_pruned(self) -> int:
        """Maximum number of synapses this layer may hard-prune cumulatively."""
        total_synapses = int(self.alive_mask.numel())
        if self.cfg.max_prune_fraction <= 0.0:
            return 0
        return min(
            total_synapses,
            int(math.ceil(self.cfg.max_prune_fraction * total_synapses)),
        )

    @torch.no_grad()
    def maybe_prune(self, weight: torch.Tensor | None = None) -> dict[str, float]:
        cfg = self.cfg

        self.last_num_candidates = 0
        self.last_num_selected = 0
        self.last_num_newly_pruned = 0

        if self.step_count < cfg.warmup_steps:
            return self.stats(did_prune=False)

        if self.step_count % cfg.prune_every != 0:
            return self.stats(did_prune=False)

        alive = self.alive_mask
        num_alive = int(alive.sum().item())
        total_synapses = int(alive.numel())
        if num_alive == 0:
            return self.stats(did_prune=False)

        already_pruned = total_synapses - num_alive
        max_total_pruned = self._max_total_hard_pruned()
        remaining_global_budget = max(0, max_total_pruned - already_pruned)

        min_alive = max(1, int(math.ceil(cfg.min_alive_fraction * total_synapses)))
        remaining_floor_budget = max(0, num_alive - min_alive)

        max_prunable_now = min(remaining_global_budget, remaining_floor_budget)
        if max_prunable_now <= 0:
            self.weak_counter.zero_()
            return self.stats(did_prune=False)

        permissive_signal = self.last_leak_signal.clamp_min(0.0)
        if float(permissive_signal.item()) <= 0.0:
            self.weak_counter.zero_()
            return self.stats(did_prune=False)

        pressure = permissive_signal * self.last_weakness
        self.last_max_pressure = float(pressure.max().item())

        candidate_mask = alive & (pressure > 0.0)
        num_candidates = int(candidate_mask.sum().item())
        self.last_num_candidates = num_candidates
        if num_candidates == 0:
            self.weak_counter.zero_()
            return self.stats(did_prune=False)

        k = min(num_candidates, max_prunable_now)
        if k <= 0:
            self.weak_counter.zero_()
            return self.stats(did_prune=False)

        flat_scores = torch.full_like(pressure, fill_value=-torch.inf).flatten()
        flat_scores[candidate_mask.flatten()] = pressure.flatten()[
            candidate_mask.flatten()
        ]

        topk_vals, topk_idx = torch.topk(flat_scores, k=k)
        selected_flat = torch.zeros_like(flat_scores, dtype=torch.bool)
        selected_flat[topk_idx] = torch.isfinite(topk_vals)
        selected = selected_flat.view_as(pressure)
        self.last_num_selected = int(selected.sum().item())

        self.weak_counter[selected] += 1
        self.weak_counter[~selected] = 0

        newly_dead = self.alive_mask & (self.weak_counter >= cfg.hard_patience)
        num_newly_dead = int(newly_dead.sum().item())

        if num_newly_dead > max_prunable_now:
            dead_scores = torch.full_like(pressure, fill_value=-torch.inf).flatten()
            dead_scores[newly_dead.flatten()] = pressure.flatten()[newly_dead.flatten()]
            vals, idx = torch.topk(dead_scores, k=max_prunable_now)
            keep_flat = torch.zeros_like(dead_scores, dtype=torch.bool)
            keep_flat[idx] = torch.isfinite(vals)
            newly_dead = keep_flat.view_as(pressure)
            num_newly_dead = int(newly_dead.sum().item())

        self.alive_mask[newly_dead] = False
        self.weak_counter[newly_dead] = 0
        self.last_num_newly_pruned = num_newly_dead

        if weight is not None and newly_dead.any():
            weight.data[newly_dead] = 0.0

        did_prune = bool(newly_dead.any().item())
        return self.stats(did_prune=did_prune)

    def masked_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight * self.effective_mask

    def masked_update(self, update: torch.Tensor) -> torch.Tensor:
        return update * self.effective_mask

    def stats(self, did_prune: bool) -> dict[str, float]:
        total = self.alive_mask.numel()
        alive = int(self.alive_mask.sum().item())
        hard_pruned = total - alive

        permissive_signal = self.last_leak_signal.clamp_min(0.0)
        pressure = permissive_signal * self.last_weakness

        return {
            "did_prune": float(did_prune),
            "alive_fraction": alive / total,
            "hard_pruned_fraction": hard_pruned / total,
            "soft_pruned_fraction": 0.0,
            "mean_soft_mask": 1.0,
            "mean_usefulness": float(self.last_usefulness.mean().item()),
            "mean_weakness": float(self.last_weakness.mean().item()),
            "mean_leak_signal": float(permissive_signal.item()),
            "mean_permissive_signal": float(permissive_signal.item()),
            "mean_pruning_pressure": float(pressure.mean().item()),
            "max_pruning_pressure": float(pressure.max().item()),
            "num_candidates": float(self.last_num_candidates),
            "num_selected": float(self.last_num_selected),
            "num_newly_pruned": float(self.last_num_newly_pruned),
            "max_weak_counter": float(self.weak_counter.max().item()),
            "max_total_hard_pruned_fraction": float(self.cfg.max_prune_fraction),
        }

    def state_dict(self) -> dict:
        return {
            "cfg": asdict(self.cfg),
            "alive_mask": self.alive_mask.detach().clone(),
            "last_usefulness": self.last_usefulness.detach().clone(),
            "last_weakness": self.last_weakness.detach().clone(),
            "last_leak_signal": self.last_leak_signal.detach().clone(),
            "weak_counter": self.weak_counter.detach().clone(),
            "step_count": int(self.step_count),
            "last_num_candidates": int(self.last_num_candidates),
            "last_num_selected": int(self.last_num_selected),
            "last_num_newly_pruned": int(self.last_num_newly_pruned),
            "last_max_pressure": float(self.last_max_pressure),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        if "cfg" in state:
            self.cfg = AstroPhagoConfig(**state["cfg"])

        self.alive_mask.copy_(
            state["alive_mask"].to(
                device=self.alive_mask.device, dtype=self.alive_mask.dtype
            )
        )
        self.last_usefulness.copy_(
            state["last_usefulness"].to(
                device=self.last_usefulness.device,
                dtype=self.last_usefulness.dtype,
            )
        )
        self.last_weakness.copy_(
            state["last_weakness"].to(
                device=self.last_weakness.device,
                dtype=self.last_weakness.dtype,
            )
        )
        self.last_leak_signal.copy_(
            state["last_leak_signal"].to(
                device=self.last_leak_signal.device,
                dtype=self.last_leak_signal.dtype,
            )
        )
        self.weak_counter.copy_(
            state["weak_counter"].to(
                device=self.weak_counter.device,
                dtype=self.weak_counter.dtype,
            )
        )
        self.step_count = int(state["step_count"])
        self.last_num_candidates = int(state.get("last_num_candidates", 0))
        self.last_num_selected = int(state.get("last_num_selected", 0))
        self.last_num_newly_pruned = int(state.get("last_num_newly_pruned", 0))
        self.last_max_pressure = float(state.get("last_max_pressure", 0.0))


def build_phago_config(
    cfg,
    layer_idx: int,
    num_layers: int,
) -> AstroPhagoConfig:
    astro_period_batches = max(
        1,
        math.ceil(
            cfg.astro_controller_update_period_steps / max(1, cfg.pc_infer_steps)
        ),
    )

    prune_every = 3 * astro_period_batches
    warmup_steps = 5 * prune_every

    is_edge_layer = (layer_idx == 0) or (layer_idx == num_layers - 1)
    min_alive_fraction = 0.75 if is_edge_layer else 0.50

    return AstroPhagoConfig(
        warmup_steps=warmup_steps,
        prune_every=prune_every,
        hard_patience=cfg.phago_hard_patience,
        max_prune_fraction=cfg.phago_max_prune_fraction,
        min_alive_fraction=min_alive_fraction,
    )


def model_has_phagocytosis(model) -> bool:
    return hasattr(model, "phago") and model.phago is not None and len(model.phago) > 0


def use_phagocytosis(model, pc_cfg) -> bool:
    return bool(
        getattr(pc_cfg.astro_controller, "phago_enabled", False)
    ) and model_has_phagocytosis(model)


def get_effective_weight(model, layer_idx: int) -> torch.Tensor:
    layer = model.layers[layer_idx]

    if model_has_phagocytosis(model) and layer_idx in model.phago:
        return model.phago[layer_idx].masked_weight(layer.weight)

    return layer.weight


def attach_phagocytosis_from_cfg(model, cfg) -> None:
    if cfg.method != "pc" or not cfg.phago_enabled:
        if hasattr(model, "detach_phagocytosis"):
            model.detach_phagocytosis()
        elif hasattr(model, "phago"):
            model.phago = None
        return

    num_layers = len(model.layers)

    phago = {
        li: AstroPhagocytosis(
            weight_shape=layer.weight.shape,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
            cfg=build_phago_config(
                cfg=cfg,
                layer_idx=li,
                num_layers=num_layers,
            ),
        )
        for li, layer in enumerate(model.layers)
    }

    model.attach_phagocytosis(phago)


def summarize_phagocytosis(model) -> list[str]:
    if not model_has_phagocytosis(model):
        return ["[Phagocytosis summary] No phagocytosis modules found."]

    total_synapses = 0
    total_alive = 0

    lines = ["[Phagocytosis summary]"]

    for layer_idx in sorted(model.phago.keys()):
        ph = model.phago[layer_idx]

        total = int(ph.alive_mask.numel())
        alive = int(ph.alive_mask.sum().item())
        pruned = total - alive

        total_synapses += total
        total_alive += alive

        lines.append(
            f"  Layer {layer_idx}: "
            f"hard-pruned {pruned}/{total} ({100.0 * pruned / max(1, total):.2f}%) "
            f"[cap={100.0 * ph.cfg.max_prune_fraction:.2f}%]"
        )

    total_pruned = total_synapses - total_alive
    cap = None
    if hasattr(model, "phago") and model.phago:
        any_phago = next(iter(model.phago.values()))
        cap = any_phago.cfg.max_prune_fraction

    cap_text = "" if cap is None else f" [per-layer cap={100.0 * cap:.2f}%]"
    lines.append(
        f"  Total: hard-pruned {total_pruned}/{total_synapses} "
        f"({100.0 * total_pruned / max(1, total_synapses):.2f}%){cap_text}"
    )

    return lines


def print_phagocytosis_summary(model) -> None:
    for line in summarize_phagocytosis(model):
        print(line, flush=True)
