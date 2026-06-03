from dataclasses import dataclass, field


@dataclass
class AstroPhagoConfig:
    warmup_steps: int = 1000
    prune_every: int = 200
    hard_patience: int = 3
    max_prune_fraction: float = 0.02
    min_alive_fraction: float = 0.5
    eps: float = 1e-8


@dataclass(frozen=True)
class AstroCConfig:
    enabled: bool = False
    update_period_steps: int = 100
    warmup_steps: int = 0
    dynamics: str = "astro_state"
    state_smoothing: float = 0.5
    gain_scale: float = 0.5
    passive_state_drive: float = 0.5
    state_coupling_enabled: bool = False
    state_coupling_radius: int = 1
    state_coupling_strength: float = 0.0
    leak_enabled: bool = True
    leak_radius: int = 1
    leak_baseline: float | None = None
    leak_baseline_updates: int = 5
    phago_enabled: bool = True
    log_every: int = 200


@dataclass(frozen=True)
class PCConfig:
    activation_function: str
    infer_steps: int
    state_lr: float
    beta: float
    lr_weights: float
    weight_decay: float = 0.0
    grad_clip: float | None = None
    frac_memory_len: int = 0
    frac_rho: float = 1.0
    frac_normalize: bool = True
    astro_controller: AstroCConfig = field(default_factory=AstroCConfig)
    log_every: int = 200
