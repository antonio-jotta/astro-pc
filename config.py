from dataclasses import dataclass

import torch

from controller_configs import AstroCConfig, PCConfig


@dataclass(frozen=True)
class Config:
    seed: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    dataset: str = "mnist"
    method: str = "pc"

    astro_controller_enabled: bool = True
    astro_controller_leak_enabled: bool = True
    phago_enabled: bool = True

    hidden_dims: tuple[int, ...] | None = None
    activation_function: str = "tanh"

    batch_size: int = 64
    epochs: int = 2
    num_workers: int = 2
    pin_memory: bool = False

    early_stop_enabled: bool = False
    early_stop_mode: str = "weight"
    early_stop_warmup_epochs: int = 10
    early_stop_metric: str = "acc"
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4
    early_stop_weight_threshold: float = 2e-4
    early_stop_weight_patience: int = 10

    pc_infer_steps: int = 50
    pc_state_lr: float = 0.025
    pc_beta: float = 3.0
    pc_lr_weights: float = 0.075
    pc_weight_decay: float = 0.0
    pc_grad_clip: float | None = None

    pc_frac_memory_len: int = 2
    pc_frac_rho: float = 0.85
    pc_frac_normalize: bool = True

    astro_controller_update_period_steps: int = 100
    astro_controller_warmup_steps: int = 0
    astro_controller_dynamics: str = "astro_state"

    astro_controller_state_smoothing: float = 0.25

    astro_controller_gain_scale: float = 0.25

    astro_controller_passive_state_drive: float = 0.5

    astro_controller_state_coupling_enabled: bool = True
    astro_controller_state_coupling_radius: int = 1
    astro_controller_state_coupling_strength: float = 0.5

    astro_controller_leak_radius: int = 1
    astro_controller_leak_baseline: float | None = None

    phago_hard_patience: int = 2
    phago_max_prune_fraction: float = 0.01

    mnist_root: str = "./data"

    train_size: int = 1000
    test_size: int = 200
    xor_noise_std: float = 0.2

    tabular_test_size: float = 0.2

    lr: float = 1e-3
    weight_decay: float = 0.0

    def dataset_kwargs(self, *, device_type: str):
        if self.dataset == "mnist":
            return dict(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                device_type=device_type,
                root=self.mnist_root,
            )
        elif self.dataset == "xor":
            return dict(
                train_size=self.train_size,
                test_size=self.test_size,
                batch_size=self.batch_size,
                noise_std=self.xor_noise_std,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                seed=self.seed,
                device_type=device_type,
            )
        elif self.dataset == "wine":
            return dict(
                batch_size=self.batch_size,
                test_size=self.tabular_test_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                seed=self.seed,
                device_type=device_type,
            )
        elif self.dataset == "breast_cancer":
            return dict(
                batch_size=self.batch_size,
                test_size=self.tabular_test_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                seed=self.seed,
                device_type=device_type,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def model_dims(self) -> tuple[int, int]:
        if self.dataset == "mnist":
            return 784, 10
        elif self.dataset == "xor":
            return 2, 2
        elif self.dataset == "wine":
            return 13, 3
        elif self.dataset == "breast_cancer":
            return 30, 2
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def get_hidden_dims(self) -> tuple[int, ...]:
        if self.hidden_dims is not None:
            return self.hidden_dims
        if self.dataset == "xor":
            return (16, 16, 16)
        elif self.dataset == "wine":
            return (64, 64, 32, 32)
        elif self.dataset == "breast_cancer":
            return (64, 64, 64, 32)
        elif self.dataset == "mnist":
            return (256, 256, 256, 128)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def make_method_cfg(self, *, log_every: int = 200):
        if self.method == "bp":
            return None
        elif self.method == "pc":
            astro_controller_cfg = AstroCConfig(
                enabled=self.astro_controller_enabled,
                update_period_steps=self.astro_controller_update_period_steps,
                warmup_steps=self.astro_controller_warmup_steps,
                dynamics=self.astro_controller_dynamics,
                state_smoothing=self.astro_controller_state_smoothing,
                gain_scale=self.astro_controller_gain_scale,
                passive_state_drive=self.astro_controller_passive_state_drive,
                state_coupling_enabled=self.astro_controller_state_coupling_enabled,
                state_coupling_radius=self.astro_controller_state_coupling_radius,
                state_coupling_strength=self.astro_controller_state_coupling_strength,
                leak_enabled=self.astro_controller_leak_enabled,
                leak_radius=self.astro_controller_leak_radius,
                leak_baseline=self.astro_controller_leak_baseline,
                phago_enabled=self.phago_enabled,
                log_every=log_every,
            )

            return PCConfig(
                activation_function=self.activation_function,
                infer_steps=self.pc_infer_steps,
                state_lr=self.pc_state_lr,
                beta=self.pc_beta,
                lr_weights=self.pc_lr_weights,
                weight_decay=self.pc_weight_decay,
                grad_clip=self.pc_grad_clip,
                frac_memory_len=self.pc_frac_memory_len,
                frac_rho=self.pc_frac_rho,
                frac_normalize=self.pc_frac_normalize,
                astro_controller=astro_controller_cfg,
                log_every=log_every,
            )
        else:
            raise ValueError(f"Unknown training method: {self.method}")

    def validate(self) -> None:
        valid_datasets = {"xor", "wine", "breast_cancer", "mnist"}
        valid_methods = {"bp", "pc"}
        valid_activations = {"tanh", "relu", "sigmoid", "identity"}
        valid_early_stop_modes = {"metric", "weight", "both"}
        valid_early_stop_metrics = {"loss", "acc"}
        valid_astro_controller_dynamics = {
            "legacy_gain",
            "astro_state",
            "passive_slow_gain",
        }

        if self.dataset not in valid_datasets:
            raise ValueError(
                f"Unknown dataset: {self.dataset}. Expected one of {sorted(valid_datasets)}."
            )
        if self.method not in valid_methods:
            raise ValueError(
                f"Unknown method: {self.method}. Expected one of {sorted(valid_methods)}."
            )
        if self.activation_function not in valid_activations:
            raise ValueError(
                f"Unknown activation_function: {self.activation_function}. Expected one of {sorted(valid_activations)}."
            )
        if self.early_stop_mode not in valid_early_stop_modes:
            raise ValueError(
                f"Unknown early_stop_mode: {self.early_stop_mode}. Expected one of {sorted(valid_early_stop_modes)}."
            )
        if self.early_stop_metric not in valid_early_stop_metrics:
            raise ValueError(
                f"Unknown early_stop_metric: {self.early_stop_metric}. Expected one of {sorted(valid_early_stop_metrics)}."
            )

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if self.lr < 0.0:
            raise ValueError("BP lr must be >= 0.")
        if self.weight_decay < 0.0:
            raise ValueError("BP weight_decay must be >= 0.")

        if self.pc_infer_steps <= 0:
            raise ValueError("pc_infer_steps must be > 0.")
        if self.pc_state_lr <= 0.0:
            raise ValueError("pc_state_lr must be > 0.")
        if self.pc_beta < 0.0:
            raise ValueError("pc_beta must be >= 0.")
        if self.pc_lr_weights <= 0.0:
            raise ValueError("pc_lr_weights must be > 0.")
        if self.pc_weight_decay < 0.0:
            raise ValueError("pc_weight_decay must be >= 0.")
        if self.pc_grad_clip is not None and self.pc_grad_clip <= 0.0:
            raise ValueError("pc_grad_clip must be > 0 when provided.")
        if self.pc_frac_memory_len < 0:
            raise ValueError("pc_frac_memory_len must be >= 0.")
        if self.pc_frac_rho < 0.0:
            raise ValueError("pc_frac_rho must be >= 0.")

        if self.astro_controller_update_period_steps < 0:
            raise ValueError("astro_controller_update_period_steps must be >= 0.")
        if self.astro_controller_warmup_steps < 0:
            raise ValueError("astro_controller_warmup_steps must be >= 0.")
        if self.astro_controller_dynamics not in valid_astro_controller_dynamics:
            raise ValueError(
                f"Unknown astro_controller_dynamics: {self.astro_controller_dynamics}. "
                f"Expected one of {sorted(valid_astro_controller_dynamics)}."
            )
        if not (0.0 <= self.astro_controller_state_smoothing <= 1.0):
            raise ValueError("astro_controller_state_smoothing must be in [0, 1].")
        if not (0.0 <= self.astro_controller_gain_scale < 1.0):
            raise ValueError("astro_controller_gain_scale must satisfy 0 <= value < 1.")
        if self.astro_controller_state_coupling_radius < 0:
            raise ValueError("astro_controller_state_coupling_radius must be >= 0.")
        if self.astro_controller_state_coupling_strength < 0.0:
            raise ValueError("astro_controller_state_coupling_strength must be >= 0.")
        if self.astro_controller_leak_radius < 0:
            raise ValueError("astro_controller_leak_radius must be >= 0.")
        if (
            self.astro_controller_leak_baseline is not None
            and self.astro_controller_leak_baseline < 0.0
        ):
            raise ValueError(
                "astro_controller_leak_baseline must be >= 0 when provided."
            )

        if self.phago_hard_patience <= 0:
            raise ValueError("phago_hard_patience must be > 0.")
        if not (0.0 <= self.phago_max_prune_fraction <= 1.0):
            raise ValueError("phago_max_prune_fraction must be in [0, 1].")
        if self.train_size <= 0:
            raise ValueError("train_size must be > 0.")
        if self.test_size <= 0:
            raise ValueError("test_size must be > 0.")
        if self.xor_noise_std < 0.0:
            raise ValueError("xor_noise_std must be >= 0.")
        if not (0.0 < self.tabular_test_size < 1.0):
            raise ValueError("tabular_test_size must be in (0, 1).")
        if self.early_stop_warmup_epochs < 0:
            raise ValueError("early_stop_warmup_epochs must be >= 0.")
        if self.early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be > 0.")
        if self.early_stop_min_delta < 0.0:
            raise ValueError("early_stop_min_delta must be >= 0.")
        if self.early_stop_weight_threshold < 0.0:
            raise ValueError("early_stop_weight_threshold must be >= 0.")
        if self.early_stop_weight_patience <= 0:
            raise ValueError("early_stop_weight_patience must be > 0.")

        if self.phago_enabled and self.method != "pc":
            raise ValueError("phago_enabled=True is only valid when method='pc'.")
        if self.astro_controller_enabled and self.method != "pc":
            raise ValueError(
                "astro_controller_enabled=True is only valid when method='pc'."
            )
        if self.astro_controller_leak_enabled and self.method != "pc":
            raise ValueError(
                "astro_controller_leak_enabled=True is only valid when method='pc'."
            )
        if self.phago_enabled and not self.astro_controller_enabled:
            raise ValueError(
                "Invalid configuration: phago_enabled=True requires astro_controller_enabled=True."
            )
