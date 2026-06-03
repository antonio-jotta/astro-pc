import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch

from astro_phagocytosis import (
    attach_phagocytosis_from_cfg,
    print_phagocytosis_summary,
)
from checkpointing import (
    load_run_status,
    load_training_checkpoint,
    metric_curves_from_history,
    save_config_snapshot,
    save_progress_artifacts,
    save_run_status,
    save_training_checkpoint,
)
from config import Config
from data import make_data_loaders
from early_stopping import EarlyStoppingState, update_early_stopping
from engine import evaluate, train_one_epoch_bp, train_one_epoch_pc
from experiment_utils import (
    _is_bad_number,
    _last_finite_value,
    _pad_metrics_after_failure,
    _should_stop_for_numerical_failure,
    build_run_name,
    build_unique_name,
)
from metrics import compute_epoch_mean_dw
from model import MLP
from pc import initialize_pc_train_state
from plots import plot_accuracy, plot_astro_leakage_metrics, plot_loss


def build_model(cfg: Config, device: torch.device) -> MLP:
    in_dim, out_dim = cfg.model_dims()
    hidden_dims = cfg.get_hidden_dims()

    model = MLP(
        in_dim=in_dim,
        hidden_dims=hidden_dims,
        out_dim=out_dim,
        activation_function=cfg.activation_function,
    )
    model = model.to(device=device, dtype=cfg.dtype)

    attach_phagocytosis_from_cfg(model, cfg)
    return model


def print_system_info(device: torch.device, cfg: Config, run_name: str) -> None:
    print(f"\n{'='*46}")
    print(f"\033[94mUsing device: {device}, dtype: {cfg.dtype}\033[0m")
    print(f"Run name: {run_name}")
    print(f"{'='*46}\n")

    if torch.cuda.is_available():
        print("\033[93mCUDA Information:\033[0m")
        print(f"  {'-'*44}")
        print(f"  CUDA device count: \033[92m{torch.cuda.device_count()}\033[0m")
        print(
            f"  Current CUDA device index: \033[92m{torch.cuda.current_device()}\033[0m"
        )
        print(
            f"  CUDA device name: \033[92m{torch.cuda.get_device_name(torch.cuda.current_device())}\033[0m"
        )
        print(f"  {'-'*44}")
    else:
        print("Running on CPU only.")

    print(f"{'='*46}\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_current_epoch_values(
    *,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
    train_losses: list[float],
    train_accs: list[float],
    test_losses: list[float],
    test_accs: list[float],
) -> tuple[float, float, float, float]:
    """Return finite values for the failure epoch where needed."""
    safe_train_loss = (
        _last_finite_value(train_losses[:-1], fallback=0.0)
        if _is_bad_number(train_loss)
        else float(train_loss)
    )
    safe_test_loss = (
        _last_finite_value(test_losses[:-1], fallback=0.0)
        if _is_bad_number(test_loss)
        else float(test_loss)
    )
    safe_train_acc = (
        _last_finite_value(train_accs[:-1], fallback=0.0)
        if _is_bad_number(train_acc)
        else float(train_acc)
    )
    safe_test_acc = (
        _last_finite_value(test_accs[:-1], fallback=0.0)
        if _is_bad_number(test_acc)
        else float(test_acc)
    )

    return safe_train_loss, safe_train_acc, safe_test_loss, safe_test_acc


def _trim_layer_diagnostics_csv(
    csv_path: str | Path, *, max_completed_epoch: int
) -> None:
    """Remove diagnostics past the last completed epoch."""
    path = Path(csv_path)
    if not path.exists():
        return

    if max_completed_epoch < 1:
        path.unlink()
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", newline="", encoding="utf-8") as src, tmp_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            return
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                epoch = int(float(row.get("epoch", 0)))
            except (TypeError, ValueError):
                continue

            if epoch <= max_completed_epoch:
                writer.writerow(row)
    tmp_path.replace(path)


def main(
    cfg: Config | None = None,
    run_name: str | None = None,
    save_run_plots: bool = True,
    output_root: str = "plots",
) -> None:
    cfg = cfg or Config()
    cfg.validate()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    if run_name is None:
        base_run_name = build_run_name(cfg)
        run_name = build_unique_name(base_run_name, output_root)

    plot_dir = os.path.join(output_root, run_name)
    os.makedirs(plot_dir, exist_ok=True)

    diagnostics_dir = os.path.join(plot_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    layer_diagnostics_csv = os.path.join(
        diagnostics_dir,
        "layer_internal_diagnostics_stream.csv",
    )

    save_config_snapshot(plot_dir, cfg)

    status = load_run_status(plot_dir)
    if status is not None and status.get("status") in (
        "completed",
        "failed_divergence",
    ):
        print(f"Run already has terminal status '{status.get('status')}': {plot_dir}")
        return

    print_system_info(device, cfg, run_name)

    train_loader, test_loader = make_data_loaders(cfg, device)
    model = build_model(cfg, device)

    bp_optimizer = None
    pc_cfg = None
    pc_state = None

    if cfg.method == "bp":
        bp_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    elif cfg.method == "pc":
        pc_cfg = cfg.make_method_cfg(log_every=200)
        pc_state = initialize_pc_train_state(
            model=model,
            cfg=pc_cfg,
        )

    else:
        raise ValueError(f"Unknown method: {cfg.method}. Use 'bp' or 'pc'.")

    train_losses: list[float] = []
    train_accs: list[float] = []
    test_losses: list[float] = []
    test_accs: list[float] = []
    metrics_logging: list[dict] = []
    method_metrics: list[dict] = []

    early_stop_state = EarlyStoppingState()
    start_epoch = 1
    elapsed_time_before_resume = 0.0
    training_finished = False
    stop_reason: str | None = None
    failed_divergence = False

    checkpoint = load_training_checkpoint(
        plot_dir,
        model=model,
        bp_optimizer=bp_optimizer,
        device=device,
    )

    if checkpoint is not None:
        start_epoch = int(checkpoint["epoch"]) + 1
        elapsed_time_before_resume = float(checkpoint.get("elapsed_time_seconds", 0.0))
        training_finished = bool(checkpoint.get("training_finished", False))
        stop_reason = checkpoint.get("stop_reason", None)
        failed_divergence = stop_reason is not None and "Numerical failure" in str(
            stop_reason
        )

        if cfg.method == "pc" and checkpoint.get("pc_state") is not None:
            pc_state = checkpoint["pc_state"]

        if checkpoint.get("early_stop_state") is not None:
            early_stop_state = checkpoint["early_stop_state"]

        metrics_logging = checkpoint.get("metrics_logging", [])
        method_metrics = checkpoint.get("method_metrics", [])
        train_losses, train_accs, test_losses, test_accs = metric_curves_from_history(
            metrics_logging
        )

        if training_finished:
            print(
                f"Training already finished for {run_name}. "
                f"Rebuilding final artifacts/plots only."
            )
        else:
            print(f"Resuming {run_name} from epoch {start_epoch}.")

    _trim_layer_diagnostics_csv(
        layer_diagnostics_csv,
        max_completed_epoch=start_epoch - 1,
    )

    last_completed_epoch = len(metrics_logging)
    segment_start_time = time.time()

    save_run_status(
        plot_dir,
        run_name=run_name,
        cfg=cfg,
        status="running",
        last_completed_epoch=last_completed_epoch,
        elapsed_time_seconds=elapsed_time_before_resume,
        stop_reason=stop_reason,
    )

    try:
        if not training_finished:
            for epoch in range(start_epoch, cfg.epochs + 1):
                if cfg.method == "bp":
                    train_loss, train_acc = train_one_epoch_bp(
                        model=model,
                        loader=train_loader,
                        optimizer=bp_optimizer,
                        device=device,
                        epoch=epoch,
                    )

                elif cfg.method == "pc":
                    train_loss, train_acc, pc_state = train_one_epoch_pc(
                        model=model,
                        loader=train_loader,
                        device=device,
                        epoch=epoch,
                        cfg=pc_cfg,
                        state=pc_state,
                        metrics_store=method_metrics,
                        diagnostics_csv_path=layer_diagnostics_csv,
                        keep_full_batch_metrics_in_memory=False,
                    )

                else:
                    raise ValueError(f"Unknown method: {cfg.method}")

                epoch_mean_dW = None
                if cfg.method == "pc":
                    epoch_mean_dW = compute_epoch_mean_dw(method_metrics, epoch)

                test_loss, test_acc = evaluate(model, test_loader, device)

                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_losses.append(test_loss)
                test_accs.append(test_acc)

                epoch_record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "method": cfg.method,
                    "dataset": cfg.dataset,
                    "seed": cfg.seed,
                    "failed": 0,
                    "imputed_after_failure": 0,
                    "failure_reason": None,
                }

                if cfg.method == "pc":
                    epoch_record.update(
                        {
                            "pc_frac_memory_len": cfg.pc_frac_memory_len,
                            "pc_frac_rho": cfg.pc_frac_rho,
                            "astro_controller_enabled": cfg.astro_controller_enabled,
                            "astro_controller_update_period_steps": cfg.astro_controller_update_period_steps,
                            "astro_controller_dynamics": cfg.astro_controller_dynamics,
                            "astro_controller_state_coupling_enabled": int(
                                cfg.astro_controller_state_coupling_enabled
                            ),
                            "astro_controller_state_coupling_strength": cfg.astro_controller_state_coupling_strength,
                            "astro_controller_leak_enabled": int(
                                cfg.astro_controller_leak_enabled
                            ),
                            "phago_enabled": int(cfg.phago_enabled),
                            "epoch_mean_dW_rms": epoch_mean_dW,
                        }
                    )

                metrics_logging.append(epoch_record)

                print(
                    f"\033[93m[Evaluation] Epoch {epoch} | "
                    f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}\033[0m"
                )

                if epoch_mean_dW is not None:
                    print(
                        f"[Epoch {epoch}] Mean dW_rms: {epoch_mean_dW:.3e}",
                        flush=True,
                    )

                numerical_failure, failure_reason = _should_stop_for_numerical_failure(
                    train_loss=train_loss,
                    train_acc=train_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
                    epoch_mean_dW=epoch_mean_dW,
                )

                if numerical_failure:
                    print(
                        f"\n[Fail-safe stop] {failure_reason}. "
                        f"Stopping at epoch {epoch} and padding remaining metrics.",
                        flush=True,
                    )

                    (
                        safe_train_loss,
                        safe_train_acc,
                        safe_test_loss,
                        safe_test_acc,
                    ) = _safe_current_epoch_values(
                        train_loss=train_loss,
                        train_acc=train_acc,
                        test_loss=test_loss,
                        test_acc=test_acc,
                        train_losses=train_losses,
                        train_accs=train_accs,
                        test_losses=test_losses,
                        test_accs=test_accs,
                    )

                    train_losses[-1] = safe_train_loss
                    train_accs[-1] = safe_train_acc
                    test_losses[-1] = safe_test_loss
                    test_accs[-1] = safe_test_acc

                    metrics_logging[-1].update(
                        {
                            "train_loss": safe_train_loss,
                            "train_acc": safe_train_acc,
                            "test_loss": safe_test_loss,
                            "test_acc": safe_test_acc,
                            "failed": 1,
                            "imputed_after_failure": 0,
                            "failure_reason": failure_reason,
                        }
                    )

                    if cfg.method == "pc":
                        metrics_logging[-1]["epoch_mean_dW_rms"] = (
                            None if _is_bad_number(epoch_mean_dW) else epoch_mean_dW
                        )

                    _pad_metrics_after_failure(
                        metrics_logging=metrics_logging,
                        start_epoch=epoch + 1,
                        final_epoch=cfg.epochs,
                        train_losses=train_losses,
                        train_accs=train_accs,
                        test_losses=test_losses,
                        test_accs=test_accs,
                        method=cfg.method,
                        dataset=cfg.dataset,
                        seed=cfg.seed,
                        cfg=cfg,
                        failed_test_acc=safe_test_acc,
                        failed_train_acc=safe_train_acc,
                        failed_test_loss=safe_test_loss,
                        failed_train_loss=safe_train_loss,
                        epoch_mean_dW=epoch_mean_dW,
                        failure_reason=failure_reason,
                    )

                    stop_reason = failure_reason
                    failed_divergence = True
                    training_finished = True
                    last_completed_epoch = epoch

                    current_elapsed = elapsed_time_before_resume + (
                        time.time() - segment_start_time
                    )

                    save_progress_artifacts(
                        plot_dir,
                        metrics_logging=metrics_logging,
                        method_metrics=method_metrics,
                    )

                    save_training_checkpoint(
                        plot_dir,
                        epoch=epoch,
                        elapsed_time_seconds=current_elapsed,
                        training_finished=True,
                        stop_reason=stop_reason,
                        model=model,
                        bp_optimizer=bp_optimizer,
                        pc_state=pc_state,
                        early_stop_state=early_stop_state,
                        metrics_logging=metrics_logging,
                        method_metrics=method_metrics,
                    )

                    save_run_status(
                        plot_dir,
                        run_name=run_name,
                        cfg=cfg,
                        status="failed_divergence",
                        last_completed_epoch=epoch,
                        elapsed_time_seconds=current_elapsed,
                        stop_reason=stop_reason,
                    )

                    break

                early_stop_state, should_stop, stop_message = update_early_stopping(
                    early_stop_state,
                    cfg,
                    epoch=epoch,
                    test_loss=test_loss,
                    test_acc=test_acc,
                    epoch_mean_dW=epoch_mean_dW,
                )

                current_elapsed = elapsed_time_before_resume + (
                    time.time() - segment_start_time
                )

                save_progress_artifacts(
                    plot_dir,
                    metrics_logging=metrics_logging,
                    method_metrics=method_metrics,
                )

                save_training_checkpoint(
                    plot_dir,
                    epoch=epoch,
                    elapsed_time_seconds=current_elapsed,
                    training_finished=False,
                    stop_reason=stop_reason,
                    model=model,
                    bp_optimizer=bp_optimizer,
                    pc_state=pc_state,
                    early_stop_state=early_stop_state,
                    metrics_logging=metrics_logging,
                    method_metrics=method_metrics,
                )

                last_completed_epoch = epoch

                save_run_status(
                    plot_dir,
                    run_name=run_name,
                    cfg=cfg,
                    status="running",
                    last_completed_epoch=last_completed_epoch,
                    elapsed_time_seconds=current_elapsed,
                    stop_reason=stop_reason,
                )

                if should_stop:
                    if stop_message is not None:
                        print(stop_message, flush=True)

                    stop_reason = stop_message or "early_stopping"
                    training_finished = True
                    break

            if not training_finished:
                training_finished = True
                stop_reason = stop_reason or "max_epochs_reached"

    except KeyboardInterrupt:
        current_elapsed = elapsed_time_before_resume + (
            time.time() - segment_start_time
        )

        save_run_status(
            plot_dir,
            run_name=run_name,
            cfg=cfg,
            status="interrupted",
            last_completed_epoch=last_completed_epoch,
            elapsed_time_seconds=current_elapsed,
            stop_reason="KeyboardInterrupt",
        )
        raise

    except Exception as exc:
        current_elapsed = elapsed_time_before_resume + (
            time.time() - segment_start_time
        )

        save_run_status(
            plot_dir,
            run_name=run_name,
            cfg=cfg,
            status="failed",
            last_completed_epoch=last_completed_epoch,
            elapsed_time_seconds=current_elapsed,
            stop_reason=f"{type(exc).__name__}: {exc}",
        )
        raise

    total_time = elapsed_time_before_resume + (time.time() - segment_start_time)

    print(f"{'='*66}")
    print("\033[95mTraining and evaluation complete!\033[0m")
    print(f"\n\033[95mTotal training time: {total_time:.2f} seconds\033[0m")
    print(f"{'='*66}\n")

    if cfg.method == "pc" and cfg.phago_enabled:
        print_phagocytosis_summary(model)

    save_progress_artifacts(
        plot_dir,
        metrics_logging=metrics_logging,
        method_metrics=method_metrics,
    )

    final_status = "failed_divergence" if failed_divergence else "completed"

    save_training_checkpoint(
        plot_dir,
        epoch=last_completed_epoch,
        elapsed_time_seconds=total_time,
        training_finished=True,
        stop_reason=stop_reason,
        model=model,
        bp_optimizer=bp_optimizer,
        pc_state=pc_state,
        early_stop_state=early_stop_state,
        metrics_logging=metrics_logging,
        method_metrics=method_metrics,
    )

    if save_run_plots:
        plot_loss(train_losses, test_losses, out_dir=plot_dir, filename="mlp_loss.png")
        plot_accuracy(train_accs, test_accs, out_dir=plot_dir, filename="mlp_acc.png")
        if cfg.method == "pc":
            plot_astro_leakage_metrics(
                method_metrics=method_metrics,
                out_dir=plot_dir,
                prefix="mlp_",
            )
            try:
                from full_model_diagnostics import generate_full_model_diagnostics

                generate_full_model_diagnostics(
                    run_dir=plot_dir,
                    out_dir=os.path.join(plot_dir, "diagnostics"),
                    verbose=False,
                )
                print(
                    f"[Diagnostics] Full-model diagnostics saved to: "
                    f"{os.path.join(plot_dir, 'diagnostics')}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[Diagnostics warning] Could not generate full-model diagnostics: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    save_run_status(
        plot_dir,
        run_name=run_name,
        cfg=cfg,
        status=final_status,
        last_completed_epoch=last_completed_epoch,
        elapsed_time_seconds=total_time,
        stop_reason=stop_reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or resume a PC/BP experiment.")

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Name of the run directory inside output_root. "
            "If this points to an existing interrupted run, training resumes from its checkpoint."
        ),
    )

    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help=(
            "Path to an existing run directory to resume, e.g. "
            "plots/pc_mnist_seed0_K2_rho0.85_ctrl-on_dyn-astro_state_period100_leak-on_phago-on__run1."
        ),
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="plots",
        help="Root directory where runs are saved.",
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plot generation after training.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_root = args.output_root
    run_name = args.run_name

    if args.resume_dir is not None:
        resume_dir = Path(args.resume_dir)
        output_root = str(resume_dir.parent)
        run_name = resume_dir.name

    main(
        run_name=run_name,
        output_root=output_root,
        save_run_plots=not args.no_plots,
    )
