#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_automotive_tests.py

Load a trained checkpoint from train_ipinn_synthetic_odeforward.py and run
a set of automotive-oriented tests on the trained model:

  1) Per-scenario full-sequence metrics (RMSE, mean_abs, max_abs, q95_abs, q99_abs)
     vs T_case_meas and T_case_true.

  2) Sensor noise robustness test on windowed data:
       - baseline (no noise)
       - with Gaussian noise on normalized inputs

  3) Sensor bias robustness test on windowed data:
       - bias on ambient / baseplate temperatures (normalized space)

  4) "Cut & restart" test on the held-out test scenario:
       - compare predictions for the full sequence vs predictions obtained
         by restarting integration from a mid-sequence point using T_pred.

This script is meant to be a *post-training test bench* to evaluate
safety-/robustness-oriented behavior of the PINN model.

Example usage:

  python run_automotive_tests.py \
      --data-dir synthetic_thermal \
      --ckpt results/mlp/mlp_final.pt \
      --arch mlp \
      --hidden 8 \
      --device cpu \
      --window 512 \
      --batch 16
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import everything we need from the training script
from train import (
    SyntheticThermalWindowDataset,
    MLPRegressor,
    TCNRegressor,
    LSTMRegressor,
    ODEForwardTrainer,
    robust_dt,
)


# ---------------------------------------------------------------------
# Generic error metrics
# ---------------------------------------------------------------------
def error_stats_from_errors(errors: np.ndarray) -> Dict[str, float]:
    """
    Compute safety-oriented error statistics from a 1D array of errors e:
      - rmse      : sqrt(mean(e^2))
      - mean_abs  : mean(|e|)
      - max_abs   : max(|e|)
      - q95_abs   : 95th percentile of |e|
      - q99_abs   : 99th percentile of |e|
    """
    e = errors[np.isfinite(errors)]
    if e.size == 0:
        return {}
    abs_e = np.abs(e)
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mean_abs": float(np.mean(abs_e)),
        "max_abs": float(np.max(abs_e)),
        "q95_abs": float(np.quantile(abs_e, 0.95)),
        "q99_abs": float(np.quantile(abs_e, 0.99)),
    }


def error_stats_over_loader(
    trainer: ODEForwardTrainer,
    loader: DataLoader,
    device: torch.device,
    target_key: str = "T_case_meas",
    noise_std: float = 0.0,
    bias_dict: Optional[Dict[int, float]] = None,
) -> Dict[str, float]:
    """
    Compute error statistics on a windowed loader, optionally adding noise/bias
    to:
      - the normalized neural-network inputs  (batch["inputs"])
      - the physical ODE quantities           (batch["P_W"], batch["T_bplate"], batch["Tamb"])

    Args:
        trainer    : ODEForwardTrainer with the trained model/parameters.
        loader     : DataLoader returning batches from SyntheticThermalWindowDataset.
        device     : torch device.
        target_key : "T_case_meas" or "T_case_true".
        noise_std  : std of Gaussian noise added to inputs (normalized).
        bias_dict  : optional {channel_index: bias_value} in normalized space,
                     channel index: 0 -> P, 1 -> T_bplate, 2 -> Tamb.

    Returns:
        dict with rmse, mean_abs, max_abs, q95_abs, q99_abs.
    """
    assert target_key in ("T_case_meas", "T_case_true")

    trainer.model.eval()
    errs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Optionally perturb inputs (normalized) and physical ODE channels.
            inputs = batch_device["inputs"]
            P_W = batch_device["P_W"]
            T_bplate = batch_device["T_bplate"]
            Tamb = batch_device["Tamb"]

            if noise_std > 0.0:
                inputs = inputs + noise_std * torch.randn_like(inputs)
                # Noise on normalized NN inputs
                inputs = inputs + noise_std * torch.randn_like(inputs)
                # Noise on physical signals used by the ODE
                P_W = P_W + noise_std * torch.randn_like(P_W)
                T_bplate = T_bplate + noise_std * torch.randn_like(T_bplate)
                Tamb = Tamb + noise_std * torch.randn_like(Tamb)

            if bias_dict:
                for ch, bias in bias_dict.items():
                    inputs[:, :, ch] = inputs[:, :, ch] + float(bias)
                    # Map channel index to physical quantity:
                    #   0 -> P_W, 1 -> T_bplate, 2 -> Tamb
                    if ch == 0:
                        P_W = P_W + float(bias)
                    elif ch == 1:
                        T_bplate = T_bplate + float(bias)
                    elif ch == 2:
                        Tamb = Tamb + float(bias)

            batch_device["inputs"] = inputs
            batch_device["P_W"] = P_W
            batch_device["T_bplate"] = T_bplate
            batch_device["Tamb"] = Tamb

            T_target = batch_device[target_key]
            mask = batch_device["mask"]

            T_pred, _ = trainer.forward_sequence(batch_device)
            e = (T_pred - T_target) * mask
            errs.append(e.detach().cpu().numpy().ravel())

    if not errs:
        return {}

    e_all = np.concatenate(errs)
    return error_stats_from_errors(e_all)


# ---------------------------------------------------------------------
# Full-sequence per-scenario evaluation
# ---------------------------------------------------------------------
def build_full_sequence_batch(
    csv_path: Path,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    device: torch.device,
) -> Tuple[Dict, np.ndarray]:
    """
    Build a single full-sequence batch (B=1) from a scenario CSV file.

    Returns:
      batch dict (on device) and the time axis (numpy array).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    for col in [
        "time_s",
        "P_W",
        "Tamb_C",
        "T_bplate_C",
        "T_case_true_C",
        "T_case_meas_C",
    ]:
        if col not in df.columns:
            raise RuntimeError(f"{csv_path}: column '{col}' not found.")

    t = df["time_s"].to_numpy(dtype=float)
    P = df["P_W"].to_numpy(dtype=float)
    Tamb = df["Tamb_C"].to_numpy(dtype=float)
    Tb = df["T_bplate_C"].to_numpy(dtype=float)
    Ttrue = df["T_case_true_C"].to_numpy(dtype=float)
    Tmeas = df["T_case_meas_C"].to_numpy(dtype=float)

    dt = robust_dt(t)
    Tlen = len(t)

    # Build raw inputs
    inputs_raw = np.zeros((1, Tlen, 3), dtype=float)
    inputs_raw[0, :, 0] = P
    inputs_raw[0, :, 1] = Tb
    inputs_raw[0, :, 2] = Tamb

    # Normalize with training stats
    x = inputs_raw.copy()
    for ch in range(3):
        nan_mask = ~np.isfinite(x[:, :, ch])
        x[:, :, ch][nan_mask] = input_mean[ch]

    mean = input_mean.reshape(1, 1, 3)
    std = input_std.reshape(1, 1, 3)
    inputs_norm = (x - mean) / std

    batch = {
        "inputs": torch.tensor(inputs_norm, dtype=torch.float32, device=device),
        "T_case_meas": torch.tensor(Tmeas.reshape(1, -1), dtype=torch.float32, device=device),
        "T_case_true": torch.tensor(Ttrue.reshape(1, -1), dtype=torch.float32, device=device),
        "mask": torch.ones((1, Tlen), dtype=torch.float32, device=device),
        "P_W": torch.tensor(P.reshape(1, -1), dtype=torch.float32, device=device),
        "T_bplate": torch.tensor(Tb.reshape(1, -1), dtype=torch.float32, device=device),
        "Tamb": torch.tensor(Tamb.reshape(1, -1), dtype=torch.float32, device=device),
        "dt": torch.tensor([dt], dtype=torch.float32, device=device),
        "scenario_id": torch.tensor([0], dtype=torch.long, device=device),  # single R_conv
    }
    return batch, t


def full_sequence_stats_for_scenario(
    model: torch.nn.Module,
    R_cond_hat: float,
    C_th_hat: float,
    Rconv_value: float,
    csv_path: Path,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate full-sequence error stats for a *single* scenario CSV.

    Returns:
        stats_meas, stats_true
    """
    batch, _ = build_full_sequence_batch(csv_path, input_mean, input_std, device)

    # Build a "test" trainer wrapper for this single scenario
    test_trainer = ODEForwardTrainer(
        model=model,
        Cth_init=C_th_hat,
        Rconv_list=[Rconv_value],
        device=str(device),
        lambda_u=0.0,       # no regularization in evaluation
        clip_grad=5.0,
        use_nn=True,
        Rcond_init=R_cond_hat,
    )
    test_trainer.model.eval()

    with torch.no_grad():
        T_pred, _ = test_trainer.forward_sequence(batch)

    T_meas = batch["T_case_meas"]
    T_true = batch["T_case_true"]
    mask = batch["mask"]

    err_meas = ((T_pred - T_meas) * mask).detach().cpu().numpy().ravel()
    err_true = ((T_pred - T_true) * mask).detach().cpu().numpy().ravel()

    stats_meas = error_stats_from_errors(err_meas)
    stats_true = error_stats_from_errors(err_true)
    return stats_meas, stats_true


# ---------------------------------------------------------------------
# Cut & restart test
# ---------------------------------------------------------------------
def cut_and_restart_test(
    model: torch.nn.Module,
    R_cond_hat: float,
    C_th_hat: float,
    Rconv_value: float,
    csv_path: Path,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    device: torch.device,
    split_ratio: float = 0.5,
    init_mode: str = "meas",
    init_offset: float = 0.0,
) -> Dict[str, float]:
    """
    "Cut & restart" consistency / robustness test on a single scenario.

    init_mode:  "pred" -> restart from T_pred_full[k_split]
                "meas" -> restart from T_meas[k_split]

    Procedure:
      1) Run the model on the full sequence -> T_pred_full.
      2) Choose a split index k_split = int(Tlen * split_ratio).
      3) Build a second batch for the subsequence [k_split, end], but:
           - set T_case_meas[0] = T_pred_full[k_split]
           - run forward again -> T_pred_restart
      4) Compare T_pred_restart vs T_pred_full[k_split:] to see how much
         error is introduced by restarting integration from model predictions.

    Returns:
        dict with max_abs_delta, q95_abs_delta, q99_abs_delta.
    """
    batch_full, t_axis = build_full_sequence_batch(csv_path, input_mean, input_std, device)
    Tlen = batch_full["T_case_meas"].shape[1]
    if Tlen < 10:
        raise RuntimeError("Sequence too short for cut & restart test.")

    k_split = int(Tlen * split_ratio)
    k_split = max(1, min(Tlen - 2, k_split))

    # Trainer for this scenario
    trainer_full = ODEForwardTrainer(
        model=model,
        Cth_init=C_th_hat,
        Rconv_list=[Rconv_value],
        device=str(device),
        lambda_u=0.0,
        clip_grad=5.0,
        use_nn=True,
        Rcond_init=R_cond_hat,
    )
    trainer_full.model.eval()

    with torch.no_grad():
        T_pred_full, _ = trainer_full.forward_sequence(batch_full)

    # Build subsequence batch [k_split, end]
    T_meas_full = batch_full["T_case_meas"]
    T_true_full = batch_full["T_case_true"]
    P_full = batch_full["P_W"]
    Tb_full = batch_full["T_bplate"]
    Tamb_full = batch_full["Tamb"]
    dt_full = batch_full["dt"]

    T_meas_seg = T_meas_full[:, k_split:].clone()
    T_true_seg = T_true_full[:, k_split:].clone()
    P_seg = P_full[:, k_split:].clone()
    Tb_seg = Tb_full[:, k_split:].clone()
    Tamb_seg = Tamb_full[:, k_split:].clone()
    Tlen_seg = T_meas_seg.shape[1]

    # Choose initial temperature for the restart:
    #   - "pred": use the model prediction at the split (previous behavior)
    #   - "meas": use the measured value at the split (more realistic reset)
    # Then apply an optional offset (e.g., to simulate a calibration error).
    if init_mode == "pred":
        base_T0 = T_pred_full[:, k_split]
    elif init_mode == "meas":
        base_T0 = T_meas_full[:, k_split]
    else:
        raise ValueError(f"Unknown init_mode='{init_mode}', expected 'pred' or 'meas'")

    T_meas_seg[:, 0] = base_T0 + float(init_offset)

    # Rebuild INPUTS (normalized) for the segment.
    # We assume the normalization used in build_full_sequence_batch
    # already produced consistent inputs, so we can just slice.
    inputs_seg = batch_full["inputs"][:, k_split:, :].clone()

    batch_seg = {
        "inputs": inputs_seg,
        "T_case_meas": T_meas_seg,
        "T_case_true": T_true_seg,
        "mask": torch.ones((1, Tlen_seg), dtype=torch.float32, device=device),
        "P_W": P_seg,
        "T_bplate": Tb_seg,
        "Tamb": Tamb_seg,
        "dt": dt_full.clone(),
        "scenario_id": torch.tensor([0], dtype=torch.long, device=device),
    }

    trainer_restart = ODEForwardTrainer(
        model=model,
        Cth_init=C_th_hat,
        Rconv_list=[Rconv_value],
        device=str(device),
        lambda_u=0.0,
        clip_grad=5.0,
        use_nn=True,
        Rcond_init=R_cond_hat,
    )
    trainer_restart.model.eval()

    with torch.no_grad():
        T_pred_seg, _ = trainer_restart.forward_sequence(batch_seg)

    # Compare restart predictions with original full prediction from k_split onward
    T_pred_full_tail = T_pred_full[:, k_split:].detach().cpu().numpy().ravel()
    T_pred_seg_np = T_pred_seg.detach().cpu().numpy().ravel()

    delta = T_pred_seg_np - T_pred_full_tail
    abs_delta = np.abs(delta)

    return {
        "max_abs_delta": float(np.max(abs_delta)),
        "q95_abs_delta": float(np.quantile(abs_delta, 0.95)),
        "q99_abs_delta": float(np.quantile(abs_delta, 0.99)),
    }


# ---------------------------------------------------------------------
# Dataset and model reconstruction
# ---------------------------------------------------------------------
def build_window_dataset_and_loader(
    data_dir: Path,
    scenarios_meta: List[Dict],
    test_scen_id: int,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    window: int,
    batch_size: int,
) -> Tuple[SyntheticThermalWindowDataset, DataLoader]:
    """
    Build a windowed dataset and loader on all *training* scenarios
    (i.e. all except the test scenario), using the normalization from
    the checkpoint (input_mean, input_std).
    """
    train_scenarios = [s for s in scenarios_meta if s["scenario_id"] != test_scen_id]
    train_csv_files = [s["csv_file"] for s in train_scenarios]
    csv_paths = [str(data_dir / name) for name in train_csv_files]

    ds = SyntheticThermalWindowDataset(
        csv_paths,
        window_size=window,
        stride=max(1, window // 2),
        preload=True,
    )
    # Override normalization with checkpoint stats
    ds.input_mean = input_mean.astype(float)
    ds.input_std = input_std.astype(float)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ds.collate_fn,
    )
    return ds, loader


def build_model_from_ckpt(arch: str, hidden: int, ckpt: Dict) -> torch.nn.Module:
    """
    Rebuild the neural model (MLP/TCN/LSTM) and load weights from checkpoint.
    """
    if arch == "mlp":
        model = MLPRegressor(in_dim=3, hidden=hidden)
    elif arch == "tcn":
        model = TCNRegressor(in_dim=3, hidden=hidden, n_blocks=3, kernel_size=3)
    elif arch == "lstm":
        model = LSTMRegressor(in_dim=3, hidden=hidden, n_layers=1, dropout=0.0)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    state_dict = ckpt["model_state"]
    model.load_state_dict(state_dict)
    return model


# ---------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Automotive-oriented test suite for trained thermal PINN models."
    )
    ap.add_argument("--data-dir", required=True, help="Directory with scenario_*.csv and true_params.json")
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint (*.pt)")
    ap.add_argument("--arch", choices=["mlp", "tcn", "lstm"], required=False, help="Model architecture (if omitted, inferred from checkpoint)")
    ap.add_argument("--hidden", type=int, default=8, help="Hidden width used in training")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0, etc.")
    ap.add_argument("--window", type=int, default=512, help="Window length (samples) for windowed tests")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for windowed tests")

    # Noise / bias test hyperparams (on normalized inputs)
    ap.add_argument("--noise-std", type=float, default=0.1, help="Std of Gaussian noise on normalized inputs for noise robustness test")
    ap.add_argument("--bias-amb", type=float, default=0.0, help="Bias added to normalized ambient temperature (channel 2)")
    ap.add_argument("--bias-bplate", type=float, default=0.0, help="Bias added to normalized baseplate temperature (channel 1)")
    ap.add_argument("--cut-restart-split", type=float, default=0.5, help="Relative split point for cut & restart test (0..1)")
    ap.add_argument("--cut-restart-init", choices=["pred", "meas"], default="meas", help="Initial condition for restart: 'pred' uses T_pred_full[k_split], 'meas' uses T_meas[k_split]")
    ap.add_argument("--cut-restart-offset", type=float, default=0.0, help="Additive offset [°C] applied to the restart initial temperature")
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)

    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)

    arch_ckpt = ckpt.get("arch", None)
    arch = args.arch or arch_ckpt
    if arch is None:
        raise SystemExit("Architecture not specified and not found in checkpoint. Use --arch.")

    # Load true_params.json
    json_path = data_dir / "true_params.json"
    if not json_path.exists():
        raise SystemExit(f"{json_path} not found.")
    with open(json_path, "r") as f:
        meta = json.load(f)

    Cth_true = float(meta["C_th_true"])
    Rcond_true = float(meta["R_cond_true"])
    scenarios_meta = sorted(meta["scenarios"], key=lambda s: s["scenario_id"])

    test_scen_id = int(ckpt["test_scen_id"])
    Rconv_scale = float(ckpt.get("Rconv_scale", 1.0))

    # Build mapping scenario_id -> (csv_file, Rconv_true, Rconv_used)
    scen_info = {}
    for s in scenarios_meta:
        sid = int(s["scenario_id"])
        Rconv_true_s = float(s["R_conv_true"])
        scen_info[sid] = {
            "csv_file": s["csv_file"],
            "Rconv_true": Rconv_true_s,
            "Rconv_used": Rconv_true_s * Rconv_scale,
        }

    print("=== Loaded checkpoint ===")
    print(f"  ckpt path     : {ckpt_path}")
    print(f"  arch          : {arch}")
    print(f"  C_th_true     : {Cth_true:.4f} J/K")
    print(f"  R_cond_true   : {Rcond_true:.4f} K/W")
    print(f"  test_scen_id  : {test_scen_id}")
    print(f"  Rconv_scale   : {Rconv_scale:.4f}")

    # Rebuild model and physical parameters
    model = build_model_from_ckpt(arch, args.hidden, ckpt).to(device)
    log_Rcond = np.array(ckpt["log_Rcond"], dtype=float)
    log_Cth = np.array(ckpt["log_Cth"], dtype=float)
    R_cond_hat = float(np.exp(log_Rcond))
    C_th_hat = float(np.exp(log_Cth))

    print("\n=== Estimated physical parameters (from checkpoint) ===")
    print(f"  R_cond_hat = {R_cond_hat:.4f} K/W")
    print(f"  C_th_hat   = {C_th_hat:.4f} J/K")

    # Load normalization stats
    stats = ckpt["dataset_stats"]
    input_mean = np.array(stats["input_mean"], dtype=float)
    input_std = np.array(stats["input_std"], dtype=float)

    print("\n=== Normalization stats (from training) ===")
    print(f"  input_mean = {input_mean}")
    print(f"  input_std  = {input_std}")

    # Build windowed dataset/loader on training scenarios
    ds, loader = build_window_dataset_and_loader(
        data_dir=data_dir,
        scenarios_meta=scenarios_meta,
        test_scen_id=test_scen_id,
        input_mean=input_mean,
        input_std=input_std,
        window=args.window,
        batch_size=args.batch,
    )

    # Trainer for windowed tests (use same Rconv list as training would)
    train_scenarios = [s for s in scenarios_meta if s["scenario_id"] != test_scen_id]
    Rconv_train_list = [float(s["R_conv_true"]) * Rconv_scale for s in train_scenarios]

    trainer = ODEForwardTrainer(
        model=model,
        Cth_init=C_th_hat,
        Rconv_list=Rconv_train_list,
        device=str(device),
        lambda_u=0.0,  # regularization not needed for evaluation
        clip_grad=5.0,
        use_nn=True,
        Rcond_init=R_cond_hat,
    )

    # -----------------------------------------------------------------
    # 1) Per-scenario full-sequence metrics
    # -----------------------------------------------------------------
    print("\n=== [1] Per-scenario FULL-SEQUENCE metrics ===")
    for sid, info in scen_info.items():
        csv_path = data_dir / info["csv_file"]
        stats_meas, stats_true = full_sequence_stats_for_scenario(
            model=model,
            R_cond_hat=R_cond_hat,
            C_th_hat=C_th_hat,
            Rconv_value=info["Rconv_used"],
            csv_path=csv_path,
            input_mean=input_mean,
            input_std=input_std,
            device=device,
        )
        flag = " (TEST)" if sid == test_scen_id else ""
        print(f"\nScenario {sid}{flag}: file={info['csv_file']}")
        print(f"  vs T_case_meas: {stats_meas}")
        print(f"  vs T_case_true: {stats_true}")

    # -----------------------------------------------------------------
    # 2) Noise robustness test (windowed)
    # -----------------------------------------------------------------
    print("\n=== [2] Sensor noise robustness (windowed) ===")
    baseline_stats = error_stats_over_loader(
        trainer, loader, device, target_key="T_case_meas", noise_std=0.0, bias_dict=None
    )
    noise_stats = error_stats_over_loader(
        trainer, loader, device, target_key="T_case_meas", noise_std=args.noise_std, bias_dict=None
    )
    print(f"  Baseline (no noise): {baseline_stats}")
    print(f"  With noise (std={args.noise_std} on normalized inputs): {noise_stats}")

    # -----------------------------------------------------------------
    # 3) Bias robustness test (windowed)
    # -----------------------------------------------------------------
    print("\n=== [3] Sensor bias robustness (windowed) ===")
    bias_dict = {}
    if abs(args.bias_bplate) > 0.0:
        bias_dict[1] = args.bias_bplate  # channel 1: T_bplate
    if abs(args.bias_amb) > 0.0:
        bias_dict[2] = args.bias_amb     # channel 2: Tamb

    if bias_dict:
        bias_stats = error_stats_over_loader(
            trainer, loader, device, target_key="T_case_meas", noise_std=0.0, bias_dict=bias_dict
        )
        print(f"  Bias (normalized) applied to channels {bias_dict}: {bias_stats}")
    else:
        print("  No bias specified (use --bias-amb / --bias-bplate to enable).")

    # -----------------------------------------------------------------
    # 4) Cut & restart test on test scenario (single configuration)
    # -----------------------------------------------------------------
    print("\n=== [4] Cut & restart test on TEST scenario ===")
    test_csv_path = data_dir / scen_info[test_scen_id]["csv_file"]
    cut_stats = cut_and_restart_test(
        model=model,
        R_cond_hat=R_cond_hat,
        C_th_hat=C_th_hat,
        Rconv_value=scen_info[test_scen_id]["Rconv_used"],
        csv_path=test_csv_path,
        input_mean=input_mean,
        input_std=input_std,
        device=device,
        split_ratio=args.cut_restart_split,
        init_mode=args.cut_restart_init,
        init_offset=args.cut_restart_offset,
    )
    print(
        f"  Cut & restart stats (split={args.cut_restart_split:.2f}, "
        f"init={args.cut_restart_init}, offset={args.cut_restart_offset:.2f} °C): "
        f"{cut_stats}"
    )

    # -----------------------------------------------------------------
    # 4b) Cut & restart sweeps: different splits and offsets
    # -----------------------------------------------------------------
    print("\n=== [4b] Cut & restart sweeps (TEST scenario, init=meas) ===")
    split_values = [0.25, 0.50, 0.75]
    offset_values = [0.10, 0.50, 1.00]

    for split in split_values:
        for offset in offset_values:
            sweep_stats = cut_and_restart_test(
                model=model,
                R_cond_hat=R_cond_hat,
                C_th_hat=C_th_hat,
                Rconv_value=scen_info[test_scen_id]["Rconv_used"],
                csv_path=test_csv_path,
                input_mean=input_mean,
                input_std=input_std,
                device=device,
                split_ratio=split,
                init_mode="meas",
                init_offset=offset,
            )
            print(
                f"  split={split:.2f}, init=meas, offset={offset:.2f} °C -> "
                f"{sweep_stats}"
            )

    print("\nAll automotive-oriented tests completed.")


if __name__ == "__main__":
    main()
