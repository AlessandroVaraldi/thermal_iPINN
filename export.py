#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_ipinn_to_c.py

Export a trained PINN (MLP / TCN / LSTM) + physical parameters to a C header.

It expects checkpoints saved by train_ipinn_synthetic_odeforward.py, i.e.:
  {
    "arch": "mlp" | "tcn" | "lstm",
    "model_state": state_dict,
    "log_Rcond": ...,
    "log_Cth": ...,
    "dataset_stats": {
        "input_mean": [3],
        "input_std": [3],
    },
    "R_conv_train_list": [...],         # optional, from training
    "R_conv_test_true": float,          # optional
    "R_conv_test_used": float,          # optional
    ...
  }

The script generates:
  - <arch>_export.h  : C header with weights, stats, physical params, LUTs,
                       and (optionally) a golden test vector.

All arrays are exported as 1D float arrays with separate dimension metadata.
"""

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import torch

# Import your model definitions from the training script
from train import (
    MLPRegressor,
    TCNRegressor,
    LSTMRegressor,
)


# ---------------------------------------------------------------------
# Small helper: robust dt (duplicated from train script so this is standalone)
# ---------------------------------------------------------------------
def robust_dt(seconds: np.ndarray) -> float:
    """Robust estimate of dt (median between p10 and p90 of positive deltas)."""
    ds = np.diff(seconds.astype(float))
    ds = ds[np.isfinite(ds) & (ds > 0)]
    if ds.size == 0:
        return 0.1
    p10, p90 = np.percentile(ds, [10, 90])
    core = ds[(ds >= p10) & (ds <= p90)]
    return float(np.median(core if core.size else ds))


# ---------------------------------------------------------------------
# Rebuild model from checkpoint
# ---------------------------------------------------------------------
def build_model_from_state_dict(arch: str, state_dict: dict) -> torch.nn.Module:
    """Reconstruct the PyTorch model using only the state_dict shapes."""
    if arch == "mlp":
        w0 = state_dict["net.0.weight"]
        hidden = w0.shape[0]
        in_dim = w0.shape[1]
        model = MLPRegressor(in_dim=in_dim, hidden=hidden)

    elif arch == "tcn":
        first_w = state_dict["tcn.0.conv.weight"]
        in_dim = first_w.shape[1]
        hidden = first_w.shape[0]
        kernel_size = first_w.shape[2]
        # Count how many TCN blocks we have
        n_blocks = len(
            [k for k in state_dict.keys()
             if k.startswith("tcn.") and k.endswith(".conv.weight")]
        )
        model = TCNRegressor(in_dim=in_dim,
                             hidden=hidden,
                             n_blocks=n_blocks,
                             kernel_size=kernel_size)

    elif arch == "lstm":
        w_ih = state_dict["lstm.weight_ih_l0"]
        hidden = w_ih.shape[0] // 4
        in_dim = w_ih.shape[1]
        # Number of layers: count weight_ih_lX
        num_layers = len(
            [k for k in state_dict.keys()
             if k.startswith("lstm.weight_ih_l")]
        )
        model = LSTMRegressor(in_dim=in_dim,
                              hidden=hidden,
                              n_layers=num_layers,
                              dropout=0.0)  # dropout irrelevant in inference

    else:
        raise ValueError(f"Unsupported arch: {arch}")

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------
# C formatting helpers
# ---------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    """Convert PyTorch param name to a valid C identifier."""
    for ch in [".", ":", "/", "-"]:
        name = name.replace(ch, "_")
    return name


def format_c_float_array(name: str, arr: np.ndarray) -> str:
    """
    Format a 1D float32 numpy array into C code:
        static const float name[N] = { ... };
    The array is flattened before formatting.
    """
    arr = arr.astype(np.float32).reshape(-1)
    size = arr.size

    # Format values with scientific notation
    vals = [f"{float(v):.8e}f" for v in arr]

    # Wrap lines nicely
    line = ", ".join(vals)
    wrapped = "\n    ".join(textwrap.wrap(line, width=80))

    code = (
        f"static const int {name}_SIZE = {size};\n"
        f"static const float {name}[{size}] = {{\n"
        f"    {wrapped}\n"
        f"}};\n"
    )
    return code


def format_c_int_array(name: str, arr: np.ndarray) -> str:
    """Format a 1D int array as C code."""
    arr = np.asarray(arr, dtype=np.int32).reshape(-1)
    size = arr.size
    vals = ", ".join(str(int(v)) for v in arr)
    code = (
        f"static const int {name}_SIZE = {size};\n"
        f"static const int {name}[{size}] = {{ {vals} }};\n"
    )
    return code


def format_param_block(arch: str, param_name: str, tensor: torch.Tensor) -> str:
    """
    Generate C code for a single parameter tensor:
      - dims metadata
      - flattened float array
    """
    cname = sanitize_name(f"{arch}_{param_name}")
    dims = np.array(list(tensor.shape), dtype=np.int32)
    arr = tensor.detach().cpu().numpy().astype(np.float32)

    dims_name = f"{cname}_DIMS"
    data_name = f"{cname}_DATA"

    code = ""
    code += format_c_int_array(dims_name, dims)
    code += "\n"
    code += format_c_float_array(data_name, arr)
    code += "\n"
    return code


# ---------------------------------------------------------------------
# LUT generation for sigmoid / tanh (for LSTM)
# ---------------------------------------------------------------------
def generate_lut(func_name: str, x_min: float, x_max: float, n: int) -> str:
    """Generate C code for LUT of sigmoid or tanh."""
    xs = np.linspace(x_min, x_max, n, dtype=np.float32)
    if func_name == "sigmoid":
        ys = 1.0 / (1.0 + np.exp(-xs))
    elif func_name == "tanh":
        ys = np.tanh(xs)
    else:
        raise ValueError(f"Unsupported LUT func: {func_name}")

    arr_name = f"LUT_{func_name.upper()}"
    xs_name = f"LUT_{func_name.upper()}_X"

    code = ""
    # Metadata
    code += f"static const float {arr_name}_X_MIN = {x_min:.8e}f;\n"
    code += f"static const float {arr_name}_X_MAX = {x_max:.8e}f;\n\n"

    # X grid
    code += format_c_float_array(xs_name, xs)
    code += "\n"
    # Y values
    code += format_c_float_array(arr_name, ys)
    code += "\n"

    return code


# ---------------------------------------------------------------------
# Golden vector generation
# ---------------------------------------------------------------------
def run_golden_forward(
    model: torch.nn.Module,
    ckpt: dict,
    golden_csv: Path,
    golden_len: int,
    golden_rconv: float,
    use_nn: bool = True,
):
    """
    Run a full ODE forward pass on one scenario and return
    (t, P, Tb, Tamb, T_meas, T_pred, dt) as numpy arrays.

    The ODE is re-implemented here for B=1.
    """
    # --- Load CSV ---
    df = pd.read_csv(golden_csv)
    for col in ["time_s", "P_W", "Tamb_C", "T_bplate_C", "T_case_meas_C"]:
        if col not in df.columns:
            raise RuntimeError(f"{golden_csv}: column '{col}' not found.")

    t = df["time_s"].to_numpy(dtype=np.float32)
    P = df["P_W"].to_numpy(dtype=np.float32)
    Tamb = df["Tamb_C"].to_numpy(dtype=np.float32)
    Tb = df["T_bplate_C"].to_numpy(dtype=np.float32)
    Tmeas = df["T_case_meas_C"].to_numpy(dtype=np.float32)

    dt = robust_dt(t)

    if golden_len > 0:
        L = min(golden_len, len(t))
        t = t[:L]
        P = P[:L]
        Tamb = Tamb[:L]
        Tb = Tb[:L]
        Tmeas = Tmeas[:L]
    else:
        L = len(t)

    # --- Normalization using stats stored in checkpoint ---
    stats = ckpt["dataset_stats"]
    input_mean = np.array(stats["input_mean"], dtype=np.float32)  # (3,)
    input_std = np.array(stats["input_std"], dtype=np.float32)    # (3,)

    inputs_raw = np.zeros((L, 3), dtype=np.float32)
    inputs_raw[:, 0] = P
    inputs_raw[:, 1] = Tb
    inputs_raw[:, 2] = Tamb

    x = inputs_raw.copy()
    for ch in range(3):
        nan_mask = ~np.isfinite(x[:, ch])
        x[nan_mask, ch] = input_mean[ch]
    inputs_norm = (x - input_mean.reshape(1, 3)) / input_std.reshape(1, 3)

    # --- Physical parameters from checkpoint ---
    log_Rcond = np.array(ckpt["log_Rcond"], dtype=np.float32).reshape(-1)[0]
    log_Cth = np.array(ckpt["log_Cth"], dtype=np.float32).reshape(-1)[0]
    Rcond = float(np.exp(log_Rcond))
    Cth = float(np.exp(log_Cth))
    Rconv = float(golden_rconv)

    # --- ODE forward, B=1, in torch ---
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(inputs_norm, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,3)
        P_t = torch.tensor(P, dtype=torch.float32, device=device).unsqueeze(0)            # (1,T)
        Tb_t = torch.tensor(Tb, dtype=torch.float32, device=device).unsqueeze(0)          # (1,T)
        Tamb_t = torch.tensor(Tamb, dtype=torch.float32, device=device).unsqueeze(0)      # (1,T)
        Tmeas_t = torch.tensor(Tmeas, dtype=torch.float32, device=device).unsqueeze(0)    # (1,T)

        if use_nn:
            u = model(X_t)  # (1,T)
        else:
            u = torch.zeros_like(P_t)

        T_pred = torch.zeros_like(P_t)
        # Initial condition = first measured temperature
        T_pred[:, 0] = Tmeas_t[:, 0]

        Cth_t = torch.tensor(Cth, dtype=torch.float32, device=device)
        Rcond_t = torch.tensor(Rcond, dtype=torch.float32, device=device)
        Rconv_t = torch.tensor(Rconv, dtype=torch.float32, device=device)
        dt_t = torch.tensor(dt, dtype=torch.float32, device=device)

        T = L
        for k in range(T - 1):
            Tk = T_pred[:, k:k+1]
            Pk = P_t[:, k:k+1]
            Tb_k = Tb_t[:, k:k+1]
            Tamb_k = Tamb_t[:, k:k+1]
            uk = u[:, k:k+1]

            dT = (dt_t / Cth_t) * (
                Pk + uk
                - (Tk - Tb_k) / Rcond_t
                - (Tk - Tamb_k) / Rconv_t
            )
            T_pred[:, k + 1:k + 2] = Tk + dT

        T_pred_np = T_pred[0].cpu().numpy()

    return t, P, Tb, Tamb, Tmeas, T_pred_np, dt


# ---------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------
def export_to_c(
    ckpt_path: Path,
    outdir: Path,
    golden_csv: Path = None,
    golden_len: int = -1,
    golden_rconv: float = None,
    lut_size: int = 2048,
    lut_xmin: float = -8.0,
    lut_xmax: float = 8.0,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    arch = ckpt.get("arch", None)
    if arch is None:
        raise RuntimeError("Checkpoint does not contain 'arch' field.")

    state_dict = ckpt["model_state"]
    model = build_model_from_state_dict(arch, state_dict)

    # Physical parameters
    log_Rcond = np.array(ckpt["log_Rcond"], dtype=np.float32).reshape(-1)[0]
    log_Cth = np.array(ckpt["log_Cth"], dtype=np.float32).reshape(-1)[0]
    Rcond = float(np.exp(log_Rcond))
    Cth = float(np.exp(log_Cth))

    stats = ckpt["dataset_stats"]
    input_mean = np.array(stats["input_mean"], dtype=np.float32)
    input_std = np.array(stats["input_std"], dtype=np.float32)

    outdir.mkdir(parents=True, exist_ok=True)
    header_path = outdir / f"{arch}_export.h"

    with open(header_path, "w") as f:
        f.write("// Auto-generated by export_ipinn_to_c.py\n")
        f.write("// Do NOT edit by hand.\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n\n")

        # ---- Basic metadata ----
        f.write("/* Model architecture: mlp=0, tcn=1, lstm=2 */\n")
        if arch == "mlp":
            arch_id = 0
        elif arch == "tcn":
            arch_id = 1
        elif arch == "lstm":
            arch_id = 2
        else:
            arch_id = -1

        f.write(f"static const int IPINN_ARCH_ID = {arch_id};\n")
        f.write(f"// IPINN_ARCH_ID corresponds to '{arch}'\n\n")

        # ---- Normalization stats ----
        f.write("/* Input normalization: x_norm = (x - mean) / std */\n")
        f.write(format_c_float_array("IPINN_INPUT_MEAN", input_mean))
        f.write("\n")
        f.write(format_c_float_array("IPINN_INPUT_STD", input_std))
        f.write("\n")

        # ---- Physical parameters ----
        f.write("/* Physical parameters (learned) */\n")
        f.write(f"static const float IPINN_RCOND = {Rcond:.8e}f; // [K/W]\n")
        f.write(f"static const float IPINN_CTH   = {Cth:.8e}f; // [J/K]\n\n")

        # Optional stored R_conv lists (if present in checkpoint)
        if "R_conv_train_list" in ckpt:
            Rconv_train_list = np.array(ckpt["R_conv_train_list"], dtype=np.float32)
            f.write("/* R_conv used during training (per scenario) */\n")
            f.write(format_c_float_array("IPINN_RCONV_TRAIN_LIST", Rconv_train_list))
            f.write("\n")

        if "R_conv_test_used" in ckpt:
            Rconv_test_used = float(ckpt["R_conv_test_used"])
            f.write("/* R_conv used for held-out test scenario in training script */\n")
            f.write(
                f"static const float IPINN_RCONV_TEST_USED = "
                f"{Rconv_test_used:.8e}f; // [K/W]\n\n"
            )

        # ---- Network parameters ----
        f.write("/* Network parameters (weights and biases) */\n\n")
        for name, tensor in state_dict.items():
            code = format_param_block(arch, name, tensor)
            f.write(code)
            f.write("\n")

        # ---- LUTs for activations ----
        f.write("/* Lookup tables for activation functions (float32) */\n\n")
        f.write("// LUT for sigmoid in range [lut_xmin, lut_xmax]\n")
        f.write(generate_lut("sigmoid", lut_xmin, lut_xmax, lut_size))
        f.write("\n")
        f.write("// LUT for tanh in range [lut_xmin, lut_xmax]\n")
        f.write(generate_lut("tanh", lut_xmin, lut_xmax, lut_size))
        f.write("\n")

        # ---- Golden test vector ----
        if golden_csv is not None:
            if golden_rconv is None:
                golden_rconv = ckpt.get("R_conv_test_used", None)
                if golden_rconv is None:
                    raise RuntimeError(
                        "You requested a golden vector but did not provide "
                        "--golden-rconv and checkpoint has no 'R_conv_test_used'."
                    )

            f.write("/* Golden test vector (one scenario) */\n")
            f.write("/* Use this to verify your C implementation. */\n\n")

            t, P, Tb, Tamb, Tmeas, Tpred, dt = run_golden_forward(
                model=model,
                ckpt=ckpt,
                golden_csv=golden_csv,
                golden_len=golden_len,
                golden_rconv=golden_rconv,
                use_nn=True,
            )

            f.write(f"static const float GOLDEN_DT = {dt:.8e}f; // [s]\n")
            f.write(f"static const int   GOLDEN_LEN = {len(t)};\n\n")

            f.write(format_c_float_array("GOLDEN_TIME", t))
            f.write("\n")
            f.write(format_c_float_array("GOLDEN_P_W", P))
            f.write("\n")
            f.write(format_c_float_array("GOLDEN_T_BPLATE", Tb))
            f.write("\n")
            f.write(format_c_float_array("GOLDEN_T_AMB", Tamb))
            f.write("\n")
            f.write(format_c_float_array("GOLDEN_T_MEAS", Tmeas))
            f.write("\n")
            f.write(format_c_float_array("GOLDEN_T_PRED", Tpred))
            f.write("\n")

        else:
            f.write("/* No golden test vector generated (no --golden-csv provided). */\n")

    print(f"[OK] C header written to: {header_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Export trained IPINN (MLP/TCN/LSTM) to a C header file."
    )
    ap.add_argument(
        "--checkpoint", required=True,
        help="Path to *_final.pt checkpoint produced by train script."
    )
    ap.add_argument(
        "--outdir", default="export_c",
        help="Output directory for the generated header."
    )

    # Golden vector options
    ap.add_argument(
        "--golden-csv", default=None,
        help="Path to a scenario_XXX.csv file for golden test vector. "
             "If omitted, no golden vector is generated."
    )
    ap.add_argument(
        "--golden-len", type=int, default=-1,
        help="Number of samples to use from golden CSV (<=0 = full sequence)."
    )
    ap.add_argument(
        "--golden-rconv", type=float, default=None,
        help="R_conv [K/W] to use for the golden scenario. "
             "If omitted, tries 'R_conv_test_used' from checkpoint."
    )

    # LUT options
    ap.add_argument(
        "--lut-size", type=int, default=2048,
        help="Number of points for tanh/sigmoid LUTs."
    )
    ap.add_argument(
        "--lut-xmin", type=float, default=-8.0,
        help="Minimum x for tanh/sigmoid LUT."
    )
    ap.add_argument(
        "--lut-xmax", type=float, default=8.0,
        help="Maximum x for tanh/sigmoid LUT."
    )

    return ap.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    outdir = Path(args.outdir)

    golden_csv = Path(args.golden_csv) if args.golden_csv is not None else None

    export_to_c(
        ckpt_path=ckpt_path,
        outdir=outdir,
        golden_csv=golden_csv,
        golden_len=args.golden_len,
        golden_rconv=args.golden_rconv,
        lut_size=args.lut_size,
        lut_xmin=args.lut_xmin,
        lut_xmax=args.lut_xmax,
    )


if __name__ == "__main__":
    main()
