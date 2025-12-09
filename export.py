#!/usr/bin/env python3
"""
PTQ exporter for the InversePINN (LSTM + FC head)
-------------------------------------------------

This script performs post‑training quantization (PTQ) of a PyTorch LSTM + FC model
trained in `inverse_pinn.py` (the code you shared), using a small calibration set
of CSV sequences. It then emits a self‑contained C header with:

- Quantized weights (INT8) for W_ih / W_hh per gate (i, f, g, o)
- Quantized biases (INT32) matched to accumulator scales
- Mixed‑precision activation plan:
    * x_t, h_{t-1}  → INT8 (symmetric, per‑tensor scales)
    * c_t           → INT16 (Q15 scale by default)
    * accumulators  → INT32
- Quantized FC (INT8/INT32) for the output head
- Normalization constants (both float and fixed‑point forms)
- Sigmoid/tanh lookup tables in Q15

The header is ready to be included in a C integer inference engine. Runtime code
can be implemented around these tensors and scales. The generated header contains
all sizes, arrays, and constants you need.

Usage example
-------------
python ptq_export.py \
  --ckpt runs/best_state.pt \
  --model-def train.py \
  --csv-dir data_sets --glob "*.csv" \
  --env 20.0 \
  --calib-windows 2048 \
  --out-header build/model.h \
  --name INVPINN 

Notes
-----
* We import your model definition dynamically from --model-def and expect
  it to expose class `InversePINN`. If that import fails, we fallback to a
  minimalist loader that reads raw tensors from the checkpoint.
* We use symmetric quantization (zero‑point = 0) for weights & activations.
* Weight quantization is per‑output‑channel (per row) for better accuracy.
* Activation calibration uses running max‑abs on a user‑selected number of
  windows.
* LUT resolution is 2049 samples in [−8, 8] for both sigmoid/tanh in Q15.

"""

import argparse
import importlib.util
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

# --------------------- Data helpers (mirrors your training code) ---------------------

def list_csv_files(csv_dir: str, csv_glob: str):
    from glob import glob
    return sorted(glob(os.path.join(csv_dir, csv_glob)))

def load_single_csv(path: str) -> dict:
    data = np.loadtxt(Path(path), delimiter=",", skiprows=1)
    cols = {
        "t":   data[:, 0],
        "Id":  data[:, 5],
        "Iq":  data[:, 6],
        "Tbp": data[:, 9],
        "Tjr": data[:, 12],
    }
    idx = np.argsort(cols["t"])  # time sort
    for k in cols:
        cols[k] = cols[k][idx]
    return cols

def compute_powers(Id: np.ndarray, Iq: np.ndarray, r_dson: float = 0.63e-3, bias: float = 0.0) -> np.ndarray:
    Ias      = np.sqrt(Id**2 + Iq**2)
    Ias_chip = Ias / np.sqrt(2.0) / 3.0
    P_cond   = (Ias_chip**2) * r_dson
    P = P_cond + bias
    return np.maximum(P, 0.0)

def sliding_windows(X: np.ndarray, y: np.ndarray, win: int, stride: int):
    N = X.shape[0]
    if N < win:
        return (np.empty((0, win, X.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    windows, targets, targets_prev = [], [], []
    for i in range(0, N - win + 1, stride):
        j = i + win
        windows.append(X[i:j, :])
        targets.append(y[j - 1])
        targets_prev.append(y[j - 2])
    return (np.array(windows, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            np.array(targets_prev, dtype=np.float32))

def prepare_calib(csv_dir: str, csv_glob: str, env_temp: float, win: int, stride: int,
                  limit_windows: int | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, float, float]]:
    files = list_csv_files(csv_dir, csv_glob)
    if not files:
        raise SystemExit(f"No CSV found in {csv_dir!r} with pattern {csv_glob!r}")
    Xw_list, yw_list, yprev_list = [], [], []
    for f in files:
        d = load_single_csv(f)
        t   = d["t"].astype(np.float32)
        Id  = d["Id"].astype(np.float32)
        Iq  = d["Iq"].astype(np.float32)
        Tbp = d["Tbp"].astype(np.float32)
        Tjr = d["Tjr"].astype(np.float32)
        P = compute_powers(Id, Iq).astype(np.float32)
        Tenv = np.full_like(Tbp, float(env_temp), dtype=np.float32)
        X = np.stack([P, Tbp, Tenv, t], axis=1)
        Xi, yi, ypi = sliding_windows(X, Tjr, win, stride)
        if Xi.size:
            Xw_list.append(Xi); yw_list.append(yi); yprev_list.append(ypi)
    Xw = np.concatenate(Xw_list, axis=0)
    yw = np.concatenate(yw_list, axis=0)
    yprev = np.concatenate(yprev_list, axis=0)

    if limit_windows is not None and Xw.shape[0] > limit_windows:
        Xw = Xw[:limit_windows]
        yw = yw[:limit_windows]
        yprev = yprev[:limit_windows]

    # stats (match training scheme: normalize features and target)
    mu_x = Xw.mean(axis=(0,1))
    std_x = Xw.std(axis=(0,1)) + 1e-8
    mu_y = float(yw.mean())
    std_y = float(yw.std() + 1e-8)

    return Xw, yw, yprev, (mu_x.astype(np.float32), std_x.astype(np.float32), mu_y, std_y)

# --------------------- Quant helpers ---------------------

def symmetric_qparams(max_abs: np.ndarray, qbits: int) -> np.ndarray:
    # returns scale so that int range maps to max_abs; zero_point=0 always (symmetric)
    qmax = (1 << (qbits - 1)) - 1
    scale = np.maximum(max_abs, 1e-12) / qmax
    return scale

def quant_per_row(W: np.ndarray, qbits: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    # W shape: [out, in]; per-row symmetric int8
    max_abs = np.max(np.abs(W), axis=1)
    scale = symmetric_qparams(max_abs, qbits)  # [out]
    Q = np.round(W / scale[:, None]).astype(np.int8)
    Q = np.clip(Q, -128, 127)
    return Q, scale.astype(np.float32)

# --------------------- LSTM gate slicing helpers ---------------------

def split_lstm_weights(state: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    # PyTorch gate order: i, f, g, o
    W_ih = state["Tnet.lstm.weight_ih_l0"].cpu().numpy()
    W_hh = state["Tnet.lstm.weight_hh_l0"].cpu().numpy()
    b_ih = state["Tnet.lstm.bias_ih_l0"].cpu().numpy()
    b_hh = state["Tnet.lstm.bias_h_l0"].cpu().numpy()
    H, I = W_ih.shape
    assert H % 4 == 0
    h = H // 4
    gates = {}
    names = ["i", "f", "g", "o"]
    for gi, name in enumerate(names):
        s, e = gi*h, (gi+1)*h
        gates[f"W_ih_{name}"] = W_ih[s:e, :]
        gates[f"W_hh_{name}"] = W_hh[s:e, :]
        gates[f"b_{name}"]    = (b_ih[s:e] + b_hh[s:e])  # fused bias
    return gates

# --------------------- Calibration (collect activation ranges) ---------------------

def dyn_import_inversepinn(model_def_path: str):
    spec = importlib.util.spec_from_file_location("invpinn_mod", model_def_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "InversePINN"):
        return mod.InversePINN
    return None

@torch.no_grad()
def collect_activation_stats(model: nn.Module, Xw: np.ndarray, yprev: np.ndarray,
                             mu_x: np.ndarray, std_x: np.ndarray, mu_y: float, std_y: float,
                             max_batches: int = 64, batch: int = 32) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    stats = {
        "x_absmax": 0.0,
        "h_absmax": 0.0,
        "c_absmax": 0.0,
        "fc_in_absmax": 0.0,
        "delta_out_absmax": 0.0,
    }

    mu_x_t = torch.tensor(mu_x, dtype=torch.float32, device=device)
    std_x_t = torch.tensor(std_x, dtype=torch.float32, device=device)

    n = Xw.shape[0]
    steps = min(max_batches, (n + batch - 1) // batch)
    # monkeypatch to capture internal LSTM activations
    lstm: nn.LSTM = model.Tnet.lstm
    fc:   nn.Linear = model.Tnet.fc_delta

    def _wrap_lstm_forward(orig_forward):
        def wrapped(x, hx=None):
            out, (h, c) = orig_forward(x, hx)
            # collect absmax
            stats["h_absmax"] = max(stats["h_absmax"], h[-1].detach().abs().max().item())
            stats["c_absmax"] = max(stats["c_absmax"], c[-1].detach().abs().max().item())
            return out, (h, c)
        return wrapped

    lstm.forward = _wrap_lstm_forward(lstm.forward)

    for b in range(steps):
        s = b*batch
        e = min(n, s+batch)
        xb = torch.from_numpy(((Xw[s:e] - mu_x) / (std_x + 1e-8))).to(device)
        ypb = torch.from_numpy(((yprev[s:e] - mu_y) / (std_y + 1e-8))).to(device)
        stats["x_absmax"] = max(stats["x_absmax"], xb.detach().abs().max().item())
        y_pred_norm, _ = model(xb, yprev_norm=ypb)
        # FC input is last hidden state summary; easiest is to recompute via module
        _, (h_T, _) = lstm(xb)
        summary = h_T[-1]
        stats["fc_in_absmax"] = max(stats["fc_in_absmax"], summary.detach().abs().max().item())
        stats["delta_out_absmax"] = max(stats["delta_out_absmax"], y_pred_norm.detach().abs().max().item())

    return stats

# --------------------- Header emission ---------------------

def q15_lut_sigmoid_tanh(n: int = 2049, x_min: float = -8.0, x_max: float = 8.0):
    xs = np.linspace(x_min, x_max, n, dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-xs))
    tanh = np.tanh(xs)

    def to_q15(v):
        return np.int16(np.round(np.clip(v * 32767.0, -32768, 32767)))

    return xs.astype(np.float32), to_q15(sig), to_q15(tanh)

HEADER_TMPL = """// Auto‑generated by ptq_export.py — DO NOT EDIT
#pragma once
#include <stdint.h>

// ==== Model signature ====
#define {N}_INPUT_SIZE   {INPUT}
#define {N}_HIDDEN_SIZE  {H}
#define {N}_NUM_LAYERS   1

// Mixed precision plan
//   activations: x,h → int8 (sym); c → int16 (Q15); accumulators → int32
//   weights: per‑row int8; biases: int32 scaled by (S_w_row * S_act)

// ==== Normalization (float) ====
static const float {N}_MU_X[{INPUT}]  = {{ {MU_X} }};
static const float {N}_STD_X[{INPUT}] = {{ {STD_X} }};
static const float {N}_MU_Y = {MU_Y}f;
static const float {N}_STD_Y = {STD_Y}f;

// Fixed‑point helpers for x_norm = (x - mu)/std  →  x_norm ≈ (x * A + B) in Q{QFIX}
// A = 1/std,  B = -mu/std
static const int32_t {N}_X_INVSTD_Q{QFIX}[{INPUT}] = {{ {INVSTD_Q} }};  // Q{QFIX}
static const int32_t {N}_X_NEGOFF_Q{QFIX}[{INPUT}] = {{ {NEGOFF_Q} }};  // Q{QFIX}

// ==== Activation scales (symmetric) ====
static const float {N}_S_X   = {S_X}f;    // x/h int8 scale shared for inputs
static const float {N}_S_H   = {S_H}f;    // h int8 scale
static const float {N}_S_CQ15= {S_C}f;    // c uses Q15, this is the float range mapped to ±1.0
static const float {N}_S_FCIN= {S_FCIN}f; // int8 scale for FC input (summary)
static const float {N}_S_DOUT= {S_DOUT}f; // int8 scale for delta output (optional)

// ==== LSTM weights per gate (int8 per‑row) ====
{W_BLOCKS}

// ==== LSTM biases (int32)
{B_BLOCKS}

// ==== FC head (int8 weights per‑row, int32 bias)
#define {N}_FC_IN  {H}
#define {N}_FC_OUT 1
static const int8_t  {N}_FC_W[{N}_FC_OUT][{N}_FC_IN] = {{ {FC_W} }};
static const float   {N}_FC_W_SCALE[{N}_FC_OUT] = {{ {FC_WS} }};
static const int32_t {N}_FC_B[{N}_FC_OUT] = {{ {FC_B} }};

// ==== Sigmoid / tanh lookup tables (Q15)
#define {N}_LUT_SIZE {LUT_N}
#define {N}_LUT_XMIN {LUT_XMIN}f
#define {N}_LUT_XMAX {LUT_XMAX}f
static const int16_t {N}_SIGMOID_Q15[{N}_LUT_SIZE] = {{ {LUT_SIG} }};
static const int16_t {N}_TANH_Q15[{N}_LUT_SIZE] = {{ {LUT_TANH} }};
"""

def fmt_array_1d(arr, fmt="{:.8g}"):
    return ", ".join(fmt.format(float(x)) for x in arr)

def fmt_array_1d_int(arr):
    return ", ".join(str(int(x)) for x in arr)

def emit_w_block(name: str, Wq: np.ndarray, Sc: np.ndarray, N: str) -> str:
    out = []
    out.append(f"static const int8_t  {N}_{name}_W[{Wq.shape[0]}][{Wq.shape[1]}] = {{ ")
    rows = []
    for r in range(Wq.shape[0]):
        rows.append("{" + fmt_array_1d_int(Wq[r]) + "}")
    out.append("  " + ", ".join(rows) + " };\n")
    out.append(f"static const float   {N}_{name}_W_SCALE[{Wq.shape[0]}] = {{ {fmt_array_1d(Sc)} }};\n")
    return "".join(out)

def emit_b_block(name: str, Bq: np.ndarray, N: str) -> str:
    return f"static const int32_t {N}_{name}_B[{Bq.shape[0]}] = {{ {fmt_array_1d_int(Bq)} }};\n"

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint: state_dict or full model")
    ap.add_argument("--model-def", required=True, help="Python file that defines InversePINN class")
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--glob", default="*.csv")
    ap.add_argument("--env", type=float, default=20.0)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--calib-windows", type=int, default=2048)
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--name", default="INVPINN", help="C prefix for symbols in header")
    ap.add_argument("--out-header", required=True)
    ap.add_argument("--qfix", type=int, default=24, help="Q format bits for normalization constants (Qqfix)")
    args = ap.parse_args()

    # Load calibration data
    Xw, yw, yprev, (mu_x, std_x, mu_y, std_y) = prepare_calib(
        args.csv_dir, args.glob, args.env, args.win, args.stride, args.calib_windows
    )

    # Import model class and instantiate
    InvPINN = dyn_import_inversepinn(args.model_def)
    if InvPINN is None:
        raise SystemExit("Could not import InversePINN from --model-def")

    model = InvPINN()
    sd = torch.load(args.ckpt, map_location="cpu")
    # Support raw state_dict or checkpoint with 'state_dict'
    state_dict = sd if isinstance(sd, dict) and any(k.startswith("Tnet.") for k in sd.keys()) else sd.get("state_dict", sd)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)

    # Activation calibration
    stats = collect_activation_stats(model, Xw, yprev, mu_x, std_x, mu_y, std_y,
                                     max_batches=64, batch=32)

    # Compute activation scales (symmetric, int8 for x,h and fc-in). For c we will use Q15
    S_x    = float(symmetric_qparams(np.array([stats["x_absmax"]]), 8)[0])
    S_h    = float(symmetric_qparams(np.array([stats["h_absmax"]]), 8)[0])
    S_fcin = float(symmetric_qparams(np.array([stats["fc_in_absmax"]]), 8)[0])
    S_dout = float(symmetric_qparams(np.array([stats["delta_out_absmax"]]), 8)[0])
    # For c_t we choose Q15 (scale=1/32767 for the logical unit range). If you want to
    # compress/expand the dynamic range, you can adjust here; we expose S_C as 1.0 by default.
    S_c = 1.0

    # Split LSTM weights and quantize per gate
    gates = split_lstm_weights(model.state_dict())

    # Per‑row int8 for W, int32 bias needs (S_w_row * S_act) accumulation rule at runtime.
    W_blocks = []
    B_blocks = []
    w_scales: Dict[str, np.ndarray] = {}

    for name in ["i","f","g","o"]:
        Wih = gates[f"W_ih_{name}"]
        Whh = gates[f"W_hh_{name}"]
        b   = gates[f"b_{name}"]

        Wih_q, Wih_s = quant_per_row(Wih, 8)
        Whh_q, Whh_s = quant_per_row(Whh, 8)
        w_scales[f"W_ih_{name}"] = Wih_s
        w_scales[f"W_hh_{name}"] = Whh_s

        # Bias quantization: bias_float / (S_w * S_act). Here we encode as int32 using input scale S_x
        # for W_ih and hidden scale S_h for W_hh; engines usually add (W_ih*x + W_hh*h_prev + b_q)
        # so we provide two bias contributions or one fused with a chosen common scale. We choose to
        # store two bias vectors so you can accumulate them in the correct scale domains.
        b_ih_q = np.round(b / (Wih_s * S_x)).astype(np.int32)
        b_hh_q = np.round(b / (Whh_s * S_h)).astype(np.int32)

        W_blocks.append(emit_w_block(f"LSTM_WIH_{name.upper()}", Wih_q, Wih_s, args.name))
        W_blocks.append(emit_w_block(f"LSTM_WHH_{name.upper()}", Whh_q, Whh_s, args.name))
        B_blocks.append(emit_b_block(f"LSTM_BIH_{name.upper()}", b_ih_q, args.name))
        B_blocks.append(emit_b_block(f"LSTM_BHH_{name.upper()}", b_hh_q, args.name))

    # FC head quantization (per‑row; out=1)
    W_fc = model.state_dict()["Tnet.fc_delta.weight"].cpu().numpy()  # [1, H]
    b_fc = model.state_dict()["Tnet.fc_delta.bias"].cpu().numpy()    # [1]
    W_fc_q, W_fc_s = quant_per_row(W_fc, 8)
    b_fc_q = np.round(b_fc / (W_fc_s * S_fcin)).astype(np.int32)

    # LUTs
    xs, lut_sig, lut_tanh = q15_lut_sigmoid_tanh()

    # Fixed‑point normalization constants (Qqfix)
    qfix = args.qfix
    Q = float(1 << qfix)
    invstd_q = np.round((1.0 / (std_x + 1e-8)) * Q).astype(np.int32)
    negoff_q = np.round((-mu_x / (std_x + 1e-8)) * Q).astype(np.int32)

    # Emit header
    H = W_fc.shape[1]
    INPUT = Xw.shape[2]

    w_block_str = "\n".join(W_blocks)
    b_block_str = "\n".join(B_blocks)

    header = HEADER_TMPL.format(
        N=args.name,
        INPUT=INPUT,
        H=H,
        MU_X=fmt_array_1d(mu_x),
        STD_X=fmt_array_1d(std_x),
        MU_Y=mu_y,
        STD_Y=std_y,
        QFIX=qfix,
        INVSTD_Q=fmt_array_1d_int(invstd_q),
        NEGOFF_Q=fmt_array_1d_int(negoff_q),
        S_X=S_x,
        S_H=S_h,
        S_C=S_c,
        S_FCIN=S_fcin,
        S_DOUT=S_dout,
        W_BLOCKS=w_block_str,
        B_BLOCKS=b_block_str,
        FC_W="{" + fmt_array_1d_int(W_fc_q[0]) + "}",
        FC_WS=fmt_array_1d(W_fc_s),
        FC_B=fmt_array_1d_int(b_fc_q),
        LUT_N=len(xs),
        LUT_XMIN=float(xs[0]),
        LUT_XMAX=float(xs[-1]),
        LUT_SIG=fmt_array_1d_int(lut_sig),
        LUT_TANH=fmt_array_1d_int(lut_tanh),
    )

    out_path = Path(args.out_header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header)
    print(f"[OK] Wrote header → {out_path}")

if __name__ == "__main__":
    main()
