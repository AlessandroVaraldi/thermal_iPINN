#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ipinn_synthetic_odeforward_refactored.py

Refactor of train_ipinn_synthetic_odeforward.py with:
  - less redundancy
  - unified evaluation / plotting
  - unified ODE forward integrator

Functionality preserved:
  - WHITE-BOX: ODE only (fit R_cond, C_th)
  - GRAY-BOX : ODE forward + NN correction u(t) (fit R_cond, C_th + NN)
  - BLACK-PINN: NN mapping inputs -> T_case + physics residual in loss (phys params fixed from WHITE-BOX)
  - fixed R_conv per scenario (from JSON), one held-out test scenario evaluated on full sequence
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Any


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def robust_dt(seconds: np.ndarray) -> float:
    ds = np.diff(seconds.astype(float))
    ds = ds[np.isfinite(ds) & (ds > 0)]
    if ds.size == 0:
        return 0.1
    p10, p90 = np.percentile(ds, [10, 90])
    core = ds[(ds >= p10) & (ds <= p90)]
    return float(np.median(core if core.size else ds))


def set_determinism(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err2 = (pred - target) ** 2 * mask
    return err2.sum() / (mask.sum() + 1e-9)


def ode_residual_fd(
    *,
    T_pred: torch.Tensor,     # (B,T)
    P_W: torch.Tensor,        # (B,T)
    T_bplate: torch.Tensor,   # (B,T)
    Tamb: torch.Tensor,       # (B,T)
    dt: torch.Tensor,         # (B,) or scalar
    Rcond: torch.Tensor,      # scalar tensor (fixed)
    Cth: torch.Tensor,        # scalar tensor (fixed)
    Rconv_eff: torch.Tensor,  # (B,1)
) -> torch.Tensor:
    """
    Physics residual for the 1R-1C thermal ODE, using forward finite differences:

      dT/dt - (1/Cth) * ( P - (T-Tb)/Rcond - (T-Tamb)/Rconv ) = 0

    Returns:
      residual: (B, T-1) evaluated on intervals k -> k+1 using values at k.
    """
    B, T = T_pred.shape
    if T < 2:
        return T_pred.new_zeros((B, 0))

    dt_b = dt.view(B, 1) if dt.ndim > 0 else dt
    dTdt = (T_pred[:, 1:] - T_pred[:, :-1]) / (dt_b + 1e-12)

    Tk = T_pred[:, :-1]
    Pk = P_W[:, :-1]
    Tb = T_bplate[:, :-1]
    Ta = Tamb[:, :-1]

    rhs = (1.0 / Cth) * (Pk - (Tk - Tb) / Rcond - (Tk - Ta) / Rconv_eff)
    return dTdt - rhs


def error_stats_from_loader(
    predict_fn,
    loader: DataLoader,
    device: torch.device,
    target_key: str = "T_case_meas",
) -> Dict[str, float]:
    assert target_key in ("T_case_meas", "T_case_true")
    errs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            pred = predict_fn(batch)  # (B,T)
            target = batch[target_key]
            mask = batch["mask"]
            e = (pred - target) * mask
            errs.append(e.detach().cpu().numpy().ravel())

    if not errs:
        return {}

    e_all = np.concatenate(errs)
    e_all = e_all[np.isfinite(e_all)]
    abs_e = np.abs(e_all)
    return {
        "rmse": float(np.sqrt(np.mean(e_all ** 2))),
        "mean_abs": float(np.mean(abs_e)),
        "max_abs": float(np.max(abs_e)),
        "q95_abs": float(np.quantile(abs_e, 0.95)),
        "q99_abs": float(np.quantile(abs_e, 0.99)),
    }


def rmse_from_loader(predict_fn, loader: DataLoader, device: torch.device, target_key: str) -> float:
    stats = error_stats_from_loader(predict_fn, loader, device, target_key=target_key)
    return float(stats.get("rmse", np.nan))


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class SyntheticThermalWindowDataset(Dataset):
    """
    Windowed dataset for synthetic CSV files.

    Each item: raw arrays.
    collate_fn: pads to fixed window, normalizes inputs.
    """

    def __init__(
        self,
        csv_paths: List[str],
        window_size: int = 512,
        stride: Optional[int] = None,
        preload: bool = True,
    ):
        super().__init__()
        self.paths = sorted(csv_paths)
        self.window = int(window_size)
        self.stride = int(stride) if stride is not None else max(1, self.window // 2)

        self.scenario_id_by_path = {p: i for i, p in enumerate(self.paths)}
        self._cache: Dict[str, pd.DataFrame] = {}
        self._windows: List[Dict] = []
        self._build_windows(preload=preload)

        self.input_mean = np.zeros(3, dtype=float)
        self.input_std = np.ones(3, dtype=float)

    def _read_csv(self, path: str) -> pd.DataFrame:
        if path in self._cache:
            return self._cache[path]
        df = pd.read_csv(path)
        self._cache[path] = df
        return df

    def _build_windows(self, preload: bool = True):
        for p in self.paths:
            df = self._read_csv(p)
            if "time_s" not in df.columns:
                raise RuntimeError(f"{p}: 'time_s' column not found.")
            t = df["time_s"].to_numpy(dtype=float)
            dt_est = robust_dt(t)
            n = len(df)
            if n <= 1:
                continue

            scen_id = self.scenario_id_by_path[p]

            if self.window <= 0 or self.window >= n:
                self._windows.append({"path": p, "start": 0, "dt": dt_est, "scenario_id": scen_id})
            else:
                for start in range(0, max(1, n - self.window + 1), self.stride):
                    self._windows.append({"path": p, "start": start, "dt": dt_est, "scenario_id": scen_id})

        if not self._windows:
            raise RuntimeError("No windows generated. Check window_size and CSV files.")

    def compute_normalization(self, n_samples_max: int = 5000, rng_seed: int = 7):
        rng = np.random.default_rng(rng_seed)
        idx = np.arange(len(self._windows))
        rng.shuffle(idx)
        idx = idx[: min(len(idx), n_samples_max)]

        P_all, Tb_all, Tamb_all = [], [], []
        for i in idx:
            arr = self._extract_window(self._windows[i])
            P_all.append(arr["P_W"])
            Tb_all.append(arr["T_bplate"])
            Tamb_all.append(arr["Tamb"])

        def concat_valid(lst):
            if not lst:
                return np.array([])
            cat = np.concatenate(lst)
            return cat[np.isfinite(cat)]

        P_cat = concat_valid(P_all)
        Tb_cat = concat_valid(Tb_all)
        Tamb_cat = concat_valid(Tamb_all)

        self.input_mean[0] = float(np.mean(P_cat)) if P_cat.size else 0.0
        self.input_std[0] = float(np.std(P_cat)) if P_cat.size else 1.0

        self.input_mean[1] = float(np.mean(Tb_cat)) if Tb_cat.size else 25.0
        self.input_std[1] = float(np.std(Tb_cat)) if Tb_cat.size else 1.0

        self.input_mean[2] = float(np.mean(Tamb_cat)) if Tamb_cat.size else 25.0
        self.input_std[2] = float(np.std(Tamb_cat)) if Tamb_cat.size else 1.0

        self.input_std = np.where(self.input_std < 1e-6, 1.0, self.input_std)

        print("[Dataset] Normalization:")
        print("  input_mean:", self.input_mean)
        print("  input_std :", self.input_std)

    def _extract_window(self, item: Dict) -> Dict:
        path = item["path"]
        start = int(item["start"])
        df = self._read_csv(path)

        required = ["time_s", "P_W", "Tamb_C", "T_bplate_C", "T_case_true_C", "T_case_meas_C"]
        for col in required:
            if col not in df.columns:
                raise RuntimeError(f"{path}: column '{col}' not found.")

        t = df["time_s"].to_numpy(dtype=float)
        P = df["P_W"].to_numpy(dtype=float)
        Tamb = df["Tamb_C"].to_numpy(dtype=float)
        Tb = df["T_bplate_C"].to_numpy(dtype=float)
        Ttrue = df["T_case_true_C"].to_numpy(dtype=float)
        Tmeas = df["T_case_meas_C"].to_numpy(dtype=float)

        n = len(t)
        end = min(start + self.window, n)
        sl = slice(start, end)

        t_win = t[sl]
        dt_local = robust_dt(t_win) if len(t_win) > 1 else float(item["dt"])

        return {
            "time_s": t_win,
            "P_W": P[sl],
            "Tamb": Tamb[sl],
            "T_bplate": Tb[sl],
            "T_case_true": Ttrue[sl],
            "T_case_meas": Tmeas[sl],
            "dt": dt_local,
            "scenario_id": item["scenario_id"],
        }

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict:
        return self._extract_window(self._windows[idx])

    def collate_fn(self, batch: List[Dict]) -> Dict:
        B = len(batch)
        T = self.window

        inputs_raw = np.zeros((B, T, 3), dtype=float)
        T_case_meas = np.zeros((B, T), dtype=float)
        T_case_true = np.zeros((B, T), dtype=float)
        mask = np.zeros((B, T), dtype=float)

        P_W = np.zeros((B, T), dtype=float)
        T_bplate = np.zeros((B, T), dtype=float)
        Tamb = np.zeros((B, T), dtype=float)

        dt_arr = np.zeros((B,), dtype=float)
        scen_ids = np.zeros((B,), dtype=np.int64)

        for i, item in enumerate(batch):
            L = len(item["P_W"])
            inputs_raw[i, :L, 0] = item["P_W"]
            inputs_raw[i, :L, 1] = item["T_bplate"]
            inputs_raw[i, :L, 2] = item["Tamb"]

            T_case_meas[i, :L] = item["T_case_meas"]
            T_case_true[i, :L] = item["T_case_true"]
            mask[i, :L] = 1.0

            P_W[i, :L] = item["P_W"]
            T_bplate[i, :L] = item["T_bplate"]
            Tamb[i, :L] = item["Tamb"]

            dt_arr[i] = item["dt"]
            scen_ids[i] = item["scenario_id"]

        # normalize inputs
        x = inputs_raw.copy()
        for ch in range(3):
            bad = ~np.isfinite(x[:, :, ch])
            x[:, :, ch][bad] = self.input_mean[ch]

        mean = self.input_mean.reshape(1, 1, 3)
        std = self.input_std.reshape(1, 1, 3)
        inputs_norm = (x - mean) / std

        return {
            "inputs": torch.tensor(inputs_norm, dtype=torch.float32),
            "T_case_meas": torch.tensor(T_case_meas, dtype=torch.float32),
            "T_case_true": torch.tensor(T_case_true, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "P_W": torch.tensor(P_W, dtype=torch.float32),
            "T_bplate": torch.tensor(T_bplate, dtype=torch.float32),
            "Tamb": torch.tensor(Tamb, dtype=torch.float32),
            "dt": torch.tensor(dt_arr, dtype=torch.float32),
            "scenario_id": torch.tensor(scen_ids, dtype=torch.long),
        }


def make_dataloaders(
    data_dir: Path,
    csv_files: List[str],
    window: int,
    batch_size: int,
    val_fraction: float,
    seed: int,
) -> Tuple[SyntheticThermalWindowDataset, DataLoader, DataLoader]:
    csv_paths = [str(data_dir / name) for name in csv_files]
    if not csv_paths:
        raise SystemExit("No training CSV files specified.")

    ds = SyntheticThermalWindowDataset(
        csv_paths,
        window_size=window,
        stride=max(1, window // 2),
        preload=True,
    )
    ds.compute_normalization(rng_seed=seed)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate_fn)
    return ds, train_loader, val_loader


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class MLPRegressor(nn.Module):
    """Per-time-step MLP: (B,T,3)->(B,T). Used as u(t) in GRAY, as T(t) in BLACK."""
    def __init__(self, in_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        y = self.net(x.reshape(B * T, D))
        return y.view(B, T)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.resample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        T = x.shape[-1]
        out = out[..., -T:]  # causal crop
        out = self.dropout(self.act(out))
        if self.resample is not None:
            x = self.resample(x)
        return out + x


class TCNRegressor(nn.Module):
    """TCN: (B,T,3)->(B,T)."""
    def __init__(self, in_dim: int = 3, hidden: int = 32, n_blocks: int = 3, kernel_size: int = 3):
        super().__init__()
        layers = []
        in_ch = in_dim
        for i in range(n_blocks):
            layers.append(TCNBlock(in_ch, hidden, kernel_size=kernel_size, dilation=2**i, dropout=0.1))
            in_ch = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)        # (B,3,T)
        feat = self.tcn(x)
        out = self.head(feat)         # (B,1,T)
        return out.squeeze(1)         # (B,T)


class LSTMRegressor(nn.Module):
    """LSTM: (B,T,3)->(B,T)."""
    def __init__(self, in_dim: int = 3, hidden: int = 64, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        """
        If return_state=True, returns (y, (h, c)) so we can do TBPTT with state carry.
        Otherwise returns y only.
        """
        out, st = self.lstm(x, state) if state is not None else self.lstm(x)
        y = self.head(out).squeeze(-1)
        if return_state:
            return y, st
        return y


def make_model(arch: str, hidden: int) -> nn.Module:
    arch = arch.lower()
    if arch == "mlp":
        return MLPRegressor(in_dim=3, hidden=hidden)
    if arch == "tcn":
        return TCNRegressor(in_dim=3, hidden=hidden, n_blocks=3, kernel_size=3)
    if arch == "lstm":
        return LSTMRegressor(in_dim=3, hidden=hidden, n_layers=1, dropout=0.1)
    raise ValueError(f"Unknown architecture: {arch}")


def model_forward_u(
    model: nn.Module,
    x: torch.Tensor,
    state: Optional[Any] = None,
    return_state: bool = False,
):
    """
    Helper to support optional recurrent state for LSTM while keeping other archs unchanged.
    Returns:
      - if return_state: (u, new_state)
      - else: u
    """
    if isinstance(model, LSTMRegressor):
        return model(x, state=state, return_state=return_state)
    # MLP/TCN: stateless
    if return_state:
        return model(x), None
    return model(x)


# ---------------------------------------------------------------------
# Unified ODE Forward
# ---------------------------------------------------------------------
def ode_forward(
    model: nn.Module,
    *,
    inputs: torch.Tensor,     # (B,T,3) normalized
    P_W: torch.Tensor,        # (B,T)
    T_bplate: torch.Tensor,   # (B,T)
    Tamb: torch.Tensor,       # (B,T)
    dt: torch.Tensor,         # (B,) or scalar
    T0: torch.Tensor,         # (B,) initial
    Rcond: torch.Tensor,      # scalar tensor
    Cth: torch.Tensor,        # scalar tensor
    Rconv_eff: torch.Tensor,  # (B,1)
    use_nn: bool,
    integrator: str = "euler",
    u_override: Optional[torch.Tensor] = None,  # (B,T) optional precomputed u
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrates:
      T_{k+1} = T_k + (dt/Cth)*( Pk + uk - (Tk-Tb)/Rcond - (Tk-Tamb)/Rconv )
    Returns:
      T_pred: (B,T)
      u:      (B,T)
    integrator:
      - "euler": explicit Euler
      - "rk4"  : explicit Runge-Kutta 4 with simple input interpolation

    """
    B, T = P_W.shape

    if u_override is not None:
        u = u_override
    else:
        if use_nn:
            u = model(inputs)  # (B,T)
        else:
            u = torch.zeros_like(P_W)

    T_pred = torch.zeros_like(P_W)
    T_pred[:, 0] = T0

    dt_b = dt.view(B, 1) if dt.ndim > 0 else dt

    integrator = integrator.lower().strip()
    if integrator not in ("euler", "rk4"):
        raise ValueError(f"Unknown integrator='{integrator}'. Use 'euler' or 'rk4'.")

    def f(Tk_1col, P_1col, Tb_1col, Ta_1col, u_1col):
        # shapes: (B,1)
        return (1.0 / Cth) * (P_1col + u_1col - (Tk_1col - Tb_1col) / Rcond - (Tk_1col - Ta_1col) / Rconv_eff)

    for k in range(T - 1):
        Tk = T_pred[:, k:k+1]
        Pk = P_W[:, k:k+1]
        Tb = T_bplate[:, k:k+1]
        Ta = Tamb[:, k:k+1]
        uk = u[:, k:k+1]

        if integrator == "euler":
            dT = dt_b * f(Tk, Pk, Tb, Ta, uk)
            T_pred[:, k+1:k+2] = Tk + dT
        else:
            # RK4 with simple interpolation of inputs between k and k+1
            Pk1 = P_W[:, k+1:k+2]
            Tb1 = T_bplate[:, k+1:k+2]
            Ta1 = Tamb[:, k+1:k+2]
            uk1 = u[:, k+1:k+2]

            Pm = 0.5 * (Pk + Pk1)
            Tbm = 0.5 * (Tb + Tb1)
            Tam = 0.5 * (Ta + Ta1)
            um = 0.5 * (uk + uk1)

            k1 = f(Tk, Pk, Tb, Ta, uk)
            k2 = f(Tk + 0.5 * dt_b * k1, Pm, Tbm, Tam, um)
            k3 = f(Tk + 0.5 * dt_b * k2, Pm, Tbm, Tam, um)
            k4 = f(Tk + dt_b * k3, Pk1, Tb1, Ta1, uk1)

            T_pred[:, k+1:k+2] = Tk + (dt_b / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return T_pred, u


# ---------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------
@dataclass
class ODEConfig:
    lambda_u: float = 1e-3
    clip_grad: float = 5.0
    use_nn: bool = True
    Rcond_prior: Optional[float] = None
    Cth_prior: Optional[float] = None
    phys_prior_weight: float = 0.0
    integrator: str = "euler"


class ODETrainer:
    """
    WHITE/GRAY trainer: learns log_Rcond, log_Cth and optionally NN params.
    Rconv is fixed per scenario (vector).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        Cth_init: float,
        Rcond_init: float,
        Rconv_list: List[float],
        device: str,
        cfg: ODEConfig,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.cfg = cfg

        self.log_Rcond = nn.Parameter(torch.log(torch.tensor(float(Rcond_init), dtype=torch.float32, device=self.device)))
        self.log_Cth = nn.Parameter(torch.log(torch.tensor(float(Cth_init), dtype=torch.float32, device=self.device)))

        self.Rconv_fixed = torch.tensor(Rconv_list, dtype=torch.float32, device=self.device)  # (N_scen,)

        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "R_cond": [], "C_th": [], "u_reg": []}

    @property
    def Rcond(self) -> torch.Tensor:
        return torch.exp(self.log_Rcond)

    @property
    def Cth(self) -> torch.Tensor:
        return torch.exp(self.log_Cth)

    def predict(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        inputs = batch["inputs"]
        P_W = batch["P_W"]
        T_bplate = batch["T_bplate"]
        Tamb = batch["Tamb"]
        dt = batch["dt"]
        scen_id = batch["scenario_id"]
        T_meas = batch["T_case_meas"]

        Rconv_eff = self.Rconv_fixed[scen_id].view(-1, 1)  # (B,1)
        T0 = T_meas[:, 0]

        T_pred, u = ode_forward(
            self.model,
            inputs=inputs,
            P_W=P_W,
            T_bplate=T_bplate,
            Tamb=Tamb,
            dt=dt,
            T0=T0,
            Rcond=self.Rcond,
            Cth=self.Cth,
            Rconv_eff=Rconv_eff,
            use_nn=self.cfg.use_nn,
            integrator=self.cfg.integrator,
        )
        return T_pred, u

    def loss(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T_pred, u = self.predict(batch)
        T_meas = batch["T_case_meas"].to(self.device)
        mask = batch["mask"].to(self.device)

        data_loss = masked_mse(T_pred, T_meas, mask)

        if self.cfg.use_nn:
            reg_u = (u ** 2).mean() * float(self.cfg.lambda_u)
        else:
            reg_u = torch.tensor(0.0, device=self.device)

        phys_prior = torch.tensor(0.0, device=self.device)
        if self.cfg.phys_prior_weight > 0.0:
            if self.cfg.Rcond_prior is not None:
                phys_prior = phys_prior + ((self.Rcond - self.cfg.Rcond_prior) / (self.cfg.Rcond_prior + 1e-9)) ** 2
            if self.cfg.Cth_prior is not None:
                phys_prior = phys_prior + ((self.Cth - self.cfg.Cth_prior) / (self.cfg.Cth_prior + 1e-9)) ** 2
            phys_prior = phys_prior * float(self.cfg.phys_prior_weight)

        total = data_loss + reg_u + phys_prior
        return total, data_loss, reg_u

    def train_epochs(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        lr: float,
        lr_phys: float,
        patience: int = 0,
        min_delta: float = 0.0,
    ):
        nn_params = list(self.model.parameters()) if self.cfg.use_nn else []
        phys_params = [self.log_Rcond, self.log_Cth]

        param_groups = [{"params": phys_params, "lr": lr_phys}]
        if nn_params:
            param_groups.insert(0, {"params": nn_params, "lr": lr})

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        best_val = float("inf")
        bad_epochs = 0
        best_state = None

        def _snapshot_best():
            return {
                "model": deepcopy(self.model.state_dict()) if self.cfg.use_nn else None,
                "log_Rcond": self.log_Rcond.detach().clone(),
                "log_Cth": self.log_Cth.detach().clone(),
            }

        for ep in range(1, epochs + 1):
            self.model.train()
            train_losses, train_u = [], []

            for batch in train_loader:
                optimizer.zero_grad()
                total, _, reg_u = self.loss(batch)
                total.backward()

                # clip both NN and phys params
                all_params = nn_params + phys_params
                if all_params:
                    nn.utils.clip_grad_norm_(all_params, max_norm=float(self.cfg.clip_grad))
                optimizer.step()

                train_losses.append(float(total.item()))
                train_u.append(float(reg_u.item()))

            train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
            train_u_mean = float(np.mean(train_u)) if train_u else 0.0

            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        total, _, _ = self.loss(batch)
                        val_losses.append(float(total.item()))
                val_mean = float(np.mean(val_losses)) if val_losses else train_mean
                scheduler.step(val_mean)
            else:
                val_mean = train_mean

            # --- Early stopping on val loss ---
            if val_loader is not None and patience and patience > 0:
                if val_mean < (best_val - float(min_delta)):
                    best_val = float(val_mean)
                    bad_epochs = 0
                    best_state = _snapshot_best()
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"[EarlyStop] No val improvement for {patience} epochs. Restoring best model (val={best_val:.6f}).")
                        if best_state is not None:
                            if best_state["model"] is not None:
                                self.model.load_state_dict(best_state["model"])
                            self.log_Rcond.data.copy_(best_state["log_Rcond"])
                            self.log_Cth.data.copy_(best_state["log_Cth"])
                        break

            self.history["epoch"].append(ep)
            self.history["train_loss"].append(train_mean)
            self.history["val_loss"].append(val_mean)
            self.history["R_cond"].append(float(self.Rcond.detach().cpu().numpy()))
            self.history["C_th"].append(float(self.Cth.detach().cpu().numpy()))
            self.history["u_reg"].append(train_u_mean)

            if ep == 1 or ep % 10 == 0 or ep == epochs:
                print(
                    f"[Epoch {ep:03d}] train={train_mean:.6f} val={val_mean:.6f} | "
                    f"R_cond={self.history['R_cond'][-1]:.4f} K/W, "
                    f"C_th={self.history['C_th'][-1]:.4f} J/K, "
                    f"u_reg={train_u_mean:.6e}"
                )

    def train_epochs_tbptt(
        self,
        train_loader_long: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        lr: float,
        lr_phys: float,
        tbptt_len: int,
        patience: int = 0,
        min_delta: float = 0.0,
    ):
        """
        Train on long windows but truncate gradients every tbptt_len samples.

        - ODE temperature state is carried across segments but detached (TBPTT).
        - If the NN is an LSTMRegressor, its hidden state is also carried and detached.
        - For other NN archs (MLP/TCN), state is None (stateless).

        NOTE: We overlap segments by 1 sample to keep time continuity without skipping steps:
          segment starts: 0, (tbptt_len-1), 2*(tbptt_len-1), ...
          and we do NOT double-count the first sample of each segment (mask it out).
        """
        tbptt_len = int(tbptt_len)
        if tbptt_len < 2:
            raise ValueError("tbptt_len must be >= 2 to allow overlapping segments.")

        nn_params = list(self.model.parameters()) if self.cfg.use_nn else []
        phys_params = [self.log_Rcond, self.log_Cth]

        param_groups = [{"params": phys_params, "lr": lr_phys}]
        if nn_params:
            param_groups.insert(0, {"params": nn_params, "lr": lr})

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        best_val = float("inf")
        bad_epochs = 0
        best_state = None

        def _snapshot_best():
            return {
                "model": deepcopy(self.model.state_dict()) if self.cfg.use_nn else None,
                "log_Rcond": self.log_Rcond.detach().clone(),
                "log_Cth": self.log_Cth.detach().clone(),
            }

        step = tbptt_len - 1

        for ep in range(1, epochs + 1):
            self.model.train()
            train_losses, train_u = [], []

            for batch in train_loader_long:
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

                inputs = batch["inputs"]        # (B,T,3)
                P_W = batch["P_W"]              # (B,T)
                T_bplate = batch["T_bplate"]    # (B,T)
                Tamb = batch["Tamb"]            # (B,T)
                dt = batch["dt"]                # (B,)
                scen_id = batch["scenario_id"]  # (B,)
                T_meas = batch["T_case_meas"]   # (B,T)
                mask = batch["mask"]            # (B,T)

                B, T = P_W.shape
                Rconv_eff = self.Rconv_fixed[scen_id].view(-1, 1)

                optimizer.zero_grad()

                # Prior term once per batch (avoid scaling it by num segments)
                phys_prior = torch.tensor(0.0, device=self.device)
                if self.cfg.phys_prior_weight > 0.0:
                    if self.cfg.Rcond_prior is not None:
                        phys_prior = phys_prior + ((self.Rcond - self.cfg.Rcond_prior) / (self.cfg.Rcond_prior + 1e-9)) ** 2
                    if self.cfg.Cth_prior is not None:
                        phys_prior = phys_prior + ((self.Cth - self.cfg.Cth_prior) / (self.cfg.Cth_prior + 1e-9)) ** 2
                    phys_prior = phys_prior * float(self.cfg.phys_prior_weight)

                # TBPTT state carry
                T0 = T_meas[:, 0]  # start from measurement at time 0
                rnn_state = None

                batch_loss_total = 0.0
                batch_u_total = 0.0
                n_seg = 0

                for s in range(0, T, step):
                    e = min(s + tbptt_len, T)
                    if e - s < 2:
                        break

                    x_seg = inputs[:, s:e, :]
                    P_seg = P_W[:, s:e]
                    Tb_seg = T_bplate[:, s:e]
                    Ta_seg = Tamb[:, s:e]
                    y_seg = T_meas[:, s:e]
                    m_seg = mask[:, s:e].clone()

                    # Don't double-count overlap sample (time index s) for s>0
                    if s > 0:
                        m_seg[:, 0] = 0.0

                    # Compute u on this segment (optionally with LSTM state carry)
                    if self.cfg.use_nn:
                        u_seg, rnn_state = model_forward_u(self.model, x_seg, state=rnn_state, return_state=True)
                        if rnn_state is not None:
                            rnn_state = tuple(v.detach() for v in rnn_state)
                        reg_u = (u_seg ** 2).mean() * float(self.cfg.lambda_u)
                    else:
                        u_seg = torch.zeros_like(P_seg)
                        reg_u = torch.tensor(0.0, device=self.device)

                    # Integrate using precomputed u (avoid re-calling the NN inside ode_forward)
                    T_pred_seg, _ = ode_forward(
                        self.model,
                        inputs=x_seg,
                        P_W=P_seg,
                        T_bplate=Tb_seg,
                        Tamb=Ta_seg,
                        dt=dt,
                        T0=T0,
                        Rcond=self.Rcond,
                        Cth=self.Cth,
                        Rconv_eff=Rconv_eff,
                        use_nn=False,
                        u_override=u_seg,
                        integrator=self.cfg.integrator,
                    )

                    data_loss = masked_mse(T_pred_seg, y_seg, m_seg)
                    loss_seg = data_loss + reg_u
                    loss_seg.backward()

                    batch_loss_total += float(loss_seg.detach().cpu().item())
                    batch_u_total += float(reg_u.detach().cpu().item())
                    n_seg += 1

                    # carry ODE state, truncate gradient
                    T0 = T_pred_seg[:, -1].detach()

                # Backprop phys prior once per batch
                if phys_prior.requires_grad and float(phys_prior.detach().cpu().item()) != 0.0:
                    phys_prior.backward()

                # clip both NN and phys params
                all_params = nn_params + phys_params
                if all_params:
                    nn.utils.clip_grad_norm_(all_params, max_norm=float(self.cfg.clip_grad))

                optimizer.step()

                if n_seg > 0:
                    train_losses.append(batch_loss_total / n_seg)
                    train_u.append(batch_u_total / n_seg)

            train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
            train_u_mean = float(np.mean(train_u)) if train_u else 0.0

            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in val_loader:
                        total, _, _ = self.loss(vb)
                        val_losses.append(float(total.item()))
                val_mean = float(np.mean(val_losses)) if val_losses else train_mean
                scheduler.step(val_mean)
            else:
                val_mean = train_mean

            # Early stopping on val loss
            if val_loader is not None and patience and patience > 0:
                if val_mean < (best_val - float(min_delta)):
                    best_val = float(val_mean)
                    bad_epochs = 0
                    best_state = _snapshot_best()
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"[EarlyStop] No val improvement for {patience} epochs. Restoring best model (val={best_val:.6f}).")
                        if best_state is not None:
                            if best_state["model"] is not None:
                                self.model.load_state_dict(best_state["model"])
                            self.log_Rcond.data.copy_(best_state["log_Rcond"])
                            self.log_Cth.data.copy_(best_state["log_Cth"])
                        break

            self.history["epoch"].append(ep)
            self.history["train_loss"].append(train_mean)
            self.history["val_loss"].append(val_mean)
            self.history["R_cond"].append(float(self.Rcond.detach().cpu().numpy()))
            self.history["C_th"].append(float(self.Cth.detach().cpu().numpy()))
            self.history["u_reg"].append(train_u_mean)

            if ep == 1 or ep % 10 == 0 or ep == epochs:
                print(
                    f"[Epoch {ep:03d}] (TBPTT) train={train_mean:.6f} val={val_mean:.6f} | "
                    f"R_cond={self.history['R_cond'][-1]:.4f} K/W, "
                    f"C_th={self.history['C_th'][-1]:.4f} J/K, "
                    f"u_reg={train_u_mean:.6e}"
                )

    def export_params(self) -> Dict[str, float]:
        return {"R_cond": float(self.Rcond.detach().cpu().numpy()), "C_th": float(self.Cth.detach().cpu().numpy())}


class BlackPINNTrainer:
    """BLACK-PINN trainer: predicts T_case, trained with data loss + physics residual loss (fixed phys params from WHITE-BOX)."""
    def __init__(self, model: nn.Module, *, device: str, clip_grad: float = 5.0, lambda_phys: float = 1.0, Rcond_fixed: float, Cth_fixed: float, Rconv_list: List[float]):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.clip_grad = float(clip_grad)
        self.lambda_phys = float(lambda_phys)

        # Fixed physical parameters (from WHITE-BOX identification)
        self.Rcond = torch.tensor(float(Rcond_fixed), dtype=torch.float32, device=self.device)
        self.Cth = torch.tensor(float(Cth_fixed), dtype=torch.float32, device=self.device)
        self.Rconv_fixed = torch.tensor(Rconv_list, dtype=torch.float32, device=self.device)  # (N_scen,)

        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "data_loss": [], "phys_loss": []}

    def predict(self, batch: Dict) -> torch.Tensor:
        x = batch["inputs"].to(self.device)
        return self.model(x)  # (B,T) 
    
    def loss(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        pred = self.predict(batch)  # (B,T)
        y = batch["T_case_meas"]
        m = batch["mask"]

        data_loss = masked_mse(pred, y, m)

        # physics residual loss (only on valid consecutive samples)
        scen_id = batch["scenario_id"]
        Rconv_eff = self.Rconv_fixed[scen_id].view(-1, 1)
        res = ode_residual_fd(
            T_pred=pred,
            P_W=batch["P_W"],
            T_bplate=batch["T_bplate"],
            Tamb=batch["Tamb"],
            dt=batch["dt"],
            Rcond=self.Rcond,
            Cth=self.Cth,
            Rconv_eff=Rconv_eff,
        )  # (B,T-1)

        if res.numel() == 0:
            phys_loss = torch.tensor(0.0, device=self.device)
        else:
            m_pair = m[:, 1:] * m[:, :-1]  # valid intervals only
            phys_loss = masked_mse(res, torch.zeros_like(res), m_pair)

        total = data_loss + self.lambda_phys * phys_loss
        return total, data_loss, phys_loss
     
    def train_epochs_tbptt(
        self,
        train_loader_long: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        lr: float,
        tbptt_len: int,
        patience: int = 0,
        min_delta: float = 0.0,
    ):
        """
        TBPTT for BLACK-PINN training on long windows.
        For LSTMRegressor: carries hidden state across segments but detaches it.
        For stateless models (MLP/TCN): state is None (segments independent).
        Uses 1-sample overlap and masks out the overlap sample to avoid double-counting.
        """
        tbptt_len = int(tbptt_len)
        if tbptt_len < 2:
            raise ValueError("tbptt_len must be >= 2.")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        best_val = float("inf")
        bad_epochs = 0
        best_state = None

        step = tbptt_len - 1

        for ep in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            train_data_losses = []
            train_phys_losses = []

            for batch in train_loader_long:
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                x = batch["inputs"]         # (B,T,3)
                y = batch["T_case_meas"]    # (B,T)
                m = batch["mask"]           # (B,T)
                P = batch["P_W"]
                Tb = batch["T_bplate"]
                Ta = batch["Tamb"]
                dt = batch["dt"]
                scen_id = batch["scenario_id"]
                Rconv_eff = self.Rconv_fixed[scen_id].view(-1, 1)

                B, T = y.shape
                optimizer.zero_grad()

                rnn_state = None
                batch_loss_total = 0.0
                batch_data_total = 0.0
                batch_phys_total = 0.0
                n_seg = 0

                for s in range(0, T, step):
                    e = min(s + tbptt_len, T)
                    if e - s < 1:
                        break

                    x_seg = x[:, s:e, :]
                    y_seg = y[:, s:e]
                    m_seg = m[:, s:e].clone()
                    if s > 0:
                        m_seg[:, 0] = 0.0

                    if isinstance(self.model, LSTMRegressor):
                        pred_seg, rnn_state = self.model(x_seg, state=rnn_state, return_state=True)
                        if rnn_state is not None:
                            rnn_state = tuple(v.detach() for v in rnn_state)
                    else:
                        pred_seg = self.model(x_seg)

                    data_loss = masked_mse(pred_seg, y_seg, m_seg)

                    # physics residual on this segment (intervals within [s, e))
                    P_seg = P[:, s:e]
                    Tb_seg = Tb[:, s:e]
                    Ta_seg = Ta[:, s:e]
                    res_seg = ode_residual_fd(
                        T_pred=pred_seg,
                        P_W=P_seg,
                        T_bplate=Tb_seg,
                        Tamb=Ta_seg,
                        dt=dt,
                        Rcond=self.Rcond,
                        Cth=self.Cth,
                        Rconv_eff=Rconv_eff,
                    )  # (B, segT-1)

                    if res_seg.numel() == 0:
                        phys_loss = torch.tensor(0.0, device=self.device)
                    else:
                        m_pair = m_seg[:, 1:] * m_seg[:, :-1]
                        phys_loss = masked_mse(res_seg, torch.zeros_like(res_seg), m_pair)

                    loss_seg = data_loss + self.lambda_phys * phys_loss
                    loss_seg.backward()

                    batch_loss_total += float(loss_seg.detach().cpu().item())
                    batch_data_total += float(data_loss.detach().cpu().item())
                    batch_phys_total += float(phys_loss.detach().cpu().item())
                    n_seg += 1

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
                optimizer.step()

                if n_seg > 0:
                    train_losses.append(batch_loss_total / n_seg)
                    train_data_losses.append(batch_data_total / n_seg)
                    train_phys_losses.append(batch_phys_total / n_seg)

            train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
            train_data_mean = float(np.mean(train_data_losses)) if train_data_losses else float("nan")
            train_phys_mean = float(np.mean(train_phys_losses)) if train_phys_losses else float("nan")

            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in val_loader:
                        total, _, _ = self.loss(vb)
                        val_losses.append(float(total.item()))
                val_mean = float(np.mean(val_losses)) if val_losses else train_mean
                scheduler.step(val_mean)
            else:
                val_mean = train_mean

            if val_loader is not None and patience and patience > 0:
                if val_mean < (best_val - float(min_delta)):
                    best_val = float(val_mean)
                    bad_epochs = 0
                    best_state = deepcopy(self.model.state_dict())
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"[EarlyStop] No val improvement for {patience} epochs. Restoring best model (val={best_val:.6f}).")
                        if best_state is not None:
                            self.model.load_state_dict(best_state)
                        break

            self.history["epoch"].append(ep)
            self.history["train_loss"].append(train_mean)
            self.history["val_loss"].append(val_mean)
            self.history["data_loss"].append(train_data_mean)
            self.history["phys_loss"].append(train_phys_mean)

            if ep == 1 or ep % 10 == 0 or ep == epochs:
                print(f"[Epoch {ep:03d}] (TBPTT) train={train_mean:.6f} val={val_mean:.6f} | data={train_data_mean:.6f} phys={train_phys_mean:.6f}")

    def train_epochs(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        lr: float,
        patience: int = 0,
        min_delta: float = 0.0,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        best_val = float("inf")
        bad_epochs = 0
        best_state = None

        for ep in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            train_data_losses = []
            train_phys_losses = []

            for batch in train_loader:
                total, data_loss, phys_loss = self.loss(batch)
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
                optimizer.step()

                train_losses.append(float(total.item()))
                train_data_losses.append(float(data_loss.item()))
                train_phys_losses.append(float(phys_loss.item()))

            train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
            train_data_mean = float(np.mean(train_data_losses)) if train_data_losses else float("nan")
            train_phys_mean = float(np.mean(train_phys_losses)) if train_phys_losses else float("nan")

            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        total, _, _ = self.loss(batch)
                        val_losses.append(float(total.item()))
                val_mean = float(np.mean(val_losses)) if val_losses else train_mean
                scheduler.step(val_mean)
            else:
                val_mean = train_mean

            # --- Early stopping on val loss ---
            if val_loader is not None and patience and patience > 0:
                if val_mean < (best_val - float(min_delta)):
                    best_val = float(val_mean)
                    bad_epochs = 0
                    best_state = deepcopy(self.model.state_dict())
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"[EarlyStop] No val improvement for {patience} epochs. Restoring best model (val={best_val:.6f}).")
                        if best_state is not None:
                            self.model.load_state_dict(best_state)
                        break

            self.history["epoch"].append(ep)
            self.history["train_loss"].append(train_mean)
            self.history["val_loss"].append(val_mean)
            self.history["data_loss"].append(train_data_mean)
            self.history["phys_loss"].append(train_phys_mean)

            if ep == 1 or ep % 10 == 0 or ep == epochs:
                print(f"[Epoch {ep:03d}] train={train_mean:.6f} val={val_mean:.6f}")


# ---------------------------------------------------------------------
# Plotting helpers (unified)
# ---------------------------------------------------------------------
def save_example_plot_generic(
    *,
    title: str,
    t_axis: np.ndarray,
    series: List[Tuple[str, np.ndarray, Dict]],
    out_path: Path,
):
    plt.figure(figsize=(10, 4))
    for name, y, style in series:
        plt.plot(t_axis, y, label=name, **style)
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def save_training_curves_ode(arch_tag: str, history: Dict, outdir: Path):
    ep = history["epoch"]
    if not ep:
        print(f"[Warn] No history for {arch_tag}")
        return

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(ep, history["train_loss"], label="train")
    axs[0, 0].plot(ep, history["val_loss"], label="val")
    axs[0, 0].set_title("Train / Val loss")
    axs[0, 0].set_xlabel("Epoch"); axs[0, 0].set_ylabel("Loss")
    axs[0, 0].grid(True, alpha=0.3); axs[0, 0].legend()

    axs[0, 1].plot(ep, history["R_cond"])
    axs[0, 1].set_title("R_cond evolution")
    axs[0, 1].set_xlabel("Epoch"); axs[0, 1].set_ylabel("R_cond [K/W]")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(ep, history["C_th"])
    axs[1, 0].set_title("C_th evolution")
    axs[1, 0].set_xlabel("Epoch"); axs[1, 0].set_ylabel("C_th [J/K]")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(ep, history["u_reg"])
    axs[1, 1].set_title("u-regularization term")
    axs[1, 1].set_xlabel("Epoch"); axs[1, 1].set_ylabel("lambda_u * mean(u^2)")
    axs[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"{arch_tag.upper()} - Training curves", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = outdir / f"{arch_tag}_training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def save_training_curves_blackbox(arch_tag: str, history: Dict, outdir: Path):
    ep = history["epoch"]
    if not ep:
        print(f"[Warn] No history for {arch_tag}")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(ep, history["train_loss"], label="train")
    plt.plot(ep, history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{arch_tag.upper()} - BLACK-PINN training curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = outdir / f"{arch_tag}_blackpinn_training_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def save_final_comparison_plot(
    outdir: Path,
    arch_list: List[str],
    white_test_rmse_meas: float,
    gray_test_rmse_meas: Dict[str, float],
    black_test_rmse_meas: Dict[str, float],
    test_scen_id: int,
):
    labels = [a.upper() for a in arch_list]
    x = np.arange(len(labels))
    width = 0.25

    white_vals = np.array([white_test_rmse_meas] * len(labels), dtype=float)
    gray_vals = np.array([gray_test_rmse_meas[a] for a in arch_list], dtype=float)
    black_vals = np.array([black_test_rmse_meas[a] for a in arch_list], dtype=float)

    plt.figure(figsize=(10, 4))
    plt.bar(x - width, white_vals, width, label="WHITE-BOX (ODE only)")
    plt.bar(x, gray_vals, width, label="GRAY-BOX (ODE + NN)")
    plt.bar(x + width, black_vals, width, label="BLACK-PINN (NN + physics residual)")
    plt.xticks(x, labels)
    plt.ylabel("RMSE on FULL test scenario (°C) vs T_case_meas")
    plt.title(f"Final comparison on held-out test scenario id={test_scen_id}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    out_path = outdir / "final_comparison_test_rmse.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Final] Saved: {out_path}")


def save_param_bar_comparison(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    values: Dict[str, float],
):
    """
    Simple bar chart for one parameter.
    Expected keys include: TRUE, WHITE, GRAY_<ARCH> (e.g., GRAY_MLP).
    Legend is placed outside to avoid covering the bars.
    """
    labels = list(values.keys())
    y = [float(values[k]) for k in labels]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    bars = ax.bar(x, y)

    # Visual emphasis:
    # - TRUE: thicker edge
    # - WHITE: dashed edge
    # - GRAY_*: hatched (generic)
    for i, lab in enumerate(labels):
        bars[i].set_edgecolor("black")
        if lab == "TRUE":
            bars[i].set_linewidth(2.2)
        elif lab == "WHITE":
            bars[i].set_linewidth(1.8)
            bars[i].set_linestyle("--")
        else:
            bars[i].set_linewidth(1.0)
            bars[i].set_hatch("..")

    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor=bars[labels.index("TRUE")].get_facecolor(), edgecolor="black", linewidth=2.2, label="TRUE (ground truth)"),
        Patch(facecolor=bars[labels.index("WHITE")].get_facecolor(), edgecolor="black", linewidth=1.8, linestyle="--", label="WHITE-BOX estimate"),
        Patch(facecolor="white", edgecolor="black", hatch="..", label="GRAY-BOX estimates (hatched)"),
    ]

    leg = ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=0.78)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", bbox_extra_artists=(leg,), pad_inches=0.25)
    plt.close(fig)
    print(f"[Final] Saved: {out_path}")


def save_rmse_variant_arch_plot(
    *,
    out_path: Path,
    test_scen_id: int,
    white_rmse: float,
    gray_rmse_by_arch: Dict[str, float],
    black_rmse_by_arch: Dict[str, float],
    arch_list: List[str],
):
    """
    RMSE chart:
      - color encodes model variant (WHITE/GRAY/BLACK)
      - hatch encodes architecture (MLP/TCN/LSTM)
    Legend placed outside and included in the saved image (no cropping).
    """
    # Colors (explicit as requested)
    variant_color = {"WHITE": "#9e9e9e", "GRAY": "#1f77b4", "BLACK": "#ff7f0e"}  # gray, blue, orange
    arch_hatch = {"mlp": "//", "tcn": "\\\\", "lstm": "xx"}

    fig, ax = plt.subplots(figsize=(12.5, 4.8))

    groups = ["WHITE"] + [a.upper() for a in arch_list]
    gx = np.arange(len(groups))
    width = 0.32

    # WHITE single bar
    ax.bar(gx[0], float(white_rmse), width=0.6, color=variant_color["WHITE"], edgecolor="black", linewidth=1.2)

    # GRAY/BLACK per-architecture
    for i, arch in enumerate(arch_list, start=1):
        g = float(gray_rmse_by_arch.get(arch, np.nan))
        b = float(black_rmse_by_arch.get(arch, np.nan))

        ax.bar(
            gx[i] - width / 2,
            g,
            width=width,
            color=variant_color["GRAY"],
            edgecolor="black",
            linewidth=1.0,
            hatch=arch_hatch.get(arch, ""),
        )
        ax.bar(
            gx[i] + width / 2,
            b,
            width=width,
            color=variant_color["BLACK"],
            edgecolor="black",
            linewidth=1.0,
            hatch=arch_hatch.get(arch, ""),
        )

    ax.set_xticks(gx, groups)
    ax.set_ylabel("RMSE [°C] vs T_case_meas (FULL test scenario)")
    ax.set_title(f"RMSE comparison (test scenario id={test_scen_id}): color=variant, hatch=architecture")
    ax.grid(True, axis="y", alpha=0.3)

    variant_legend = [
        Patch(facecolor=variant_color["WHITE"], edgecolor="black", label="WHITE-BOX"),
        Patch(facecolor=variant_color["GRAY"], edgecolor="black", label="GRAY-BOX"),
        Patch(facecolor=variant_color["BLACK"], edgecolor="black", label="BLACK-PINN"),
    ]
    arch_legend = [
        Patch(facecolor="white", edgecolor="black", hatch=arch_hatch["mlp"], label="MLP"),
        Patch(facecolor="white", edgecolor="black", hatch=arch_hatch["tcn"], label="TCN"),
        Patch(facecolor="white", edgecolor="black", hatch=arch_hatch["lstm"], label="LSTM"),
    ]

    leg1 = ax.legend(handles=variant_legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, title="Variant")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=arch_legend, loc="upper left", bbox_to_anchor=(1.02, 0.55), borderaxespad=0.0, title="Architecture")

    # Make sure legends aren't cropped
    fig.subplots_adjust(right=0.78)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", bbox_extra_artists=(leg1, leg2), pad_inches=0.25)
    plt.close(fig)
    print(f"[Final] Saved: {out_path}")


# ---------------------------------------------------------------------
# Full-scenario evaluation (unified)
# ---------------------------------------------------------------------
def load_full_scenario(csv_path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = ["time_s", "P_W", "Tamb_C", "T_bplate_C", "T_case_true_C", "T_case_meas_C"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"{csv_path}: column '{col}' not found.")
    return {
        "t": df["time_s"].to_numpy(float),
        "P": df["P_W"].to_numpy(float),
        "Tamb": df["Tamb_C"].to_numpy(float),
        "Tb": df["T_bplate_C"].to_numpy(float),
        "Ttrue": df["T_case_true_C"].to_numpy(float),
        "Tmeas": df["T_case_meas_C"].to_numpy(float),
    }


def normalize_inputs_full(P, Tb, Tamb, ds_norm: SyntheticThermalWindowDataset) -> np.ndarray:
    x = np.stack([P, Tb, Tamb], axis=-1)[None, ...]  # (1,T,3)
    x = x.astype(float)
    for ch in range(3):
        bad = ~np.isfinite(x[:, :, ch])
        x[:, :, ch][bad] = ds_norm.input_mean[ch]
    mean = ds_norm.input_mean.reshape(1, 1, 3)
    std = ds_norm.input_std.reshape(1, 1, 3)
    return (x - mean) / std


def eval_full_test_ode(
    *,
    tag: str,
    model: nn.Module,
    R_cond_hat: float,
    C_th_hat: float,
    Rconv_test: float,
    csv_path: Path,
    ds_norm: SyntheticThermalWindowDataset,
    device: torch.device,
    lambda_u: float,
    use_nn: bool,
    outdir: Path,
    integrator: str = "euler",
) -> Tuple[float, float]:
    d = load_full_scenario(csv_path)
    t, P, Tamb, Tb, Ttrue, Tmeas = d["t"], d["P"], d["Tamb"], d["Tb"], d["Ttrue"], d["Tmeas"]
    dt = robust_dt(t)
    Tlen = len(t)

    inputs_norm = normalize_inputs_full(P, Tb, Tamb, ds_norm)
    inputs_t = torch.tensor(inputs_norm, dtype=torch.float32, device=device)  # (1,T,3)

    P_t = torch.tensor(P.reshape(1, -1), dtype=torch.float32, device=device)
    Tb_t = torch.tensor(Tb.reshape(1, -1), dtype=torch.float32, device=device)
    Ta_t = torch.tensor(Tamb.reshape(1, -1), dtype=torch.float32, device=device)
    dt_t = torch.tensor([dt], dtype=torch.float32, device=device)

    Rcond = torch.tensor(float(R_cond_hat), dtype=torch.float32, device=device)
    Cth = torch.tensor(float(C_th_hat), dtype=torch.float32, device=device)
    Rconv_eff = torch.tensor([[float(Rconv_test)]], dtype=torch.float32, device=device)

    T0 = torch.tensor([Tmeas[0]], dtype=torch.float32, device=device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        T_pred, _ = ode_forward(
            model,
            inputs=inputs_t,
            P_W=P_t,
            T_bplate=Tb_t,
            Tamb=Ta_t,
            dt=dt_t,
            T0=T0,
            Rcond=Rcond,
            Cth=Cth,
            Rconv_eff=Rconv_eff,
            use_nn=use_nn,
        )

    T_pred_np = T_pred[0].detach().cpu().numpy()
    rmse_meas = float(np.sqrt(np.mean((T_pred_np - Tmeas) ** 2)))
    rmse_true = float(np.sqrt(np.mean((T_pred_np - Ttrue) ** 2)))

    out_path = outdir / f"{tag}_test_full.png"
    save_example_plot_generic(
        title=f"{tag.upper()} - FULL test scenario",
        t_axis=t,
        series=[
            ("T_case pred", T_pred_np, {"linewidth": 1.5}),
            ("T_case meas", Tmeas, {"linestyle": "--", "linewidth": 1.0}),
            ("T_case true", Ttrue, {"linestyle": ":", "linewidth": 1.0}),
        ],
        out_path=out_path,
    )
    return rmse_meas, rmse_true


def eval_full_test_blackbox(
    *,
    tag: str,
    model: nn.Module,
    csv_path: Path,
    ds_norm: SyntheticThermalWindowDataset,
    device: torch.device,
    outdir: Path,
) -> Tuple[float, float]:
    d = load_full_scenario(csv_path)
    t, P, Tamb, Tb, Ttrue, Tmeas = d["t"], d["P"], d["Tamb"], d["Tb"], d["Ttrue"], d["Tmeas"]

    inputs_norm = normalize_inputs_full(P, Tb, Tamb, ds_norm)
    inputs_t = torch.tensor(inputs_norm, dtype=torch.float32, device=device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        T_pred = model(inputs_t)[0].detach().cpu().numpy()

    rmse_meas = float(np.sqrt(np.mean((T_pred - Tmeas) ** 2)))
    rmse_true = float(np.sqrt(np.mean((T_pred - Ttrue) ** 2)))

    out_path = outdir / f"{tag}_blackbox_test_full.png"
    save_example_plot_generic(
        title=f"{tag.upper()} - BLACK-BOX FULL test scenario",
        t_axis=t,
        series=[
            ("T_case pred (NN)", T_pred, {"linewidth": 1.5}),
            ("T_case meas", Tmeas, {"linestyle": "--", "linewidth": 1.0}),
            ("T_case true", Ttrue, {"linestyle": ":", "linewidth": 1.0}),
        ],
        out_path=out_path,
    )
    return rmse_meas, rmse_true


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Training WHITE/GRAY/BLACK on synthetic thermal dataset "
            "(MLP/TCN/LSTM). Learn R_cond & C_th, fixed R_conv per scenario, "
            "1 held-out full-sequence test scenario."
        )
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--train-window", type=int, default=None,  help="Training window length for TBPTT (e.g., 2048/4096). If None, uses --window.")
    ap.add_argument("--tbptt-len", type=int, default=None, help="TBPTT truncation length. If None, uses --window. Must be >=2.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lambda-u", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arch", choices=["mlp", "tcn", "lstm", "all"], default="all")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--no-nn", action="store_true")
    ap.add_argument("--lambda-phys", type=float, default=1e-5, help="Weight of physics residual loss in BLACK-PINN training.")
    ap.add_argument("--Rcond-init", type=float, default=1.0)
    ap.add_argument("--Cth-init-scale", type=float, default=0.2)
    ap.add_argument("--Rconv-scale", type=float, default=1.0)
    ap.add_argument("--test-scen-id", type=int, default=-1)
    ap.add_argument("--integrator", choices=["euler", "rk4"], default="euler", help="ODE integrator for WHITE/GRAY forward simulation.")

    ap.add_argument("--phys-prior-weight", type=float, default=1.0)
    ap.add_argument("--Rcond-prior", type=float, default=None)
    ap.add_argument("--Cth-prior", type=float, default=None)

    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss (0=disabled).")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="Minimum val loss decrease to count as improvement.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_determinism(args.seed)

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / "true_params.json"
    if not json_path.exists():
        raise SystemExit(f"{json_path} not found. Run make_synthetic_thermal_dataset.py first.")

    with open(json_path, "r") as f:
        meta = json.load(f)

    Cth_true = float(meta["C_th_true"])
    Rcond_true = float(meta["R_cond_true"])
    scenarios = sorted(meta["scenarios"], key=lambda s: s["scenario_id"])

    # select test scenario
    test_scen_id = scenarios[-1]["scenario_id"] if args.test_scen_id < 0 else args.test_scen_id
    test_meta = next((s for s in scenarios if s["scenario_id"] == test_scen_id), None)
    if test_meta is None:
        raise SystemExit(f"Test scenario with id={test_scen_id} not found in true_params.json")

    test_csv_file = test_meta["csv_file"]
    test_csv_path = data_dir / test_csv_file
    Rconv_test_true = float(test_meta["R_conv_true"])

    train_scenarios = [s for s in scenarios if s["scenario_id"] != test_scen_id]
    train_csv_files = [s["csv_file"] for s in train_scenarios]

    Rconv_scale = float(args.Rconv_scale)
    Rconv_train_list = [float(s["R_conv_true"]) * Rconv_scale for s in train_scenarios]
    Rconv_test_used = Rconv_test_true * Rconv_scale

    print("=== Ground truth parameters ===")
    print(f"C_th_true   = {Cth_true:.4f} J/K")
    print(f"R_cond_true = {Rcond_true:.4f} K/W")
    print(f"[R_conv] Global scaling factor: {Rconv_scale:.4f}")
    print(f"[Split] Held-out TEST scenario: id={test_scen_id}, file={test_csv_file}")
    print(f"[Split] TRAIN scenarios: {[s['scenario_id'] for s in train_scenarios]}")

    print("\n[Main] Building dataloaders (train scenarios only)...")
    ds, train_loader, val_loader = make_dataloaders(
        data_dir=data_dir,
        csv_files=train_csv_files,
        window=args.window,
        batch_size=args.batch,
        val_fraction=args.val_frac,
        seed=args.seed,
    )

    device = torch.device(args.device)
    arch_list = ["mlp", "tcn", "lstm"] if args.arch == "all" else [args.arch]

    # shared init
    Cth_init = Cth_true * float(args.Cth_init_scale)
    Rcond_init = float(args.Rcond_init)

    # -----------------------------------------------------------------
    # WHITE-BOX
    # -----------------------------------------------------------------
    print("\n===== Training WHITE-BOX baseline (ODE only) =====")
    white_outdir = outdir / "whitebox"
    white_outdir.mkdir(parents=True, exist_ok=True)

    white_cfg = ODEConfig(
        lambda_u=args.lambda_u,
        clip_grad=5.0,
        use_nn=False,
        Rcond_prior=args.Rcond_prior,
        Cth_prior=args.Cth_prior,
        phys_prior_weight=args.phys_prior_weight,
        integrator=args.integrator,
    )

    white_model = nn.Identity()
    white_tr = ODETrainer(
        white_model,
        Cth_init=Cth_init,
        Rcond_init=Rcond_init,
        Rconv_list=Rconv_train_list,
        device=args.device,
        cfg=white_cfg,
    )

    # --- Optional TBPTT training on long windows ---
    train_window = args.train_window if args.train_window is not None else args.window
    tbptt_len = args.tbptt_len if args.tbptt_len is not None else args.window
    if int(train_window) > int(args.window):
        # build long-window loader for training, reuse normalization from ds
        ds_long = SyntheticThermalWindowDataset(
            [str(data_dir / name) for name in train_csv_files],
            window_size=int(train_window),
            stride=max(1, int(train_window) // 2),
            preload=True,
        )
        ds_long.input_mean = ds.input_mean.copy()
        ds_long.input_std = ds.input_std.copy()
        train_loader_long = DataLoader(ds_long, batch_size=args.batch, shuffle=True, collate_fn=ds_long.collate_fn)

        white_tr.train_epochs_tbptt(
            train_loader_long,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            lr_phys=args.lr * 10.0,
            tbptt_len=int(tbptt_len),
            patience=args.patience,
            min_delta=args.min_delta,
        )
    else:
        white_tr.train_epochs(
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            lr_phys=args.lr * 10.0,
            patience=args.patience,
            min_delta=args.min_delta,
        )

    white_params = white_tr.export_params()
    white_ckpt = {
        "variant": "whitebox",
        "log_Rcond": white_tr.log_Rcond.detach().cpu().numpy().tolist(),
        "log_Cth": white_tr.log_Cth.detach().cpu().numpy().tolist(),
        "C_th_true": Cth_true,
        "R_cond_true": Rcond_true,
        "R_conv_train_list": Rconv_train_list,
        "test_scen_id": test_scen_id,
        "R_conv_test_true": Rconv_test_true,
        "R_conv_test_used": Rconv_test_used,
        "Rconv_scale": Rconv_scale,
        "dataset_stats": {"input_mean": ds.input_mean.tolist(), "input_std": ds.input_std.tolist()},
        "history": white_tr.history,
    }
    torch.save(white_ckpt, white_outdir / "whitebox_final.pt")
    print(f"[Main] Saved WHITE-BOX checkpoint to {white_outdir / 'whitebox_final.pt'}")

    # window RMSE
    white_rmse_meas_val = rmse_from_loader(lambda b: white_tr.predict(b)[0], val_loader, device, "T_case_meas")
    white_rmse_true_val = rmse_from_loader(lambda b: white_tr.predict(b)[0], val_loader, device, "T_case_true")
    print(f"[Post] WHITE-BOX RMSE (val) vs T_case_meas = {white_rmse_meas_val:.4f} °C")
    print(f"[Post] WHITE-BOX RMSE (val) vs T_case_true = {white_rmse_true_val:.4f} °C")

    # val example plot (first batch)
    batch = next(iter(val_loader))
    batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        T_pred, _ = white_tr.predict(batch_dev)
    dt0 = float(batch_dev["dt"][0].cpu().numpy())
    t_axis = np.arange(T_pred.shape[1]) * dt0
    mask0 = batch_dev["mask"][0].cpu().numpy()
    meas0 = np.where(mask0 > 0.5, batch_dev["T_case_meas"][0].cpu().numpy(), np.nan)
    true0 = np.where(mask0 > 0.5, batch_dev["T_case_true"][0].cpu().numpy(), np.nan)
    pred0 = T_pred[0].cpu().numpy()
    save_example_plot_generic(
        title="WHITEBOX - Validation example",
        t_axis=t_axis,
        series=[
            ("T_case pred", pred0, {"linewidth": 1.5}),
            ("T_case meas", meas0, {"linestyle": "--", "linewidth": 1.0}),
            ("T_case true", true0, {"linestyle": ":", "linewidth": 1.0}),
        ],
        out_path=white_outdir / "whitebox_val_example.png",
    )
    save_training_curves_ode("whitebox", white_tr.history, white_outdir)

    white_rmse_test_meas, white_rmse_test_true = eval_full_test_ode(
        tag="whitebox",
        model=white_model,
        R_cond_hat=white_params["R_cond"],
        C_th_hat=white_params["C_th"],
        Rconv_test=Rconv_test_used,
        csv_path=test_csv_path,
        ds_norm=ds,
        device=device,
        lambda_u=args.lambda_u,
        use_nn=False,
        outdir=white_outdir,
        integrator=args.integrator,
    )
    print(f"[Test] WHITE-BOX on test scenario id={test_scen_id}:")
    print(f"       RMSE (FULL) vs T_case_meas = {white_rmse_test_meas:.4f} °C")
    print(f"       RMSE (FULL) vs T_case_true = {white_rmse_test_true:.4f} °C")

    # -----------------------------------------------------------------
    # GRAY + BLACK per architecture
    # -----------------------------------------------------------------
    final_gray_test_rmse_meas: Dict[str, float] = {}
    final_black_test_rmse_meas: Dict[str, float] = {}
    final_gray_params: Dict[str, Dict[str, float]] = {}

    for arch in arch_list:
        print(f"\n===== Training arch: {arch.upper()} (GRAY-BOX then BLACK-BOX) =====")
        model_outdir = outdir / arch
        model_outdir.mkdir(parents=True, exist_ok=True)

        # ---------------- GRAY-BOX ----------------
        gray_model = make_model(arch, args.hidden)
        gray_cfg = ODEConfig(
            lambda_u=args.lambda_u,
            clip_grad=5.0,
            use_nn=not args.no_nn,
            Rcond_prior=args.Rcond_prior,
            Cth_prior=args.Cth_prior,
            phys_prior_weight=args.phys_prior_weight,
        )
        gray_tr = ODETrainer(
            gray_model,
            Cth_init=Cth_init,
            Rcond_init=Rcond_init,
            Rconv_list=Rconv_train_list,
            device=args.device,
            cfg=gray_cfg,
        )

        # --- Optional TBPTT training on long windows ---
        if int(train_window) > int(args.window):
            ds_long = SyntheticThermalWindowDataset(
                [str(data_dir / name) for name in train_csv_files],
                window_size=int(train_window),
                stride=max(1, int(train_window) // 2),
                preload=True,
            )
            ds_long.input_mean = ds.input_mean.copy()
            ds_long.input_std = ds.input_std.copy()
            train_loader_long = DataLoader(ds_long, batch_size=args.batch, shuffle=True, collate_fn=ds_long.collate_fn)

            gray_tr.train_epochs_tbptt(
                train_loader_long,
                val_loader,
                epochs=args.epochs,
                lr=args.lr,
                lr_phys=args.lr * 10.0,
                tbptt_len=int(tbptt_len),
                patience=args.patience,
                min_delta=args.min_delta,
            )
        else:
            gray_tr.train_epochs(
                train_loader,
                val_loader,
                epochs=args.epochs,
                lr=args.lr,
                lr_phys=args.lr * 10.0,
                patience=args.patience,
                min_delta=args.min_delta,
            )

        gray_params = gray_tr.export_params()
        final_gray_params[arch] = dict(gray_params)
        ckpt = {
            "variant": "graybox",
            "model_state": gray_model.state_dict(),
            "log_Rcond": gray_tr.log_Rcond.detach().cpu().numpy().tolist(),
            "log_Cth": gray_tr.log_Cth.detach().cpu().numpy().tolist(),
            "C_th_true": Cth_true,
            "R_cond_true": Rcond_true,
            "R_conv_train_list": Rconv_train_list,
            "test_scen_id": test_scen_id,
            "R_conv_test_true": Rconv_test_true,
            "R_conv_test_used": Rconv_test_used,
            "Rconv_scale": Rconv_scale,
            "dataset_stats": {"input_mean": ds.input_mean.tolist(), "input_std": ds.input_std.tolist()},
            "history": gray_tr.history,
        }
        ckpt_path = model_outdir / f"{arch}_graybox_final.pt"
        torch.save(ckpt, ckpt_path)
        print(f"[Main] Saved GRAY-BOX checkpoint to {ckpt_path}")

        # window metrics
        gray_rmse_meas_val = rmse_from_loader(lambda b: gray_tr.predict(b)[0], val_loader, device, "T_case_meas")
        gray_rmse_true_val = rmse_from_loader(lambda b: gray_tr.predict(b)[0], val_loader, device, "T_case_true")
        print(f"[Post] {arch.upper()} GRAY-BOX RMSE (val) vs T_case_meas = {gray_rmse_meas_val:.4f} °C")
        print(f"[Post] {arch.upper()} GRAY-BOX RMSE (val) vs T_case_true = {gray_rmse_true_val:.4f} °C")

        val_stats_meas = error_stats_from_loader(lambda b: gray_tr.predict(b)[0], val_loader, device, "T_case_meas")
        val_stats_true = error_stats_from_loader(lambda b: gray_tr.predict(b)[0], val_loader, device, "T_case_true")
        print(f"[Safety] {arch.upper()} GRAY-BOX val vs T_case_meas: {val_stats_meas}")
        print(f"[Safety] {arch.upper()} GRAY-BOX val vs T_case_true: {val_stats_true}")

        # example plot from val batch
        batch = next(iter(val_loader))
        batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            T_pred, _ = gray_tr.predict(batch_dev)
        dt0 = float(batch_dev["dt"][0].cpu().numpy())
        t_axis = np.arange(T_pred.shape[1]) * dt0
        mask0 = batch_dev["mask"][0].cpu().numpy()
        meas0 = np.where(mask0 > 0.5, batch_dev["T_case_meas"][0].cpu().numpy(), np.nan)
        true0 = np.where(mask0 > 0.5, batch_dev["T_case_true"][0].cpu().numpy(), np.nan)
        pred0 = T_pred[0].cpu().numpy()
        save_example_plot_generic(
            title=f"{arch.upper()} GRAY-BOX - Validation example",
            t_axis=t_axis,
            series=[
                ("T_case pred", pred0, {"linewidth": 1.5}),
                ("T_case meas", meas0, {"linestyle": "--", "linewidth": 1.0}),
                ("T_case true", true0, {"linestyle": ":", "linewidth": 1.0}),
            ],
            out_path=model_outdir / f"{arch}_graybox_val_example.png",
        )
        save_training_curves_ode(f"{arch}_graybox", gray_tr.history, model_outdir)

        # full test
        rmse_test_meas, rmse_test_true = eval_full_test_ode(
            tag=f"{arch}_graybox",
            model=gray_model,
            R_cond_hat=gray_params["R_cond"],
            C_th_hat=gray_params["C_th"],
            Rconv_test=Rconv_test_used,
            csv_path=test_csv_path,
            ds_norm=ds,
            device=device,
            lambda_u=args.lambda_u,
            use_nn=not args.no_nn,
            outdir=model_outdir,
            integrator=args.integrator,
        )
        print(f"[Test] {arch.upper()} GRAY-BOX on test scenario id={test_scen_id}:")
        print(f"       RMSE (FULL) vs T_case_meas = {rmse_test_meas:.4f} °C")
        print(f"       RMSE (FULL) vs T_case_true = {rmse_test_true:.4f} °C")
        final_gray_test_rmse_meas[arch] = rmse_test_meas

        # ---------------- BLACK-PINN ----------------
        print(f"\n----- Training {arch.upper()} BLACK-PINN (NN + physics residual) -----")
        bb_model = make_model(arch, args.hidden)
        bb_tr = BlackPINNTrainer(
            bb_model,
            device=args.device,
            clip_grad=5.0,
            lambda_phys=args.lambda_phys,
            Rcond_fixed=float(white_params["R_cond"]),
            Cth_fixed=float(white_params["C_th"]),
            Rconv_list=Rconv_train_list,
        )
 
        if int(train_window) > int(args.window):
            ds_long = SyntheticThermalWindowDataset(
                [str(data_dir / name) for name in train_csv_files],
                window_size=int(train_window),
                stride=max(1, int(train_window) // 2),
                preload=True,
            )
            ds_long.input_mean = ds.input_mean.copy()
            ds_long.input_std = ds.input_std.copy()
            train_loader_long = DataLoader(ds_long, batch_size=args.batch, shuffle=True, collate_fn=ds_long.collate_fn)

            bb_tr.train_epochs_tbptt(
                train_loader_long,
                val_loader,
                epochs=args.epochs,
                lr=args.lr,
                tbptt_len=int(tbptt_len),
                patience=args.patience,
                min_delta=args.min_delta,
            )
        else:
            bb_tr.train_epochs(
                train_loader,
                val_loader,
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                min_delta=args.min_delta,
            )

        bb_ckpt = {
            "arch": arch,
            "variant": "blackpinn",
            "model_state": bb_model.state_dict(),
            "C_th_true": Cth_true,
            "R_cond_true": Rcond_true,
            "R_conv_train_list": Rconv_train_list,
            "test_scen_id": test_scen_id,
            "R_conv_test_true": Rconv_test_true,
            "R_conv_test_used": Rconv_test_used,
            "Rconv_scale": Rconv_scale,
            "whitebox_phys_used": {"R_cond_hat": float(white_params["R_cond"]), "C_th_hat": float(white_params["C_th"])},
            "lambda_phys": float(args.lambda_phys),
            "dataset_stats": {"input_mean": ds.input_mean.tolist(), "input_std": ds.input_std.tolist()},
            "history": bb_tr.history,
        }
        bb_ckpt_path = model_outdir / f"{arch}_blackpinn_final.pt"
        torch.save(bb_ckpt, bb_ckpt_path)
        print(f"[Main] Saved BLACK-PINN checkpoint to {bb_ckpt_path}")

        bb_rmse_meas_val = rmse_from_loader(lambda b: bb_tr.predict(b), val_loader, device, "T_case_meas")
        bb_rmse_true_val = rmse_from_loader(lambda b: bb_tr.predict(b), val_loader, device, "T_case_true")
        print(f"[Post] {arch.upper()} BLACK-BOX RMSE (val) vs T_case_meas = {bb_rmse_meas_val:.4f} °C")
        print(f"[Post] {arch.upper()} BLACK-BOX RMSE (val) vs T_case_true = {bb_rmse_true_val:.4f} °C")

        bb_val_stats_meas = error_stats_from_loader(lambda b: bb_tr.predict(b), val_loader, device, "T_case_meas")
        bb_val_stats_true = error_stats_from_loader(lambda b: bb_tr.predict(b), val_loader, device, "T_case_true")
        print(f"[Safety] {arch.upper()} BLACK-PINN val vs T_case_meas: {bb_val_stats_meas}")
        print(f"[Safety] {arch.upper()} BLACK-PINN val vs T_case_true: {bb_val_stats_true}")

        # val example plot
        batch = next(iter(val_loader))
        batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            T_pred = bb_tr.predict(batch_dev)
        dt0 = float(batch_dev["dt"][0].cpu().numpy())
        t_axis = np.arange(T_pred.shape[1]) * dt0
        mask0 = batch_dev["mask"][0].cpu().numpy()
        meas0 = np.where(mask0 > 0.5, batch_dev["T_case_meas"][0].cpu().numpy(), np.nan)
        true0 = np.where(mask0 > 0.5, batch_dev["T_case_true"][0].cpu().numpy(), np.nan)
        pred0 = T_pred[0].cpu().numpy()
        save_example_plot_generic(
            title=f"{arch.upper()} BLACK-PINN - Validation example",
            t_axis=t_axis,
            series=[
                ("T_case pred (NN)", pred0, {"linewidth": 1.5}),
                ("T_case meas", meas0, {"linestyle": "--", "linewidth": 1.0}),
                ("T_case true", true0, {"linestyle": ":", "linewidth": 1.0}),
            ],
            out_path=model_outdir / f"{arch}_blackpinn_val_example.png",
        )
        save_training_curves_blackbox(arch, bb_tr.history, model_outdir)

        bb_rmse_test_meas, bb_rmse_test_true = eval_full_test_blackbox(
            tag=arch,
            model=bb_model,
            csv_path=test_csv_path,
            ds_norm=ds,
            device=device,
            outdir=model_outdir,
        )
        print(f"[Test] {arch.upper()} BLACK-PINN on test scenario id={test_scen_id}:")
        print(f"       RMSE (FULL) vs T_case_meas = {bb_rmse_test_meas:.4f} °C")
        print(f"       RMSE (FULL) vs T_case_true = {bb_rmse_test_true:.4f} °C")
        final_black_test_rmse_meas[arch] = bb_rmse_test_meas

    # -----------------------------------------------------------------
    # Final comparison artifacts (PLOT + CSV)  <-- FIX: previously missing
    # -----------------------------------------------------------------
    # Bar plot across architectures for WHITE/GRAY/BLACK on FULL test scenario
    save_final_comparison_plot(
        outdir=outdir,
        arch_list=arch_list,
        white_test_rmse_meas=white_rmse_test_meas,
        gray_test_rmse_meas=final_gray_test_rmse_meas,
        black_test_rmse_meas=final_black_test_rmse_meas,
        test_scen_id=test_scen_id,
    )

    # CSV summary (one row per arch)
    rows = []
    for a in arch_list:
        rows.append(
            {
                "arch": a,
                "white_rmse_meas": float(white_rmse_test_meas),
                "gray_rmse_meas": float(final_gray_test_rmse_meas.get(a, np.nan)),
                "black_rmse_meas": float(final_black_test_rmse_meas.get(a, np.nan)),
                "test_scen_id": int(test_scen_id),
            }
        )
    df_cmp = pd.DataFrame(rows)
    csv_path = outdir / "final_comparison_test_rmse.csv"
    df_cmp.to_csv(csv_path, index=False)
    print(f"[Final] Saved: {csv_path}")

    # -----------------------------------------------------------------
    # Extra final figures (requested): R_cond, C_th, RMSE (variant color + arch hatch)
    # -----------------------------------------------------------------
    # 1) R_cond chart: TRUE vs WHITE vs GRAY per-arch
    rcond_vals = {"TRUE": float(Rcond_true), "WHITE": float(white_params["R_cond"])}
    for a in arch_list:
        if a in final_gray_params:
            rcond_vals[f"GRAY_{a.upper()}"] = float(final_gray_params[a]["R_cond"])
    save_param_bar_comparison(
        out_path=outdir / "final_Rcond_comparison.png",
        title="R_cond comparison: TRUE vs WHITE vs GRAY-BOX (per architecture)",
        ylabel="R_cond [K/W]",
        values=rcond_vals,
    )

    # 2) C_th chart: TRUE vs WHITE vs GRAY per-arch
    cth_vals = {"TRUE": float(Cth_true), "WHITE": float(white_params["C_th"])}
    for a in arch_list:
        if a in final_gray_params:
            cth_vals[f"GRAY_{a.upper()}"] = float(final_gray_params[a]["C_th"])
    save_param_bar_comparison(
        out_path=outdir / "final_Cth_comparison.png",
        title="C_th comparison: TRUE vs WHITE vs GRAY-BOX (per architecture)",
        ylabel="C_th [J/K]",
        values=cth_vals,
    )

    # 3) RMSE chart: variant=color, architecture=hatch (legend outside, not cropped)
    save_rmse_variant_arch_plot(
        out_path=outdir / "final_RMSE_variant_arch.png",
        test_scen_id=int(test_scen_id),
        white_rmse=float(white_rmse_test_meas),
        gray_rmse_by_arch=final_gray_test_rmse_meas,
        black_rmse_by_arch=final_black_test_rmse_meas,
        arch_list=arch_list,
    )

    # -----------------------------------------------------------------
    # Improved final summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    # Split info
    print(f"Dataset split:")
    print(f"  TEST scenario id={test_scen_id}  file={test_csv_file}")
    print(f"  TRAIN scenarios: {[s['scenario_id'] for s in train_scenarios]}")
    print(f"R_conv scaling used by the model: Rconv_scale = {Rconv_scale:.4f}")
    print(f"  TEST R_conv_true = {Rconv_test_true:.4f} K/W | R_conv_used = {Rconv_test_used:.4f} K/W")

    # Physical parameters: ground-truth vs estimated
    print("\nPhysical parameter identification (ground truth vs estimates):")
    print(f"  Ground truth: R_cond = {Rcond_true:.6f} K/W | C_th = {Cth_true:.6f} J/K")

    print(f"  WHITE-BOX    : R_cond_hat = {white_params['R_cond']:.6f} K/W "
          f"({(white_params['R_cond']-Rcond_true)/Rcond_true*100:+.2f}%) | "
          f"C_th_hat = {white_params['C_th']:.6f} J/K "
          f"({(white_params['C_th']-Cth_true)/Cth_true*100:+.2f}%)")

    for arch in arch_list:
        # you have gray params inside the loop; here we reconstruct from saved dicts
        gray_ckpt_path = outdir / arch / f"{arch}_graybox_final.pt"
        if gray_ckpt_path.exists():
            ck = torch.load(gray_ckpt_path, map_location="cpu")
            R_hat = float(np.exp(ck["log_Rcond"]))
            C_hat = float(np.exp(ck["log_Cth"]))
            print(f"  {arch.upper():<10} : R_cond_hat = {R_hat:.6f} K/W "
                  f"({(R_hat-Rcond_true)/Rcond_true*100:+.2f}%) | "
                  f"C_th_hat = {C_hat:.6f} J/K "
                  f"({(C_hat-Cth_true)/Cth_true*100:+.2f}%)")
        else:
            print(f"  {arch.upper():<10} : (GRAY checkpoint not found)")

    # Final full-test RMSE comparison
    print("\nReconstruction accuracy on FULL held-out TEST scenario (vs T_case_meas):")
    print(f"  WHITE-BOX RMSE = {white_rmse_test_meas:.6f} °C")
    for arch in arch_list:
        g = final_gray_test_rmse_meas.get(arch, np.nan)
        b = final_black_test_rmse_meas.get(arch, np.nan)
        print(f"  {arch.upper():<5}  GRAY-BOX RMSE = {g:.6f} °C | BLACK-PINN RMSE = {b:.6f} °C")

    # Who wins
    all_models = [("white", white_rmse_test_meas)]
    all_models += [(f"gray_{a}", final_gray_test_rmse_meas.get(a, np.inf)) for a in arch_list]
    all_models += [(f"blackpinn_{a}", final_black_test_rmse_meas.get(a, np.inf)) for a in arch_list]
    best_name, best_rmse = min(all_models, key=lambda x: x[1])
    print(f"\nBest model on FULL test (RMSE vs meas): {best_name}  ->  {best_rmse:.6f} °C")

    # Remind artifacts
    print("\nArtifacts saved:")
    print(f"  - Final RMSE plot : {outdir / 'final_comparison_test_rmse.png'}")
    print(f"  - Final RMSE CSV  : {outdir / 'final_comparison_test_rmse.csv'}")
    print(f"  - R_cond plot     : {outdir / 'final_Rcond_comparison.png'}")
    print(f"  - C_th plot       : {outdir / 'final_Cth_comparison.png'}")
    print(f"  - RMSE styled plot: {outdir / 'final_RMSE_variant_arch.png'}")
    print("  - Per-model folders contain: checkpoints (*.pt), training curves, val example, full test plot")
    print("=" * 78)

if __name__ == "__main__":
    main()
