#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synthetic_thermal_dataset_nonideal.py

Generate a "less ideal", more realistic synthetic dataset for the
identification of thermal resistances R_cond and R_conv in a 1-node
(case) model.

IMPORTANT:
  - The generated CSV/JSON format is compatible with the existing
    train_ipinn_synthetic_odeforward.py script
    (same column names, same JSON keys).

PHYSICAL MODEL USED TO GENERATE DATA (NON-IDEAL):

We *simulate* a system that is slightly more complex than the model used
for training. The training model assumes:

  Case node:
    C_th * dT_case/dt = P(t)
                        - (T_case - T_bplate)/R_cond
                        - (T_case - T_amb)/R_conv

In this generator we use:

  1) A backplate node (just for generating T_bplate, not for fitting),
     thermally close to the case node (e.g. an NTC a few cm away).
     Its dynamics are:

       C_bp * dT_bplate/dt =
           alpha * P(t)
           + (T_case - T_bplate)/R_cond_eff(t)
           - (T_bplate - T_amb)/R_bp_amb

  2) A CASE NODE WITH NON-IDEALITIES:
       C_th_eff(t) * dT_case/dt =
           P(t)
           - (T_case - T_bplate)/R_cond_eff(t)
           - (T_case - T_amb)/R_conv_eff(t)
           - (T_case - T_amb)/R_extra

     where:
       - C_th_eff(t)  = C_th_true   * (1 + slow_drift_Cth(t))
       - R_cond_eff(t)= R_cond_true * (1 + slow_drift_Rcond(t))
       - R_conv_eff(t)= R_conv_true * (1 + slow_drift_Rconv(t))

     R_extra is an additional, hidden thermal leak to the ambient,
     not modeled in the PINN. This forces the NN correction term u(t)
     to be non-negligible in order to match T_case_meas.

  3) SENSOR LAG AND NOISE on T_case_meas:
       - T_case_true_C is the "true" case node temperature from ODE
       - T_case_meas_C is obtained by:
           * first-order lag filter on T_case_true
           * plus low-frequency correlated noise

  4) BACKPLATE NOISE:
       - T_bplate_C in the CSV is a noisy version of T_bplate_true

Each scenario has:
  - its own ambient temperature Tamb
  - its own nominal convective resistance R_conv_true (via airflow)
  - a "shark-fin" like power profile P(t)

Each CSV contains:
  time_s, P_W, Tamb_C, T_bplate_C, T_case_true_C, T_case_meas_C

We also save a JSON file with global "true" nominal parameters
and per-scenario information.

Example usage:
  python make_synthetic_thermal_dataset_nonideal.py \
      --outdir synthetic_thermal_nonideal \
      --n-scenarios 6 \
      --t-total 2000 \
      --dt 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from scipy.optimize import least_squares


# ------------------ Power profile generation (shark-fin) ------------------ #
def generate_power_profile_shark(
    total_time: float,
    dt: float,
    p_max: float = 8.0,
    n_cycles_range=(2, 4),
    on_duty_range=(0.3, 0.7),
    seed: int = None,
) -> (np.ndarray, np.ndarray):
    """
    Generate a "shark-fin" power profile P(t):

    - The total timeline is divided into n_cycles (2..4 by default).
    - For each cycle:
         idle (P≈0) -> power burst (P nearly constant & high) -> idle.
      This produces temperature ramps with sharp rises and slower decays.

    Returns:
      t: (N,) time axis [s]
      P: (N,) power profile [W]
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.round(total_time / dt)) + 1
    t = np.linspace(0.0, total_time, n_steps)

    P = np.zeros_like(t)

    n_cycles = rng.integers(n_cycles_range[0], n_cycles_range[1] + 1)
    cycle_len = total_time / n_cycles

    for c in range(n_cycles):
        # cycle window [t_c_start, t_c_end]
        t_c_start = c * cycle_len
        t_c_end = (c + 1) * cycle_len

        # on-duty between 30% and 70% of the cycle
        duty = float(rng.uniform(on_duty_range[0], on_duty_range[1]))
        on_len = duty * cycle_len

        # burst start inside the cycle
        margin = 0.05 * cycle_len
        max_start = max(t_c_end - on_len - margin, t_c_start + margin)
        if max_start <= t_c_start + margin:
            t_on_start = t_c_start + margin
        else:
            t_on_start = float(rng.uniform(t_c_start + margin, max_start))

        t_on_end = t_on_start + on_len
        t_on_end = min(t_on_end, t_c_end - margin)

        # indices
        idx_on_start = int(np.clip(np.round(t_on_start / dt), 0, n_steps - 1))
        idx_on_end = int(np.clip(np.round(t_on_end / dt), 0, n_steps - 1))

        if idx_on_end <= idx_on_start:
            continue

        # power level for this burst
        level = float(rng.uniform(0.4 * p_max, p_max))
        P[idx_on_start:idx_on_end] = level

        # optional modulation within the burst
        if idx_on_end - idx_on_start > 10:
            mid = (idx_on_start + idx_on_end) // 2
            P[mid:idx_on_end] *= float(rng.uniform(0.6, 1.0))

    return t, P


# ------------------ low-frequency AR(1) noise / drift ------------------ #
def low_freq_noise(rng, sigma: float, size: int, rho: float = 0.95) -> np.ndarray:
    """
    Generate AR(1)-type low-frequency noise:

      n[k] = rho * n[k-1] + eps[k],
      with eps ~ N(0, sigma^2*(1-rho^2)), so that var(n) ~ sigma^2.

    Used both as:
      - measurement noise (slow IR-like drift/jitter)
      - slow multiplicative drift for physical parameters.
    """
    if sigma <= 0:
        return np.zeros(size, dtype=float)

    eps_sigma = sigma * np.sqrt(1.0 - rho ** 2)
    eps = rng.normal(0.0, eps_sigma, size=size)
    n = np.zeros(size, dtype=float)
    for k in range(1, size):
        n[k] = rho * n[k - 1] + eps[k]
    return n


# ------------------ Non-ideal thermal simulation ------------------ #
def simulate_thermal_scenario_nonideal(
    t: np.ndarray,
    P: np.ndarray,
    C_th_nom: float,
    R_cond_nom: float,
    R_conv_nom: float,
    C_bp: float,
    R_bp_amb: float,
    Tamb: float,
    alpha_bp: float = 0.5,
    Tcase_init: float = None,
    Tbplate_init: float = None,
    rng=None,
    drift_rel: float = 0.15,
    extra_leak_R: float = 50.0,
) -> dict:
    """
    Simulate one thermal scenario (case + backplate) with non-idealities:

      - Time-varying (slowly drifting) C_th, R_cond, R_conv
      - Additional unmodeled thermal leak R_extra to ambient
      - Backplate dynamics driven by alpha*P and cooled via R_bp_amb

    The nominal (constant) values C_th_nom, R_cond_nom, R_conv_nom are the
    ones saved in the JSON as "true" reference values, but the ODE uses
    time-varying effective coefficients internally.
    """
    assert t.shape == P.shape
    n = len(t)
    if n <= 1:
        raise ValueError("Time vector must have length > 1.")

    dt_vec = np.diff(t)

    if Tcase_init is None:
        Tcase_init = Tamb
    if Tbplate_init is None:
        Tbplate_init = Tamb

    # States
    T_case = np.zeros(n, dtype=float)
    T_bplate = np.zeros(n, dtype=float)
    T_case[0] = Tcase_init
    T_bplate[0] = Tbplate_init

    # Slow multiplicative drift for C_th, R_cond, R_conv (relative)
    if rng is None:
        rng = np.random.default_rng()

    if drift_rel > 0:
        drift_Cth = low_freq_noise(rng, sigma=drift_rel, size=n, rho=0.995)
        drift_Rcond = low_freq_noise(rng, sigma=drift_rel, size=n, rho=0.995)
        drift_Rconv = low_freq_noise(rng, sigma=drift_rel, size=n, rho=0.995)
    else:
        drift_Cth = np.zeros(n)
        drift_Rcond = np.zeros(n)
        drift_Rconv = np.zeros(n)

    # Clamp multiplicative factors to reasonable range
    factor_Cth   = np.clip(1.0 + drift_Cth,   0.5, 1.5)
    factor_Rcond = np.clip(1.0 + drift_Rcond, 0.5, 1.5)
    factor_Rconv = np.clip(1.0 + drift_Rconv, 0.5, 1.5)

    # Extra leak resistance (unmodeled in PINN)
    R_extra = float(extra_leak_R) if extra_leak_R > 0 else None

    for k in range(n - 1):
        dt = dt_vec[k]

        # Effective physical parameters at step k
        C_th_eff   = C_th_nom   * factor_Cth[k]
        R_cond_eff = R_cond_nom * factor_Rcond[k]
        R_conv_eff = R_conv_nom * factor_Rconv[k]

        # Safety guards
        C_th_eff   = max(C_th_eff, 1e-6)
        R_cond_eff = max(R_cond_eff, 1e-6)
        R_conv_eff = max(R_conv_eff, 1e-6)

        # Backplate dynamics:
        #  - heated by a fraction alpha_bp of P
        #  - thermally coupled to the case node via R_cond_eff (bidirectional conduction)
        #  - cooled to ambient via R_bp_amb
        dTbp_dt = (
            alpha_bp * P[k]
            + (T_case[k] - T_bplate[k]) / R_cond_eff
            - (T_bplate[k] - Tamb) / R_bp_amb
        ) / C_bp
        T_bplate[k + 1] = T_bplate[k] + dt * dTbp_dt

        # Case dynamics with extra leak
        leak_extra = 0.0

        if R_extra is not None and R_extra > 0:
            leak_extra = (T_case[k] - Tamb) / R_extra

        dTcase_dt = (
            P[k]
            - (T_case[k] - T_bplate[k]) / R_cond_eff
            - (T_case[k] - Tamb) / R_conv_eff
            - leak_extra
        ) / C_th_eff

        T_case[k + 1] = T_case[k] + dt * dTcase_dt

    return {"T_case": T_case, "T_bplate": T_bplate}

# ------------------ Simplified 1-node model + classical fit ------------------ #
def simulate_case_simplified_ode(
    t: np.ndarray,
    P: np.ndarray,
    T_bplate: np.ndarray,
    Tamb: float,
    C_th: float,
    R_cond: float,
    R_conv: float,
    T_init: float,
) -> np.ndarray:
    """
    Forward simulation of the simplified 1-node case model used in training:

      C_th * dT_case/dt =
          P(t)
          - (T_case - T_bplate)/R_cond
          - (T_case - T_amb)/R_conv

    Integrated with explicit Euler using the provided time vector t.

    This is the same structure used in the PINN training script, and is
    employed here for a "classical" identification of (C_th, R_cond)
    directly from T_case_meas.
    """
    assert t.shape == P.shape == T_bplate.shape
    n = len(t)
    if n <= 1:
        raise ValueError("Time vector must have length > 1 for simplified ODE.")

    dt_vec = np.diff(t)
    T = np.zeros_like(P, dtype=float)
    T[0] = float(T_init)

    # Safety guards
    C_th = float(max(C_th, 1e-9))
    R_cond = float(max(R_cond, 1e-9))
    R_conv = float(max(R_conv, 1e-9))

    for k in range(n - 1):
        dt = dt_vec[k]
        Tk = T[k]
        dTdt = (
            P[k]
            - (Tk - T_bplate[k]) / R_cond
            - (Tk - Tamb) / R_conv
        ) / C_th
        T[k + 1] = Tk + dt * dTdt

    return T


def fit_classic_global_params(
    fit_data,
    C_th_init: float,
    R_cond_init: float,
) -> tuple[float, float]:
    """
    Perform a global "classical" fit of (C_th, R_cond) over all scenarios,
    using the simplified 1-node model and T_case_meas as the target.

    We minimize the sum of squared residuals:

      r = T_case_pred - T_case_meas

    where T_case_pred is obtained by integrating the simplified ODE forward
    for each scenario. A non-linear least-squares (Levenberg-Marquardt / TRF)
    solver from SciPy is used, which is accurate and can be relatively
    expensive (O(N * iterations)) but acceptable for typical dataset sizes.
    """

    x0 = np.array([float(C_th_init), float(R_cond_init)], dtype=float)

    def residuals(x):
        C_th, R_cond = x
        # enforce positivity inside the residual function to avoid NaNs
        C_th = max(C_th, 1e-9)
        R_cond = max(R_cond, 1e-9)
        res_list = []

        for item in fit_data:
            t = item["t"]
            P = item["P"]
            Tb = item["T_bplate"]
            Tamb = item["Tamb"]
            T_meas = item["T_case_meas"]
            R_conv = item["R_conv_nominal"]

            T_pred = simulate_case_simplified_ode(
                t=t,
                P=P,
                T_bplate=Tb,
                Tamb=Tamb,
                C_th=C_th,
                R_cond=R_cond,
                R_conv=R_conv,
                T_init=T_meas[0],  # match initial condition to first measured sample
            )
            res_list.append(T_pred - T_meas)

        if not res_list:
            return np.zeros(0, dtype=float)

        return np.concatenate(res_list, axis=0)

    # Bounds enforce positive parameters. "trf" handles bounds robustly.
    result = least_squares(
        residuals,
        x0,
        bounds=([1e-6, 1e-6], [np.inf, np.inf]),
        method="trf",
    )

    C_th_hat, R_cond_hat = result.x
    return float(C_th_hat), float(R_cond_hat)

# ------------------ Argument parsing ------------------ #
def parse_args():
    ap = argparse.ArgumentParser(
        description="Generator of non-ideal synthetic thermal dataset (PINN-compatible)"
    )
    ap.add_argument("--outdir", required=True, help="Output directory for CSV and JSON")
    ap.add_argument("--n-scenarios", type=int, default=6, help="Number of scenarios to generate")
    ap.add_argument("--t-total", type=float, default=2000.0, help="Simulation duration per scenario [s]")
    ap.add_argument("--dt", type=float, default=0.1, help="Integration time step [s]")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed")

    # Nominal thermal parameters for the case node
    ap.add_argument("--Cth-true", type=float, default=5.0, help="Nominal C_th [J/K]")
    ap.add_argument("--Rcond-true", type=float, default=1.0, help="Nominal R_cond [K/W]")
    ap.add_argument("--Rconv-base", type=float, default=50.0, help="Base R_conv [K/W] (airflow will scale it)")

    # Backplate parameters
    ap.add_argument("--Cbp-true", type=float, default=50.0, help="C_bp [J/K]")
    ap.add_argument("--Rbp-amb", type=float, default=10.0, help="R_bp_amb [K/W]")
    ap.add_argument("--alpha-bp", type=float, default=0.2, help="Fraction of P heating the backplate")

    # Measurement noise
    ap.add_argument("--sigma-case", type=float, default=0.25, help="Std dev of noise on T_case_meas [°C]")
    ap.add_argument("--sigma-bp", type=float, default=0.3, help="Std dev of noise on T_bplate [°C]")

    # Non-idealities
    ap.add_argument(
        "--drift-rel",
        type=float,
        default=0.02,
        help="Relative std of slow drift for C_th, R_cond, R_conv (0 disables drift)",
    )
    ap.add_argument(
        "--extra-leak-R",
        type=float,
        default=100.0,
        help="Additional unmodeled thermal resistance to ambient [K/W] (0 disables extra leak)",
    )
    ap.add_argument(
        "--sensor-tau",
        type=float,
        default=2.0,
        help="Time constant [s] for sensor lag on T_case_meas (0 disables lag)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng_global = np.random.default_rng(args.seed)

    # Global nominal parameters
    C_th_true = float(args.Cth_true)
    R_cond_true = float(args.Rcond_true)
    R_conv_base = float(args.Rconv_base)
    C_bp_true = float(args.Cbp_true)
    R_bp_amb_true = float(args.Rbp_amb)
    alpha_bp_true = float(args.alpha_bp)

    print("=== Nominal 'true' global parameters ===")
    print(f"C_th_true   = {C_th_true:.3f} J/K")
    print(f"R_cond_true = {R_cond_true:.3f} K/W")
    print(f"R_conv_base = {R_conv_base:.3f} K/W")
    print(f"C_bp_true   = {C_bp_true:.3f} J/K")
    print(f"R_bp_amb    = {R_bp_amb_true:.3f} K/W")
    print(f"alpha_bp    = {alpha_bp_true:.3f}")
    print(f"drift_rel   = {args.drift_rel:.3f}")
    print(f"extra_leak_R= {args.extra_leak_R:.3f} K/W")
    print(f"sensor_tau  = {args.sensor_tau:.3f} s")

    # Ambient temperatures and airflow scaling factors
    possible_Tamb = [25.0, 40.0, 60.0]   # [°C]
    airflow_factors = [1.0, 0.6, 0.3]    # dimensionless scaling

    # Pre-generate unique (Tamb, airflow) combinations.
    # As long as n_scenarios <= len(all_combos), this guarantees that
    # the last scenario (used as test case by the training script)
    # has an unseen (Tamb, airflow) pair compared to the training ones.
    all_combos = list(product(possible_Tamb, airflow_factors))
    rng_global.shuffle(all_combos)

    scenario_params = []

    # Data to be used for the global "classical" fit of (C_th, R_cond)
    # using the simplified 1-node model on T_case_meas.
    classic_fit_data = []

    for scen_id in range(args.n_scenarios):
        # Scenario-specific RNG (used for drifts and noise)
        rng_scen = np.random.default_rng(args.seed + scen_id * 1024 + 1)

        # Ambient temperature and airflow factor
        if scen_id < len(all_combos):
            Tamb, airflow = all_combos[scen_id]
        else:
            # Fallback when there are more scenarios than unique combos.
            # In this case we cannot strictly guarantee "unseen" conditions
            # for the last scenario, but we keep the previous random behavior.
            Tamb = float(rng_global.choice(possible_Tamb))
            airflow = float(rng_global.choice(airflow_factors))

        # Nominal R_conv for this scenario (this is what we store as "true")
        R_conv_nominal = R_conv_base / airflow

        # Shark-fin power profile
        seed_scen_power = args.seed + scen_id * 17
        t, P = generate_power_profile_shark(
            total_time=args.t_total,
            dt=args.dt,
            p_max=8.0,
            n_cycles_range=(2, 4),
            on_duty_range=(0.3, 0.7),
            seed=seed_scen_power,
        )

        # Non-ideal thermal simulation (noise-free "true" signals)
        sim = simulate_thermal_scenario_nonideal(
            t=t,
            P=P,
            C_th_nom=C_th_true,
            R_cond_nom=R_cond_true,
            R_conv_nom=R_conv_nominal,
            C_bp=C_bp_true,
            R_bp_amb=R_bp_amb_true,
            Tamb=Tamb,
            alpha_bp=alpha_bp_true,
            Tcase_init=Tamb,
            Tbplate_init=Tamb,
            rng=rng_scen,
            drift_rel=args.drift_rel,
            extra_leak_R=args.extra_leak_R,
        )

        T_case_true = sim["T_case"]
        T_bplate_true = sim["T_bplate"]

        n = len(t)
        dt_vec = np.diff(t)

        # --- Sensor lag on case temperature (before adding noise) ---
        if args.sensor_tau > 0:
            T_case_lag = np.zeros_like(T_case_true)
            T_case_lag[0] = T_case_true[0]
            tau = float(args.sensor_tau)
            for k in range(n - 1):
                dt = dt_vec[k]
                alpha = dt / max(tau, 1e-6)
                alpha = min(alpha, 1.0)  # avoid overshoot if dt > tau
                # simple explicit Euler on first-order sensor dynamics
                T_case_lag[k + 1] = T_case_lag[k] + alpha * (T_case_true[k] - T_case_lag[k])
            T_case_base_for_meas = T_case_lag
        else:
            T_case_base_for_meas = T_case_true.copy()

        # --- Low-frequency correlated noise (case & backplate) ---
        noise_case = low_freq_noise(rng_scen, args.sigma_case, size=n, rho=0.95)
        noise_bp = low_freq_noise(rng_scen, args.sigma_bp, size=n, rho=0.98)

        T_case_meas = T_case_base_for_meas + noise_case
        T_bplate_meas = T_bplate_true + noise_bp

        # Build DataFrame
        df = pd.DataFrame(
            {
                "time_s": t,
                "P_W": P,
                "Tamb_C": Tamb * np.ones_like(t),
                "T_bplate_C": T_bplate_meas,
                "T_case_true_C": T_case_true,
                "T_case_meas_C": T_case_meas,
            }
        )

        csv_path = outdir / f"scenario_{scen_id:03d}.csv"
        df.to_csv(csv_path, index=False)
        print(
            f"[Scenario {scen_id}] saved {csv_path} | "
            f"Tamb={Tamb:.1f} °C, R_conv_nominal={R_conv_nominal:.3f} K/W"
        )

        # -------- Per-scenario plot (for quick inspection) --------
        plt.figure(figsize=(10, 4))
        # Case node (true + measured)
        plt.plot(t, T_case_true, label="T_case_true", linewidth=1.5)
        plt.plot(t, T_case_meas, label="T_case_meas", linestyle="--", linewidth=1.0)

        # Backplate sensor (true + measured)
        plt.plot(t, T_bplate_true, label="T_bplate_true", linewidth=1.0)
        plt.plot(
            t, T_bplate_meas, label="T_bplate_meas", linestyle=":", linewidth=0.9
        )

        # Optionally overlay scaled power profile to see drive cycles
        if np.max(P) > 0:
            P_scaled = (P / np.max(P) * 10.0) + Tamb  # shifted around Tamb
            plt.plot(t, P_scaled, label="P (scaled)", alpha=0.5)

        plt.xlabel("Time [s]")
        plt.ylabel("Temperature [°C]")
        plt.title(f"Scenario {scen_id:03d} - Case & backplate temperatures")
        plt.grid(True, alpha=0.3)
        plt.legend()
        png_path = outdir / f"scenario_{scen_id:03d}_temps.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[Scenario {scen_id}] saved plot {png_path}")

        scenario_params.append(
            {
                "scenario_id": scen_id,
                "Tamb_C": Tamb,
                "R_conv_true": R_conv_nominal,   # nominal value used by training script
                "airflow_factor": airflow,
                "csv_file": csv_path.name,
            }
        )

        # Store data for classical global fit (simplified model on T_case_meas)
        classic_fit_data.append(
            {
                "t": t,
                "P": P,
                "T_bplate": T_bplate_meas,
                "Tamb": Tamb,
                "T_case_meas": T_case_meas,
                "R_conv_nominal": R_conv_nominal,
            }
        )

    # --- Classical global identification of (C_th, R_cond) on T_case_meas --- #
    print("\n[Classic fit] Estimating global C_th and R_cond from T_case_meas "
          "using simplified 1-node model...")
    C_th_classic, R_cond_classic = fit_classic_global_params(
        classic_fit_data,
        C_th_init=C_th_true,
        R_cond_init=R_cond_true,
    )
    print(f"[Classic fit] C_th_classic   = {C_th_classic:.6f} J/K")
    print(f"[Classic fit] R_cond_classic = {R_cond_classic:.6f} K/W")

    # JSON with global nominal parameters and scenario info
    meta = {
        "C_th_true": C_th_true,
        "R_cond_true": R_cond_true,
        "C_th_classic": C_th_classic,
        "R_cond_classic": R_cond_classic,
        "R_conv_base": R_conv_base,
        "C_bp_true": C_bp_true,
        "R_bp_amb_true": R_bp_amb_true,
        "alpha_bp_true": alpha_bp_true,
        "sigma_case": float(args.sigma_case),
        "sigma_bplate": float(args.sigma_bp),
        "drift_rel": float(args.drift_rel),
        "extra_leak_R": float(args.extra_leak_R),
        "sensor_tau": float(args.sensor_tau),
        "Tamb_values_used": possible_Tamb,
        "airflow_factors": airflow_factors,
        "scenarios": scenario_params,
    }

    json_path = outdir / "true_params.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved nominal ground-truth parameters to {json_path}")


if __name__ == "__main__":
    main()
