# %%
"""DEP Lift and Drag Polar Evolution vs. Angle of Attack 

Implements the Chapter 2 Aero-Propulsive Model (Section 2.1, Eqs. 54-67) 
with propeller diameter D governed by Eq. 70/71: 
D = (1 / (1 + xi)) * (b_e / (N + 2 * (zeta - 1)))
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 1. PATTERSON FINITE-SLIPSTREAM SURROGATE MATRIX (Eq. 58)
# =============================================================================
K_PATTERSON = np.array(
    [
        [0.378269, 0.748135, -0.179986, -0.056464, -0.146746, -0.015255],
        [3.071020, -1.769885, 0.436595, 0.148643, -0.9889332, 0.197940],
        [-2.827730, 2.054064, -0.467410, -0.277325, 0.698981, -0.008226],
        [0.997936, -0.916118, 0.199829, 0.157810, -0.143368, -0.057385],
        [-0.127645, 0.135543, -0.028919, -0.026546, 0.010470, 0.012221],
    ],
    dtype=float,
)


# =============================================================================
# 2. GEOMETRY & FREESTREAM CONFIGURATION (Tables 2.1, 2.3 & 3.1)
# =============================================================================
@dataclass
class DEPModelInputs:
  """Geometric and flow parameters for the Chapter 2 DEP model.

  Propeller diameter D is computed automatically from b, be_b, N, xi, and zeta.
  """

  N: int = 8  # Number of distributed propellers [-]
  b: float = 9.6  # Wing span [m]
  be_b: float = 0.873  # Effective blown wingspan fraction b_e / b [-]
  xi: float = 0.2058  # Relative lateral clearance between propellers [-]
  zeta: float = 1.0  # Tip placement parameter (1.0 = inboard, 0.5 = tip) [-]
  x_c: float = 0.31  # Axial propeller distance ahead of LE as fraction of chord (x/c) [-]
  c: float = 0.645  # Wing reference chord [m]
  AR: float = 15.0  # Wing aspect ratio [-]
  M: float = 0.0878  # Freestream Mach number [-]
  Lambda_c2_deg: float = 1.9  # Half-chord sweep angle [deg]
  phi_deg: float = 0.0  # Propeller inclination angle relative to wing vertical frame [deg]
  alpha_twist_deg: float = 0.0  # Wing twist angle relative to wing horizontal frame [deg]
  e: float = 0.8  # Oswald span efficiency factor [-]
  c_f: float = 0.009  # Sectional skin friction coefficient [-]

  # Freestream parameters used when converting dimensional Thrust [N] or Power [W] into T_C
  V_inf: float = 29.837  # Freestream velocity [m/s]
  rho: float = 1.225  # Freestream air density [kg/m^3]
  eta_P: float = 0.7  # Propeller efficiency [-]

  @property
  def b_e(self) -> float:
    """Effective blown wingspan b_e [m] (Eq. 69)."""
    return self.be_b * self.b

  @property
  def D(self) -> float:
    """Propeller diameter D [m] from b_e, N, xi, and zeta."""
    if self.N <= 0:
      return 0.0
    return self.b_e / ((1.0 + self.xi) * (self.N + 2.0 * (self.zeta - 1.0)))

  @property
  def R(self) -> float:
    """Propeller radius [m]."""
    return 0.5 * self.D

  @property
  def R_c(self) -> float:
    """Ratio of propeller disk radius to wing chord (R/c) [-]."""
    return self.R / self.c

  @property
  def x_R(self) -> float:
    """Axial propeller position as a fraction of propeller radius (Eq. 55)."""
    if self.R_c <= 0.0:
      return 0.0
    return self.x_c / self.R_c

  def thrust_to_TC(self, T_total: float) -> float:
    """Converts total DEP array thrust T_DP [N] to single-propeller T_C (Eq. 27, 72)."""
    if self.N <= 0 or self.D <= 0.0:
      return 0.0
    T_single = T_total / self.N
    return T_single / (self.rho * (self.V_inf**2) * (self.D**2))

  def power_to_TC(self, P_total: float) -> float:
    """Converts total DEP shaft power P_DP [W] to single-propeller T_C (Eq. 73)."""
    T_total = (self.eta_P * P_total) / self.V_inf
    return self.thrust_to_TC(T_total)


# =============================================================================
# 3. CHAPTER 2 DEP AERODYNAMIC SOLVER (Equations 54 - 67)
# =============================================================================
def compute_dep_polar(
    alpha_deg: Sequence[float],
    CL_u: Sequence[float],
    CD_u: Sequence[float],
    T_C: float,
    cfg: DEPModelInputs,
) -> Dict[str, Union[np.ndarray, float]]:
  """Computes the blown lift and drag coefficients across an angle-of-attack array
  given unblown reference data (CL_u, CD_u) and single-propeller thrust coefficient T_C.
  """
  alpha_deg = np.asarray(alpha_deg, dtype=float)
  CL_u = np.asarray(CL_u, dtype=float)
  CD_u = np.asarray(CD_u, dtype=float)

  if not (alpha_deg.shape == CL_u.shape == CD_u.shape):
    raise ValueError("alpha_deg, CL_u, and CD_u must have the exact same shape.")

  if cfg.N <= 0 or T_C <= 0.0:
    zeros = np.zeros_like(CL_u)
    return {
        "alpha_deg": alpha_deg,
        "CL_u": CL_u,
        "CD_u": CD_u,
        "dCl_2d": zeros,
        "dCL": zeros,
        "dCD0": zeros,
        "dCDi": zeros,
        "dCD": zeros,
        "CL_blown": CL_u.copy(),
        "CD_blown": CD_u.copy(),
        "T_C": 0.0,
        "a_P": 0.0,
        "a_w": 0.0,
        "beta": 1.0,
        "Ds_D": 1.0,
    }

  # 1. Axial induction factor at the propeller disk a_P (Eq. 54)
  a_P = 0.5 * (np.sqrt(1.0 + (8.0 / np.pi) * T_C) - 1.0)

  # 2. Slipstream contraction ratio D_s / D at the wing leading edge (Eq. 55, 56, 64)
  x_factor = 1.0 + (cfg.x_R / np.sqrt(cfg.x_R**2 + 1.0))
  Ds_D = np.sqrt((1.0 + a_P) / (1.0 + a_P * x_factor))
  b_blown = cfg.D * Ds_D

  # 3. Axial induction factor at the wing leading edge a_w (Eq. 57)
  a_w = ((a_P + 1.0) / (Ds_D**2)) - 1.0

  # 4. Finite-slipstream correction factor beta from Patterson's surrogate model (Eq. 58, 59)
  s = 2.0 * a_P + 1.0
  X = np.array([1.0, cfg.x_c, cfg.x_c**2, cfg.x_c * s, s, s**2], dtype=float)
  rc_eff = float(np.clip(cfg.R_c, 0.05, 2.0))
  R_powers = np.array([1.0, rc_eff, rc_eff**2, rc_eff**3, rc_eff**4], dtype=float)
  beta = float(np.clip(R_powers @ K_PATTERSON @ X, 0.0, 1.0))

  # 5. Effective 3D wing angle of attack from unblown lift coefficient CL_u (Eq. 61)
  lam_rad = np.radians(cfg.Lambda_c2_deg)
  sweep_term = 1.0 + (np.tan(lam_rad) ** 2) / (1.0 - cfg.M**2)
  alpha_eff = (CL_u / (2.0 * np.pi * cfg.AR)) * (
      2.0 + np.sqrt((cfg.AR**2) * (1.0 - cfg.M**2) * sweep_term + 4.0)
  )

  # 6. Angle between propeller axis and wing chord i_p (Eq. 62)
  i_p = np.radians(cfg.phi_deg) - alpha_eff - np.radians(cfg.alpha_twist_deg)

  # 7. Sectional 2D lift coefficient increase dC_l (Eq. 60)
  aw_beta = a_w * beta
  term1 = np.sin(alpha_eff) - aw_beta * np.sin(i_p)
  term2 = np.sqrt(aw_beta**2 + 2.0 * aw_beta * np.cos(alpha_eff + i_p) + 1.0)
  dCl_2d = 2.0 * np.pi * (term1 * term2 - np.sin(alpha_eff))

  # 8. Three-dimensional lift coefficient increase dC_L (Eq. 63)
  span_fraction = cfg.N * (b_blown / cfg.b)
  dCL = dCl_2d * span_fraction

  # 9. Three-dimensional drag coefficient increases dC_D0, dC_Di, and dC_D (Eq. 65, 66, 67)
  dCD0 = np.full_like(CL_u, (a_w**2) * cfg.c_f * span_fraction)
  dCDi = (dCL**2 + 2.0 * CL_u * dCL) / (np.pi * cfg.AR * cfg.e)
  dCD = dCD0 + dCDi

  return {
      "alpha_deg": alpha_deg,
      "CL_u": CL_u,
      "CD_u": CD_u,
      "dCl_2d": dCl_2d,
      "dCL": dCL,
      "dCD0": dCD0,
      "dCDi": dCDi,
      "dCD": dCD,
      "CL_blown": CL_u + dCL,
      "CD_blown": CD_u + dCD,
      "T_C": T_C,
      "a_P": a_P,
      "a_w": a_w,
      "beta": beta,
      "Ds_D": Ds_D,
  }


# =============================================================================
# 4. DATA LOADER FOR EXTERNAL XFOIL / EXPERIMENTAL POLARS
# =============================================================================
def load_unblown_polar(
    filepath: Union[str, Path],
    delimiter: Optional[str] = None,
    skip_header: int = 0,
    cols: tuple = (0, 1, 2),
):
  """Loads unblown (alpha_deg, CL_u, CD_u) columns from an XFOIL or CSV/text polar file."""
  data = np.loadtxt(
      filepath, delimiter=delimiter, skiprows=skip_header, usecols=cols
  )
  return data[:, 0], data[:, 1], data[:, 2]


# =============================================================================
# 5. PLOTTING ROUTINE (Lift & Drag vs. AoA for Single or Multiple Thrust Levels)
# =============================================================================
def plot_dep_aoa_evolution(
    alpha_deg: Sequence[float],
    CL_u: Sequence[float],
    CD_u: Sequence[float],
    cfg: DEPModelInputs,
    TC_values: Optional[Sequence[float]] = None,
    T_total_values: Optional[Sequence[float]] = None,
    title_prefix: str = "DEP Aerodynamic Coefficients vs. Angle of Attack",
):
  """Plots unblown and DEP-blown C_L(alpha) and C_D(alpha) side-by-side."""
  if TC_values is None and T_total_values is None:
    raise ValueError("Provide either `TC_values` or `T_total_values`.")

  if T_total_values is not None:
    cases = [
        (
            cfg.thrust_to_TC(T),
            (
                f"DEP Blown ($T_{{DP}}={T:.0f}$ N,"
                f" $T_C={cfg.thrust_to_TC(T):.3f}$)"
            ),
        )
        for T in T_total_values
    ]
  else:
    cases = [(tc, f"DEP Blown ($T_C={tc:.3f}$)") for tc in TC_values]

  fig, (ax_cl, ax_cd) = plt.subplots(1, 2, figsize=(13, 5.5))

  ax_cl.plot(
      alpha_deg,
      CL_u,
      color="#d62728",
      marker="o",
      linewidth=2.0,
      label="Unblown Wing (Reference)",
  )
  ax_cd.plot(
      alpha_deg,
      CD_u,
      color="#d62728",
      marker="o",
      linewidth=2.0,
      label="Unblown Wing (Reference)",
  )

  cmap = plt.cm.Blues(np.linspace(0.55, 0.95, max(len(cases), 1)))

  for idx, (tc_val, label) in enumerate(cases):
    res = compute_dep_polar(alpha_deg, CL_u, CD_u, T_C=tc_val, cfg=cfg)
    color = "#1f77b4" if len(cases) == 1 else cmap[idx]
    ax_cl.plot(
        alpha_deg,
        res["CL_blown"],
        color=color,
        linestyle="--",
        marker="s",
        linewidth=2.0,
        label=label,
    )
    ax_cd.plot(
        alpha_deg,
        res["CD_blown"],
        color=color,
        linestyle="--",
        marker="s",
        linewidth=2.0,
        label=label,
    )

  ax_cl.set_xlabel("Angle of Attack, $\\alpha$ $[deg]$", fontsize=12)
  ax_cl.set_ylabel("Lift Coefficient, $C_L$ $[-]$", fontsize=12)
  ax_cl.set_title("Lift Coefficient $C_L$ vs. $\\alpha$", fontsize=13)
  ax_cl.grid(True, linestyle=":", alpha=0.7)
  ax_cl.legend(loc="best", frameon=True, fontsize=9.5)

  ax_cd.set_xlabel("Angle of Attack, $\\alpha$ $[deg]$", fontsize=12)
  ax_cd.set_ylabel("Drag Coefficient, $C_D$ $[-]$", fontsize=12)
  ax_cd.set_title("Drag Coefficient $C_D$ vs. $\\alpha$", fontsize=13)
  ax_cd.grid(True, linestyle=":", alpha=0.7)
  ax_cd.legend(loc="best", frameon=True, fontsize=9.5)

  fig.suptitle(
      f"{title_prefix} ($N={cfg.N}$, $D={cfg.D:.3f}$ m, $M={cfg.M}$, $AR={cfg.AR}$)",
      fontsize=14,
      y=1.02,
  )
  plt.tight_layout()
  plt.show()


# =============================================================================
# 6. EXAMPLE EXECUTION
# =============================================================================
if __name__ == "__main__":
  # 1. Initialize geometry & flow configuration
  # (Defaults in DEPModelInputs match the Experiment I baseline; D is automatically computed)
  config = DEPModelInputs()
  print(f"Computed Propeller Diameter D = {config.D:.4f} m (R/c = {config.R_c:.4f}, x/R = {config.x_R:.4f})")

  # 2. Input unblown reference data (e.g., from XFOIL, CFD, or wind tunnel)
  alpha_ref = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0])
  CL_u_ref = np.array(
      [1.68, 1.86, 2.03, 2.19, 2.32, 2.40, 2.43, 2.439, 2.41, 2.32]
  )
  CD_u_ref = np.array(
      [0.110, 0.122, 0.136, 0.151, 0.168, 0.185, 0.193, 0.200, 0.212, 0.230]
  )

  # 3. Plot single operating point (Experiment I: T_C = 0.611)
  plot_dep_aoa_evolution(
      alpha_deg=alpha_ref,
      CL_u=CL_u_ref,
      CD_u=CD_u_ref,
      cfg=config,
      TC_values=[0.611],
      title_prefix="Experiment I Validation (Single Thrust Setting)",
  )

  # 4. Plot evolution across increasing total DEP thrust levels T_DP [N]
  plot_dep_aoa_evolution(
      alpha_deg=alpha_ref,
      CL_u=CL_u_ref,
      CD_u=CD_u_ref,
      cfg=config,
      T_total_values=[1000.0, 2000.0, 2652.9, 3500.0],
      title_prefix="Lift & Drag Augmentation for Increasing DEP Thrust",
  )