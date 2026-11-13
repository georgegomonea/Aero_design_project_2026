# %%
"""DEP Aerodynamic Increments vs. Total Shaft Power & Number of Propellers

Implements the Chapter 2 Aero-Propulsive Model (Section 2.1, Eqs. 54-67) 
to evaluate lift and drag augmentation as a function of total DEP power.
Propeller diameter D scales automatically with N (Eq. 70/71).
"""

from dataclasses import dataclass, field, replace
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
# 2. GEOMETRY, FREESTREAM, & SWEEP CONFIGURATION
# =============================================================================
@dataclass
class DEPModelInputs:
    """Geometric and flow parameters for the Chapter 2 DEP model."""
    
    # --- Sweep Study Parameters ---
    N_sweep: list = field(default_factory=lambda: [2, 4, 6, 8, 12, 18, 24])
    power_kw_max: float = 300.0         # Maximum total shaft power [kW]
    power_kw_points: int = 400          # Resolution of the power array
    CL_u_ref: float = 2.0             # Baseline unblown lift coefficient for the sweep [-]

    # --- Geometric & Flow Parameters ---
    N: int = 12                     # Current number of distributed propellers (overwritten in sweep) [-]
    b: float = 9.6                  # Wing span [m]
    be_b: float = 0.873             # Effective blown wingspan fraction b_e / b [-]
    xi: float = 0.2058              # Relative lateral clearance between propellers [-]
    zeta: float = 1.0               # Tip placement parameter (1.0 = inboard, 0.5 = tip) [-]
    x_c: float = 0.31               # Axial propeller distance ahead of LE as fraction of chord (x/c) [-]
    c: float = 0.645                # Wing reference chord [m]
    AR: float = 15.0                # Wing aspect ratio [-]
    M: float = 0.0878               # Freestream Mach number [-]
    Lambda_c2_deg: float = 1.9      # Half-chord sweep angle [deg]
    phi_deg: float = 0.0            # Propeller inclination angle relative to wing vertical frame [deg]
    alpha_twist_deg: float = 0.0    # Wing twist angle relative to wing horizontal frame [deg]
    e: float = 0.8                  # Oswald span efficiency factor [-]
    c_f: float = 0.009              # Sectional skin friction coefficient [-]

    # --- Freestream parameters ---
    V_inf: float = 29.837           # Freestream velocity [m/s]
    rho: float = 1.225              # Freestream air density [kg/m^3]
    eta_P: float = 0.7              # Propeller efficiency [-]

    @property
    def power_kw_sweep(self) -> np.ndarray:
        """Generates the total DEP shaft power array [kW] for the sweep study."""
        return np.linspace(0.0, self.power_kw_max, self.power_kw_points)

    @property
    def b_e(self) -> float:
        """Effective blown wingspan b_e [m] (Eq. 69)."""
        return self.be_b * self.b

    @property
    def D(self) -> float:
        """Propeller diameter D [m] from b_e, N, xi, and zeta (Eq. 70)."""
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
        return self.R / self.c if self.c > 0 else 0.0

    @property
    def x_R(self) -> float:
        """Axial propeller position as a fraction of propeller radius (Eq. 55)."""
        return self.x_c / self.R_c if self.R_c > 0 else 0.0

    def power_to_TC(self, P_total: np.ndarray) -> np.ndarray:
        """Converts an array of total DEP shaft power [W] to single-propeller T_C (Eq. 73)."""
        if self.N <= 0 or self.D <= 0.0:
            return np.zeros_like(P_total)
        T_total = (self.eta_P * P_total) / self.V_inf
        T_single = T_total / self.N
        return T_single / (self.rho * (self.V_inf**2) * (self.D**2))


# =============================================================================
# 3. VECTORIZED DEP DELTA SOLVER (Sweeping T_C arrays instead of alpha)
# =============================================================================
def compute_dep_deltas_vs_power(
    T_C_array: np.ndarray, 
    cfg: DEPModelInputs
) -> dict:
    """Computes delta CL and delta CD arrays as a function of the thrust coefficient array
    for the baseline unblown lift coefficient defined in cfg.CL_u_ref.
    """
    T_C = np.asarray(T_C_array, dtype=float)
    
    if cfg.N <= 0:
        zeros = np.zeros_like(T_C)
        return {"dCL": zeros, "dCD0": zeros, "dCDi": zeros, "dCD": zeros}

    # 1. Axial induction factor at the propeller disk a_P (Eq. 54)
    a_P = 0.5 * (np.sqrt(1.0 + (8.0 / np.pi) * np.maximum(T_C, 0.0)) - 1.0)

    # 2. Slipstream contraction ratio D_s / D (Eq. 56, 64) and LE induction a_w (Eq. 57)
    x_factor = 1.0 + (cfg.x_R / np.sqrt(cfg.x_R**2 + 1.0))
    Ds_D = np.sqrt((1.0 + a_P) / (1.0 + a_P * x_factor))
    b_blown = cfg.D * Ds_D
    a_w = ((a_P + 1.0) / (Ds_D**2)) - 1.0

    # 3. Finite-slipstream correction factor beta (Eq. 58, 59)
    s = 2.0 * a_P + 1.0
    ones = np.ones_like(s)
    X = np.stack([ones, cfg.x_c * ones, (cfg.x_c**2) * ones, cfg.x_c * s, s, s**2], axis=0)
    
    rc_eff = float(np.clip(cfg.R_c, 0.05, 2.0))
    R_powers = np.array([1.0, rc_eff, rc_eff**2, rc_eff**3, rc_eff**4], dtype=float)
    
    # Vectorized polynomial evaluation (R_powers @ K_PATTERSON yields shape (6,))
    beta = np.clip(R_powers @ K_PATTERSON @ X, 0.0, 1.0)

    # 4. Effective 3D wing angle of attack from unblown CL_u (Eq. 61)
    lam_rad = np.radians(cfg.Lambda_c2_deg)
    sweep_term = 1.0 + (np.tan(lam_rad) ** 2) / (1.0 - cfg.M**2)
    alpha_eff = (cfg.CL_u_ref / (2.0 * np.pi * cfg.AR)) * (
        2.0 + np.sqrt((cfg.AR**2) * (1.0 - cfg.M**2) * sweep_term + 4.0)
    )

    # 5. Angle between propeller axis and wing chord i_p (Eq. 62)
    i_p = np.radians(cfg.phi_deg) - alpha_eff - np.radians(cfg.alpha_twist_deg)

    # 6. Sectional 2D lift coefficient increase dC_l (Eq. 60)
    aw_beta = a_w * beta
    term1 = np.sin(alpha_eff) - aw_beta * np.sin(i_p)
    term2 = np.sqrt(aw_beta**2 + 2.0 * aw_beta * np.cos(alpha_eff + i_p) + 1.0)
    dCl_2d = 2.0 * np.pi * (term1 * term2 - np.sin(alpha_eff))

    # 7. 3D lift and drag coefficient increments (Eq. 63, 65, 66, 67)
    span_fraction = cfg.N * (b_blown / cfg.b)
    dCL = dCl_2d * span_fraction
    dCD0 = (a_w**2) * cfg.c_f * span_fraction
    
    # Notice the presence of CL_u_ref in the cross-term here:
    dCDi = (dCL**2 + 2.0 * cfg.CL_u_ref * dCL) / (np.pi * cfg.AR * cfg.e)
    
    dCD = dCD0 + dCDi

    return {
        "dCL": dCL,
        "dCD0": dCD0,
        "dCDi": dCDi,
        "dCD": dCD
    }


# =============================================================================
# 4. PLOTTING ROUTINE 
# =============================================================================
def plot_dep_deltas_vs_power(base_cfg: DEPModelInputs):
    """Plots Delta CL and Delta CD vs Total DEP Power for multiple propeller counts."""
    fig, (ax_cl, ax_cd) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Use a sequential colormap to distinguish N sweeps
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(base_cfg.N_sweep), 1)))
    
    P_total_w = base_cfg.power_kw_sweep * 1000.0  # Convert kW to Watts

    for idx, N_val in enumerate(base_cfg.N_sweep):
        if N_val <= 0:
            continue
            
        # Dynamically instantiate a new config with the updated N 
        # (This correctly recalculates D, R_c, x_R via the class @properties)
        cfg = replace(base_cfg, N=N_val)
        
        # Convert the X-axis Power array into T_C arrays
        T_C_array = cfg.power_to_TC(P_total_w)
        
        # Evaluate the aerodynamic deltas
        res = compute_dep_deltas_vs_power(T_C_array, cfg)
        
        ax_cl.plot(
            base_cfg.power_kw_sweep, 
            res["dCL"], 
            color=cmap[idx], 
            linewidth=2.0, 
            label=f"$N = {N_val}$"
        )
        ax_cd.plot(
            base_cfg.power_kw_sweep, 
            res["dCD"], 
            color=cmap[idx], 
            linewidth=2.0, 
            label=f"$N = {N_val}$"
        )

    # Left subplot: Delta CL vs Power
    ax_cl.set_xlabel("Total DEP Shaft Power, $P_{DP}$ $[kW]$", fontsize=12)
    ax_cl.set_ylabel("Lift Augmentation, $\\Delta C_L$ $[-]$", fontsize=12)
    ax_cl.set_title(f"$\\Delta C_L$ vs. Total Power (Baseline $C_{{L,u}}={base_cfg.CL_u_ref}$)", fontsize=13)
    ax_cl.grid(True, linestyle=":", alpha=0.7)
    ax_cl.legend(loc="upper left", frameon=True, fontsize=10)

    # Right subplot: Delta CD vs Power
    ax_cd.set_xlabel("Total DEP Shaft Power, $P_{DP}$ $[kW]$", fontsize=12)
    ax_cd.set_ylabel("Drag Augmentation, $\\Delta C_D$ $[-]$", fontsize=12)
    ax_cd.set_title(f"$\\Delta C_D$ vs. Total Power (Baseline $C_{{L,u}}={base_cfg.CL_u_ref}$)", fontsize=13)
    ax_cd.grid(True, linestyle=":", alpha=0.7)
    ax_cd.legend(loc="upper left", frameon=True, fontsize=10)

    fig.suptitle(
        f"Aero-Propulsive Increments vs. Total System Power ($V_\\infty={base_cfg.V_inf}$ m/s, $M={base_cfg.M}$, $AR={base_cfg.AR}$)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. EXECUTION 
# =============================================================================
if __name__ == "__main__":
    
    # Initialize Base Geometry & Sweep Configuration 
    config = DEPModelInputs()

    # Plot evaluating the high-lift (Stall / Takeoff) reference baseline
    plot_dep_deltas_vs_power(base_cfg=config)