# %%
"""Performance-Based Preliminary Sizing of Aircraft with Distributed Propulsion

General-purpose Sizing Matrix Plot (SMP) solver implementing the aerodynamic and
flight-mechanics equations from Chapters 2 and 3 of Andrea Matrone's thesis.
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 1. PATTERSON FINITE-SLIPSTREAM SURROGATE COEFFICIENTS (Eq. 20, 58, 82)
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
# 2. AIRCRAFT, PROPULSION & MISSION INPUT PARAMETERS (Tables 3.2 - 3.9)
# =============================================================================
@dataclass
class AircraftInputs:
  """All input parameters for the DEP Sizing Matrix Plot.

  Change `N` or any aircraft parameter here directly.
  """

  # --- DEP & Wing Geometry ---
  N: int = (
      8   # Number of high-lift DEP propellers (0 = classical unblown)
  )
  chi: float = 1.0  # Thrust fraction of DEP array T_DP / T_total [-]
  AR: float = 8.5  # Wing aspect ratio [-]
  be_b: float = 0.873  # Effective blown wingspan ratio b_e / b [-]
  x_c: float = 0.31  # Propeller axial distance ahead of LE per chord x / c [-]
  Lambda_c2_deg: float = 0  # Wing half-chord sweep angle [deg]
  phi_deg: float = 0.0  # Propeller inclination angle [deg]
  alpha_twist_deg: float = 0.0  # Wing geometric twist angle [deg]
  xi: float = 0.0  # Relative lateral clearance between propellers [-]
  zeta: float = 1.0  # Tip placement factor (1.0 = inboard, 0.5 = wingtip) [-]
  e: float = 0.8  # Oswald span efficiency factor [-]
  c_f: float = 0.009  # Sectional skin friction coefficient [-]

  # --- Propeller Efficiencies by Flight Phase (Tables 3.3 - 3.9) ---
  eta_P_stall: float = 0.7  # Stall propeller efficiency [-]
  eta_P_TO: float = 0.7  # Take-off propeller efficiency [-]
  eta_P_CR: float = 0.8  # Cruise propeller efficiency [-]
  eta_P_climb: float = 0.7  # Climb & ceiling propeller efficiency [-]
  eta_P_L: float = 0.7  # Landing propeller efficiency [-]

  # --- Mission Performance Requirements (Tables 3.3 - 3.9) ---
  V_S: float = 25.0  # Stall speed at sea level [m/s]
  S_TO: float = 90.0  # Take-off distance [m]
  V_CR_max: float = 128.0  # Maximum cruise speed [m/s]
  ROC_max: float = 3.2  # Maximum rate of climb [m/s]
  CGR: float = 0.083  # Climb gradient [-]
  ROC_SC: float = 0.5  # Rate of climb at service ceiling [m/s]
  S_L: float = 90.0  # Total landing distance [m]
  k_W: float = 1.0  # Landing-to-take-off weight ratio W_L / W_TO [-]

  # --- Runway Friction Coefficients (Tables 3.4 & 3.9) ---
  mu_TO: float = 0.04  # Take-off rolling friction coefficient [-]
  mu_L: float = 0.05  # Landing rolling friction coefficient [-]
  mu_B: float = 0.30  # Braking friction coefficient [-]

  # --- Atmospheric Properties (ISA at specified altitudes) ---
  g: float = 9.81  # Gravitational acceleration [m/s^2]
  a_sound: float = 340.0  # Speed of sound [m/s]
  rho_0: float = 1.225  # Sea-level air density (h = 0 m) [kg/m^3]
  rho_TO: float = 1.225  # Take-off air density (h_TO = 0 m) [kg/m^3]
  rho_1: float = 1.056  # Climb air density (h_1 = 1524 m) [kg/m^3]
  rho_CR: float = 0.963  # Cruise air density (h_CR = 2438.4 m) [kg/m^3]
  rho_SC: float = 0.909  # Service ceiling air density (h_SC = 3000 m) [kg/m^3]

  # --- Unblown Baseline Aerodynamic Coefficients ---
  CL_max_u: float = 2.439  # Unblown maximum lift coefficient [-]
  CL_TO_u: float = 1.600  # Unblown take-off lift coefficient [-]
  CD_TO_u: float = 0.100  # Unblown take-off drag coefficient [-]
  CL_R_u: float = 1.800  # Unblown take-off rotation lift coefficient [-]
  CD0_CR_u: float = 0.025  # Unblown zero-lift drag coefficient in cruise [-]
  CD0_1_u: float = 0.035  # Unblown zero-lift drag coefficient in climb [-]
  CL_1_u: float = 0.700  # Unblown lift coefficient in climb [-]
  CL_L_u: float = 2.439  # Unblown landing lift coefficient [-]
  CD_L_u: float = 0.200  # Unblown landing drag coefficient [-]

  # --- Speed Evaluation Mode ---
  # False: uses fixed speeds (V_TO = 1.2*V_S, V_1 = 1.2*V_S, V_L = 1.3*V_S)
  # True:  evaluates V(W/S) from vertical equilibrium (Eq. 114, 130, 136, 156)
  dynamic_dep_speeds: bool = False

  # --- Plot Search Space Limits ---
  WS_min: float = 1.0  # Minimum wing loading [N/m^2]
  WS_max: float = 1500.0  # Maximum wing loading [N/m^2]
  WP_min: float = 0.0  # Minimum power loading [N/W]
  WP_max: float = 0.2  # Maximum power loading [N/W]
  points: int = 600  # Grid resolution along W/S

  @property
  def D_b(self) -> float:
    """Non-dimensional propeller diameter-to-span ratio nu = D / b (Eq. 71)."""
    if self.N <= 0:
      return 0.0
    return self.be_b / ((1.0 + self.xi) * (self.N + 2.0 * (self.zeta - 1.0)))

  @property
  def R_c(self) -> float:
    """Propeller radius-to-chord ratio R / c = 0.5 * AR * (D / b)."""
    if self.N <= 0:
      return 0.0
    return 0.5 * self.AR * self.D_b

  @property
  def x_R(self) -> float:
    """Axial propeller distance ahead of LE per radius x / R = (x / c) / (R / c) (Eq. 55)."""
    if self.N <= 0:
      return 0.0
    return self.x_c / self.R_c

  @property
  def CD_1_u(self) -> float:
    """Unblown climb drag coefficient from parabolic polar (Eq. 139)."""
    return self.CD0_1_u + (self.CL_1_u**2) / (np.pi * self.AR * self.e)


# =============================================================================
# 3. DEP AERO-PROPULSIVE & SIZING EQUATIONS (Chapters 2 & 3)
# =============================================================================
class DEPSizingModel:
  """Evaluates DEP aerodynamic increments (Section 3.2) and solves the seven

  aircraft performance sizing boundaries (Section 3.3).
  """

  def __init__(self, inp: AircraftInputs):
    self.inp = inp
    # Clamp R_c inside the surrogate polynomial's valid interpolation domain [0.05, 2.0]
    rc_eff = np.clip(inp.R_c, 0.05, 2.0)
    r_powers = np.array([1.0, rc_eff, rc_eff**2, rc_eff**3, rc_eff**4])
    self._beta_weights = r_powers @ K_PATTERSON

  def compute_dep_deltas(self, WS, WP, V, rho, CL_u, eta_P):
    p = self.inp
    WS = np.asarray(WS, dtype=float)
    WP = np.asarray(WP, dtype=float)

    if p.N <= 0:
      zeros = np.zeros(np.broadcast_shapes(WS.shape, WP.shape), dtype=float)
      return zeros, zeros, zeros

    # Correct denominator from D = (1 / (1 + xi)) * (b_e / (N + 2*(zeta - 1)))
    geom_denom = (1.0 + p.xi) * (p.N + 2.0 * (p.zeta - 1.0))
    T_c = (
        (p.chi * eta_P)
        / (p.N * rho * (V**3))
        * (WS / np.maximum(WP, 1e-15))
        * (1.0 / p.AR)
        * ((geom_denom / p.be_b) ** 2)
    )

    # Eq. 54, 76, 77: Axial induction factor at the propeller disk a_P
    a_P = 0.5 * (np.sqrt(1.0 + (8.0 / np.pi) * np.maximum(T_c, 0.0)) - 1.0)

    # Eq. 56, 57, 64, 79, 80: Slipstream contraction D_s/D and LE induction a_w
    x_factor = 1.0 + (p.x_R / np.sqrt(p.x_R**2 + 1.0))
    contraction_ratio = np.sqrt((1.0 + a_P) / (1.0 + a_P * x_factor))
    a_w = a_P * x_factor

    # Eq. 58-59, 82-83: Finite-slipstream correction factor beta
    s = 2.0 * a_P + 1.0
    w = self._beta_weights
    beta = (
        w[0]
        + w[1] * p.x_c
        + w[2] * (p.x_c**2)
        + w[3] * p.x_c * s
        + w[4] * s
        + w[5] * (s**2)
    )
    beta = np.clip(beta, 0.0, 1.0)

    # Eq. 61, 62, 87, 88: 3D wing angle of attack alpha and slipstream angle i_p
    M = np.minimum(V / p.a_sound, 0.95)
    lam_rad = np.radians(p.Lambda_c2_deg)
    sweep_factor = 1.0 + (np.tan(lam_rad) ** 2) / (1.0 - M**2)
    alpha = (CL_u / (2.0 * np.pi * p.AR)) * (
        2.0 + np.sqrt((p.AR**2) * (1.0 - M**2) * sweep_factor + 4.0)
    )
    i_p = np.radians(p.phi_deg) - alpha - np.radians(p.alpha_twist_deg)

    # Eq. 60, 84: Sectional 2D lift coefficient increment dC_l
    aw_beta = a_w * beta
    t1 = np.sin(alpha) - aw_beta * np.sin(i_p)
    t2 = np.sqrt(aw_beta**2 + 2.0 * aw_beta * np.cos(alpha + i_p) + 1.0)
    dCl_2d = 2.0 * np.pi * (t1 * t2 - np.sin(alpha))

    # Eq. 63-67, 91, 95, 97: 3D lift and drag increments
    span_factor = (p.be_b * p.N) / geom_denom
    dCL = span_factor * dCl_2d * contraction_ratio
    dCD0 = span_factor * (a_w**2) * p.c_f * contraction_ratio
    dCDi = (dCL**2 + 2.0 * CL_u * dCL) / (np.pi * p.AR * p.e)

    return dCL, dCD0, dCDi

  # -------------------------------------------------------------------------
  # Section 3.3: Performance Requirement Feasibility Margins
  # Convention: margin >= 0 means (WS, WP) satisfies the requirement
  # -------------------------------------------------------------------------

  def margin_stall(self, WS, WP):
    """Stall speed requirement (Section 3.3.1, Eq. 101 - 102)."""
    p = self.inp
    dCL, _, _ = self.compute_dep_deltas(
        WS, WP, p.V_S, p.rho_0, p.CL_max_u, p.eta_P_stall
    )
    return 0.5 * p.rho_0 * (p.V_S**2) * (p.CL_max_u + dCL) - WS

  def margin_takeoff(self, WS, WP):
    """Take-off run requirement (Section 3.3.2, Eq. 116 - 118)."""
    p = self.inp
    V_TO_ref = 1.2 * p.V_S
    V_TO = (
        np.sqrt(2.0 * WS / (p.rho_TO * p.CL_TO_u))
        if p.dynamic_dep_speeds
        else V_TO_ref
    )
    V_R = (
        np.sqrt(2.0 * WS / (p.rho_TO * p.CL_R_u))
        if p.dynamic_dep_speeds
        else V_TO_ref
    )

    dCL_TO, dCD0_TO, dCDi_TO = self.compute_dep_deltas(
        WS, WP, V_TO, p.rho_TO, p.CL_TO_u, p.eta_P_TO
    )
    dCL_R, _, _ = self.compute_dep_deltas(
        WS, WP, V_R, p.rho_TO, p.CL_R_u, p.eta_P_TO
    )

    CL_TO_star = p.CL_TO_u + dCL_TO
    CD_TO_star = p.CD_TO_u + dCD0_TO + dCDi_TO
    CL_R_star = p.CL_R_u + dCL_R

    cd_minus_mu_cl = CD_TO_star - p.mu_TO * CL_TO_star
    A = 0.6 * p.rho_TO * p.g * cd_minus_mu_cl * p.S_TO / WS
    B = cd_minus_mu_cl / CL_R_star

    # Overflow-free evaluation of (1 - exp(A)) / (mu_TO - (mu_TO + B) * exp(A))
    pos_A = A >= 0.0
    exp_neg = np.exp(np.where(pos_A, -A, 0.0))
    exp_pos = np.exp(np.where(pos_A, 0.0, A))

    ratio = np.where(
        pos_A,
        (exp_neg - 1.0) / (p.mu_TO * exp_neg - (p.mu_TO + B)),
        (1.0 - exp_pos) / (p.mu_TO - (p.mu_TO + B) * exp_pos),
    )
    wp_limit = ratio * (p.eta_P_TO / V_TO_ref)
    return wp_limit - WP

  def margin_cruise(self, WS, WP):
    """Maximum cruise speed requirement (Section 3.3.3, Eq. 119 - 128)."""
    p = self.inp
    CL_CR_u = (2.0 * WS) / (p.rho_CR * (p.V_CR_max**2))
    _, dCD0_CR, _ = self.compute_dep_deltas(
        WS, WP, p.V_CR_max, p.rho_CR, CL_CR_u, p.eta_P_CR
    )
    CD0_CR_star = p.CD0_CR_u + dCD0_CR

    term_parasitic = (
        0.5 * p.rho_CR * (p.V_CR_max**3 / p.eta_P_CR) * CD0_CR_star
    )
    term_induced = (2.0 * (WS**2)) / (
        p.rho_CR * p.V_CR_max * np.pi * p.AR * p.e * p.eta_P_CR
    )
    wp_limit = (WS * (p.rho_CR / p.rho_0)) / (term_parasitic + term_induced)
    return wp_limit - WP

  def margin_roc(self, WS, WP):
    """Rate of climb requirement (Section 3.3.4, Eq. 131 - 134)."""
    p = self.inp
    CL_roc_u = np.sqrt(3.0 * p.CD0_1_u * np.pi * p.AR * p.e)
    V_1 = (
        np.sqrt(2.0 * WS / (p.rho_1 * CL_roc_u))
        if p.dynamic_dep_speeds
        else 1.2 * p.V_S
    )

    _, dCD0_1, _ = self.compute_dep_deltas(
        WS, WP, V_1, p.rho_1, p.CL_1_u, p.eta_P_climb
    )
    CD0_1_star = p.CD0_1_u + dCD0_1

    k_roc = np.sqrt(2.0 / p.rho_0) * (4.0 / ((3.0 * np.pi) ** 0.75))  # 0.95017
    p_sink = (
        np.sqrt(WS)
        * (CD0_1_star**0.25)
        / (np.sqrt(p.rho_1 / p.rho_0) * ((p.AR * p.e) ** 0.75))
    ) * k_roc
    wp_limit = p.eta_P_climb / (p.ROC_max + p_sink)
    return wp_limit - WP

  def margin_cgr(self, WS, WP):
    """Climb gradient requirement (Section 3.3.5, Eq. 137 - 140)."""
    p = self.inp
    V_1 = (
        np.sqrt(2.0 * WS / (p.rho_1 * p.CL_1_u))
        if p.dynamic_dep_speeds
        else 1.2 * p.V_S
    )

    dCL_1, dCD0_1, dCDi_1 = self.compute_dep_deltas(
        WS, WP, V_1, p.rho_1, p.CL_1_u, p.eta_P_climb
    )
    CL_1_star = p.CL_1_u + dCL_1
    CD_1_star = p.CD_1_u + dCD0_1 + dCDi_1

    k_cgr = np.sqrt(2.0 / p.rho_0)  # 1.27775
    num = p.eta_P_climb * np.sqrt(p.rho_1 / p.rho_0) * np.sqrt(CL_1_star)
    denom = np.sqrt(WS) * (p.CGR + (CD_1_star / CL_1_star)) * k_cgr
    return (num / denom) - WP

  def margin_ceiling(self, WS, WP):
    """Service ceiling requirement (Section 3.3.6, Eq. 143 - 144)."""
    p = self.inp
    CL_ceil_u = np.sqrt(3.0 * p.CD0_1_u * np.pi * p.AR * p.e)
    V_c = (
        np.sqrt(2.0 * WS / (p.rho_SC * CL_ceil_u))
        if p.dynamic_dep_speeds
        else 1.2 * p.V_S
    )

    _, dCD0_c, _ = self.compute_dep_deltas(
        WS, WP, V_c, p.rho_SC, p.CL_1_u, p.eta_P_climb
    )
    CD0_c_star = p.CD0_1_u + dCD0_c

    k_roc = np.sqrt(2.0 / p.rho_0) * (4.0 / ((3.0 * np.pi) ** 0.75))  # 0.95017
    p_sink_c = (
        np.sqrt(WS)
        * (CD0_c_star**0.25)
        / (np.sqrt(p.rho_SC / p.rho_0) * ((p.AR * p.e) ** 0.75))
    ) * k_roc
    wp_limit = (p.eta_P_climb * (p.rho_SC / p.rho_0)) / (p.ROC_SC + p_sink_c)
    return wp_limit - WP

  def margin_landing(self, WS, WP):
    """Landing distance requirement (Section 3.3.7, Eq. 155 - 158)."""
    p = self.inp
    V_L = (
        np.sqrt(2.0 * p.k_W * WS / (p.rho_0 * p.CL_L_u))
        if p.dynamic_dep_speeds
        else 1.3 * p.V_S
    )

    dCL_L, _, _ = self.compute_dep_deltas(
        WS, WP, V_L, p.rho_0, p.CL_L_u, p.eta_P_L
    )
    CL_L_star = p.CL_L_u + dCL_L

    cd_minus_mu_cl = p.CD_L_u - p.mu_L * p.CL_L_u
    log_arg = (p.mu_L + p.mu_B + (cd_minus_mu_cl / CL_L_star)) / (
        p.mu_L + p.mu_B
    )
    S_L_req_actual = (
        (1.938 * p.k_W * WS) / (p.g * p.rho_0 * cd_minus_mu_cl)
    ) * np.log(np.maximum(log_arg, 1e-12))
    return p.S_L - S_L_req_actual

  # -------------------------------------------------------------------------
  # Vectorized Boundary Solver
  # -------------------------------------------------------------------------

  def solve_curve(
      self, WS_array, margin_fn, wp_min=1e-7, wp_max=10.0, n_iter=60
  ):
    """Vectorized bisection solver across the entire WS_array simultaneously."""
    WS = np.asarray(WS_array, dtype=float)
    lo = np.full_like(WS, wp_min)
    hi = np.full_like(WS, wp_max)
    r_lo = margin_fn(WS, lo)
    r_hi = margin_fn(WS, hi)

    bracketed = (r_lo * r_hi) <= 0.0

    b_lo, b_hi, b_r_lo = lo.copy(), hi.copy(), r_lo.copy()
    for _ in range(n_iter):
      mid = 0.5 * (b_lo + b_hi)
      r_mid = margin_fn(WS, mid)
      keep_upper = (b_r_lo * r_mid) > 0.0
      b_lo = np.where(keep_upper, mid, b_lo)
      b_r_lo = np.where(keep_upper, r_mid, b_r_lo)
      b_hi = np.where(keep_upper, b_hi, mid)

    root = 0.5 * (b_lo + b_hi)
    return np.where(bracketed, root, np.where(r_hi >= 0.0, wp_max, 0.0))

  def evaluate_smp(self, WS_array, wp_min=1e-7, wp_max=10.0):
    """Computes all 7 sizing boundaries and the overall feasible design space."""
    boundaries = {
        "Stall Speed": self.solve_curve(
            WS_array, self.margin_stall, wp_min, wp_max
        ),
        "Take-off Run": self.solve_curve(
            WS_array, self.margin_takeoff, wp_min, wp_max
        ),
        "Max Cruise Speed": self.solve_curve(
            WS_array, self.margin_cruise, wp_min, wp_max
        ),
        "Rate of Climb": self.solve_curve(
            WS_array, self.margin_roc, wp_min, wp_max
        ),
        "Climb Gradient": self.solve_curve(
            WS_array, self.margin_cgr, wp_min, wp_max
        ),
        "Service Ceiling": self.solve_curve(
            WS_array, self.margin_ceiling, wp_min, wp_max
        ),
        "Landing Distance": self.solve_curve(
            WS_array, self.margin_landing, wp_min, wp_max
        ),
    }
    boundaries["Feasible"] = np.minimum.reduce(list(boundaries.values()))
    return boundaries


# =============================================================================
# 4. PLOTTING
# =============================================================================
def plot_smp(inp: AircraftInputs):
  """Solves and plots the Sizing Matrix Plot for the given AircraftInputs."""
  WS = np.linspace(inp.WS_min, inp.WS_max, inp.points)
  model = DEPSizingModel(inp)
  b = model.evaluate_smp(WS)

  styles = {
      "Stall Speed": ("#1f77b4", "--", f"Stall Speed ($V_S={inp.V_S}$ m/s)"),
      "Landing Distance": (
          "#17becf",
          "--",
          f"Landing Distance ($S_L={inp.S_L}$ m)",
      ),
      "Take-off Run": ("#2ca02c", "-", f"Take-off Run ($S_{{TO}}={inp.S_TO}$ m)"),
      "Max Cruise Speed": (
          "#d62728",
          "-",
          f"Max Cruise Speed ($V_{{CR,\\max}}={inp.V_CR_max}$ m/s)",
      ),
      "Rate of Climb": (
          "#e377c2",
          "-",
          f"Rate of Climb ($ROC={inp.ROC_max}$ m/s)",
      ),
      "Climb Gradient": (
          "#ff7f0e",
          "-",
          f"Climb Gradient ($CGR={inp.CGR*100:.1f}\\%$)",
      ),
      "Service Ceiling": (
          "#9467bd",
          "-",
          f"Service Ceiling ($ROC_{{SC}}={inp.ROC_SC}$ m/s)",
      ),
  }

  fig, ax = plt.subplots(figsize=(11, 7))

  for name, (color, ls, label) in styles.items():
    ax.plot(WS, b[name], color=color, linestyle=ls, linewidth=2.0, label=label)

  ax.fill_between(
      WS,
      b["Feasible"],
      0.0,
      color="#8ecae6",
      alpha=0.35,
      label="Feasible Design Space",
  )

  ax.set_xlim(inp.WS_min, inp.WS_max)
  ax.set_ylim(inp.WP_min, inp.WP_max)
  ax.set_xlabel("Wing Loading, $W_{TO}/S$ $[N/m^2]$", fontsize=12)
  ax.set_ylabel("Power Loading, $W_{TO}/P_b$ $[N/W]$", fontsize=12)
  ax.set_title(f"DEP Aircraft Sizing Matrix Plot ($N = {inp.N}$)", fontsize=14)
  ax.grid(True, linestyle=":", alpha=0.7)
  ax.legend(loc="upper right", frameon=True, fontsize=10)
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  inputs = AircraftInputs()
  plot_smp(inputs)