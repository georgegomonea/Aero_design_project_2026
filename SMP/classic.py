#%%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ==========================================
# 1. PARAMETER DEFINITIONS
# ==========================================

@dataclass
class DesignSpaceLimits:
    """Plotting and computation search boundaries"""
    WS_min: float = 50.0            # Minimum Wing Loading [N/m^2]
    WS_max: float = 5000.0          # Maximum Wing Loading [N/m^2]
    WP_min: float = 0.0             # Minimum Power Loading [N/W]
    WP_max: float = 0.5             # Maximum Power Loading [N/W]
    points: int = 200               # Evaluation array resolution

@dataclass
class HighLevelRequirements:
    """NASA X-57 Maxwell TLARs"""
    v_stall: float = 29.83          # Stall speed limit [m/s]
    S_TO: float = 394.0             # Take-off run distance [m]
    v_TO: float = 29.83 * 1.2       # Take-off speed [m/s]
    v_CR_max: float = 92.6          # Maximum cruise speed [m/s]
    ROC: float = 3.2                # Rate of climb MAX [m/s]
    CGR: float = 0.083              # Climb gradient (8.3%)
    ROC_c: float = 0.5              # Rate of climb at service ceiling [m/s]
    S_L: float = 349.0              # Landing distance [m]
    k_W: float = 1.0                # Landing weight fraction
    
    # Atmospheric Densities [kg/m^3] based on given altitudes
    rho_0: float = 1.225            # SL (h = 0 m)
    rho_TO: float = 1.225           # Take-off altitude (h_TO = 0 m)
    rho_CR: float = 0.963           # Cruise altitude (h_CR = 2438.4 m)
    rho_1: float = 1.056            # Climb altitude (h_1 = 1524 m)
    rho_c: float = 0.909            # Service ceiling altitude (h_SC = 3000 m)
    g: float = 9.81                 # Gravity [m/s^2]
    
    # Friction coefficients
    mu_TO: float = 0.04             # Take-off rolling friction
    mu_L: float = 0.05              # Landing rolling friction
    mu_B: float = 0.3               # Braking friction

@dataclass
class AeroParameters:
    """Aerodynamic and Propulsion coefficients (Classical Plane)"""
    AR: float = 15.0                # Aspect Ratio
    e: float = 0.8                  # Oswald efficiency factor
    
    # Phase-specific propeller efficiencies
    eta_P_TO: float = 0.7           # Take-off propeller efficiency
    eta_P_CR: float = 0.8           # Cruise propeller efficiency
    eta_P_climb: float = 0.7        # Climb propeller efficiency
    
    # Unblown (Classical) Baseline Aerodynamic Coefficients
    CL_max: float = 2.439           # Maximum unblown lift coefficient
    CL_L: float = 2.439             # Unblown lift coefficient during landing
    CD_L: float = 0.200             # Unblown drag coefficient during landing
    
    CL_TO: float = 1.6              # Unblown lift coeff during take-off ground roll
    CD_TO: float = 0.10             # Unblown drag coeff during take-off ground roll
    CL_R: float = 1.8               # Unblown lift coeff at take-off rotation
    
    CD0_CR: float = 0.025           # Unblown zero-lift drag coefficient in cruise
    CD0_1: float = 0.035            # Unblown zero-lift drag coefficient in climb
    CL_1: float = 0.7               # Unblown lift coefficient in climb
    CD_1: float = 0.05              # Unblown total drag coefficient in climb         

# ==========================================
# 2. SIZING MODULES (Classical Equations)
# ==========================================

def calc_stall_speed(reqs: HighLevelRequirements, params: AeroParameters):
    q = 0.5 * reqs.rho_0 * reqs.v_stall**2
    return q * params.CL_max

def calc_landing_distance(reqs: HighLevelRequirements, params: AeroParameters):
    term_CD_CL = params.CD_L - reqs.mu_L * params.CL_L
    log_arg = (reqs.mu_L + reqs.mu_B + (term_CD_CL / params.CL_L)) / (reqs.mu_L + reqs.mu_B)
    
    numerator = reqs.S_L * reqs.g * reqs.rho_0 * term_CD_CL
    denominator = 1.938 * reqs.k_W * np.log(log_arg)
    return numerator / denominator

def calc_takeoff_run(WS, reqs: HighLevelRequirements, params: AeroParameters):
    A = 0.6 * reqs.rho_TO * reqs.g * (params.CD_TO - reqs.mu_TO * params.CL_TO) * reqs.S_TO * (1.0 / WS)
    E = np.exp(A)
    B = (params.CD_TO - reqs.mu_TO * params.CL_TO) / params.CL_R
    
    numerator = 1 - E
    denominator = reqs.mu_TO - (reqs.mu_TO + B) * E
    return (numerator / denominator) * (params.eta_P_TO / reqs.v_TO)

def calc_cruise_speed(WS, reqs: HighLevelRequirements, params: AeroParameters):
    q_CR = 0.5 * reqs.rho_CR * reqs.v_CR_max**3
    term1 = (q_CR * params.CD0_CR) / params.eta_P_CR
    term2 = (WS**2 * 2.0) / (reqs.rho_CR * reqs.v_CR_max * np.pi * params.AR * params.e * params.eta_P_CR)
    
    return (WS * (reqs.rho_CR / reqs.rho_0)) / (term1 + term2)

def calc_rate_of_climb(WS, reqs: HighLevelRequirements, params: AeroParameters):
    p_req = ((WS**0.5 * params.CD0_1**0.25) / ((reqs.rho_1 / reqs.rho_0)**0.5 * (params.AR * params.e)**0.75)) * 0.95017
    return params.eta_P_climb / (reqs.ROC + p_req)

def calc_climb_gradient(WS, reqs: HighLevelRequirements, params: AeroParameters):
    num = params.eta_P_climb * (reqs.rho_1 / reqs.rho_0)**0.5 * params.CL_1**0.5
    denom = WS**0.5 * (reqs.CGR + (params.CD_1 / params.CL_1)) * 1.2777
    return num / denom

def calc_ceiling(WS, reqs: HighLevelRequirements, params: AeroParameters):
    p_req_c = ((WS**0.5 * params.CD0_1**0.25) / ((reqs.rho_c / reqs.rho_0)**0.5 * (params.AR * params.e)**0.75)) * 0.95017
    return (params.eta_P_climb * (reqs.rho_c / reqs.rho_0)) / (reqs.ROC_c + p_req_c)

# ==========================================
# 3. MAIN EXECUTION & PLOTTING
# ==========================================
if __name__ == "__main__":
    reqs = HighLevelRequirements()
    params = AeroParameters()
    limits = DesignSpaceLimits()
    
    WS_range = np.linspace(limits.WS_min, limits.WS_max, limits.points)
    
    ws_max_stall = calc_stall_speed(reqs, params)
    ws_max_land = calc_landing_distance(reqs, params)
    
    wp_to = calc_takeoff_run(WS_range, reqs, params)
    wp_cr = calc_cruise_speed(WS_range, reqs, params)
    wp_roc = calc_rate_of_climb(WS_range, reqs, params)
    wp_cgr = calc_climb_gradient(WS_range, reqs, params)
    wp_ceil = calc_ceiling(WS_range, reqs, params)
    
    wp_feasible = np.minimum.reduce([wp_to, wp_cr, wp_roc, wp_cgr, wp_ceil])
    ws_max_overall = min(ws_max_stall, ws_max_land)
    
    plt.figure(figsize=(12, 8))
    
    plt.axvline(x=ws_max_stall, color='b', linestyle='--', label=f'Stall Speed ($V_s$={reqs.v_stall} m/s)')
    plt.axvline(x=ws_max_land, color='c', linestyle='--', label=f'Landing Distance ($S_L$={reqs.S_L} m)')
    
    plt.plot(WS_range, wp_to, color='g', label='Take-off Run')
    plt.plot(WS_range, wp_cr, color='r', label='Max Cruise Speed')
    plt.plot(WS_range, wp_roc, color='m', label='Rate of Climb')
    plt.plot(WS_range, wp_cgr, color='orange', label='Climb Gradient')
    plt.plot(WS_range, wp_ceil, color='purple', label='Service Ceiling')
    
    feasible_WS = WS_range[WS_range <= ws_max_overall]
    feasible_WP = wp_feasible[WS_range <= ws_max_overall]
    plt.fill_between(feasible_WS, feasible_WP, 0, color='grey', alpha=0.3, label='Feasible Design Space')
    
    plt.xlabel('Wing Loading, $W/S$ $[N/m^2]$', fontsize=12)
    plt.ylabel('Power Loading, $W/P_b$ $[N/W]$', fontsize=12)
    plt.title('Classical Aircraft Sizing Matrix Plot (NASA X-57 Unblown)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.xlim(limits.WS_min, limits.WS_max)
    plt.ylim(limits.WP_min, limits.WP_max)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    plt.show()