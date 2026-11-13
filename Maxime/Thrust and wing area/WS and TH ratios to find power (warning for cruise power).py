import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMÈTRES D'ENTRÉE ET HYPOTHÈSES
# ==========================================
m_kg = 3400.0
W_lbf = m_kg * 2.20462  # ~7496 lbf
g = 32.174

# (slugs/ft^3)
rho_SL = 0.002378              # Niveau de la mer ISA+18°F
rho_5000 = 0.002048            # Croisière à 5000 ft
rho_9000 = 0.001793            # Croisière à 9000 ft

# HYPOTHÈSES STOL EXTRÊMES (Adapté au RFP)
C_Lmax_TO = 4.5    # Portance massive grâce à la DEP soufflant l'aile
C_Lmax_LDG = 4.5   # DEP active pour l'approche
C_Dmin = 0.035      # Traînée parasite (profil + fuselage + nacelle) -> valeur du livre
C_DTO = 0.045       # Traînée au décollage (avec DEP) -> valeur du livre
C_LTO = 0.8         # Portance au roulage (avec DEP) -> valeur du livre
mu = 0.04          # Friction au roulage -> # ! Peut-être changer
mu_brakes = 0.3    # Friction de freinage -> # ! Peut-être changer
T_rev_W = 0.4      # Reverse thrust électrique (freine à 40% du poids) -> # ! Peut-être changer

AR = 8.5
e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64  # Efficacité d'Oswald (Gudmundsson, p. 65)
k = 1.0 / (np.pi * AR * e)

# Distances
S_G = 150.0  # Roulage alloué au décollage (pour un total < 300ft)
gamma_app_deg = 15.0 # Pente d'approche abrupte (15°)
S_air = 50.0 / np.tan(np.radians(gamma_app_deg)) # Distance air depuis 50ft obs : ~186.6 ft
S_LGR = 300.0 - S_air # Roulage atterrissage disponible : ~113.4 ft

# Vitesses et taux
V_cruise_kt = 250.0
V_cruise = V_cruise_kt * 1.68781
ROC_fpm = 1500.0
V_v = ROC_fpm / 60.0
V_climb_kt = 100.0
V_climb = V_climb_kt * 1.68781

# Rendements propulsifs
eta_TO = 0.65
eta_climb = 0.75
eta_cruise = 0.85

# Plage de WS recalibrée pour le STOL
WS = np.linspace(15, 60, 500)

# ==========================================
# 2. ÉQUATIONS DE GUDMUNDSSON
# ==========================================
# Décollage
TW_takeoff = (1.21 / (g * rho_SL * C_Lmax_TO * S_G)) * WS + (0.605 / C_Lmax_TO) * (C_DTO - mu * C_LTO) + mu

# Montée
q_climb = 0.5 * rho_SL * V_climb**2
TW_climb = (V_v / V_climb) + (q_climb / WS) * C_Dmin + (k / q_climb) * WS

# Croisière
q_cruise = 0.5 * rho_5000 * V_cruise**2
TW_cruise = (q_cruise * C_Dmin) / WS + (k / q_cruise) * WS

# Atterrissage (Limite physique verticale W/S)
# On a les deux contributions de freinage
# On considère qu'on touche le sol à 0.15% de V_stall donc V_TD = 0.15 * V_stall et on sait que V_stall = sqrt(2 * WS / (rho_SL * C_Lmax_LDG))
# donc V_TD**2 = 0.15**2 * 2 * WS / (rho_SL * C_Lmax_LDG)
# on défini facteur_VTD = 0.15**2 * 2.0 / (rho_SL * C_Lmax_LDG) donc V_TD**2 = facteur_VTD * WS
# distance de freinage = d = v**2 / (2 * accel) -> physique de base
# S_LGR = d = facteur_VTD * WS / (2 * a_freinage) -> WS = (S_LGR * 2 * a_freinage) / facteur_VTD
a_freinage = g * (mu_brakes + T_rev_W)
facteur_VTD = 1.15**2 * 2.0 / (rho_SL * C_Lmax_LDG)
WS_landing_max = (S_LGR * 2.0 * a_freinage) / facteur_VTD

# ==========================================
# 3. RECHERCHE DU POINT OPTIMAL
# ==========================================
valid_indices = np.where(WS <= WS_landing_max)[0]
WS_opt = WS[valid_indices[-1]]

TW_req_at_opt = max(
    np.interp(WS_opt, WS, TW_takeoff),
    np.interp(WS_opt, WS, TW_climb),
    np.interp(WS_opt, WS, TW_cruise)
)

# ==========================================
# 4. CALCUL DES DIMENSIONS ET PUISSANCES
# ==========================================
S_ft2 = W_lbf / WS_opt
S_m2 = S_ft2 * 0.092903
b_ft = np.sqrt(AR * S_ft2)
b_m = b_ft * 0.3048

V_LOF = 1.1 * np.sqrt((2 * WS_opt) / (rho_SL * C_Lmax_TO))
V_TO_eval = V_LOF / np.sqrt(2)

Thrust_TO = W_lbf * np.interp(WS_opt, WS, TW_takeoff)
Power_TO_hp = (Thrust_TO * V_TO_eval) / (550.0 * eta_TO)

Thrust_climb = W_lbf * np.interp(WS_opt, WS, TW_climb)
Power_climb_hp = (Thrust_climb * V_climb) / (550.0 * eta_climb)

Thrust_cruise = W_lbf * np.interp(WS_opt, WS, TW_cruise)
Power_cruise_hp = (Thrust_cruise * V_cruise) / (550.0 * eta_cruise)

def hp_to_kw(hp): return hp * 0.7457

# ==========================================
# 5. AFFICHAGE DES RÉSULTATS
# ==========================================
print("=== DESIGN POINT OPTIMAL ===")
print(f"Charge alaire (W/S) max: {WS_opt:.2f} lbf/ft^2")
print(f"Rapport Poussée/Poids (T/W) requis: {TW_req_at_opt:.3f}\n")
print("="*40)

print("=== GÉOMÉTRIE DE L'AILE ===")
print(f"Surface alaire (S) : {S_ft2:.1f} ft^2  |  {S_m2:.1f} m^2")
print(f"Envergure (b)      : {b_ft:.1f} ft    |  {b_m:.1f} m\n")
print("="*40)

print("=== PUISSANCES REQUISES A L'ARBRE (SHP / kW) ===")
print(f"Décollage (STOL)   : {Power_TO_hp:.0f} HP  |  {hp_to_kw(Power_TO_hp):.0f} kW")
print(f"Montée (1500 fpm)  : {Power_climb_hp:.0f} HP  |  {hp_to_kw(Power_climb_hp):.0f} kW")
print(f"Croisière (250 kt) : {Power_cruise_hp:.0f} HP  |  {hp_to_kw(Power_cruise_hp):.0f} kW\n")
print("="*40)

# ==========================================
# 6. AFFICHAGE DES HYPOTHÈSES, print de toutes les variables imposées
# ==========================================
print("=== HYPOTHÈSES ET VALEURS SÉLECTIONNÉES ===")
print(f"Masse brute (m)                : {m_kg} kg ({W_lbf:.0f} lbf)")
print(f"Rho air (densité)                : {rho_SL:.6f} slugs/ft³ (niveau de la mer), {rho_5000:.6f} slugs/ft³ (5000 ft), {rho_9000:.6f} slugs/ft³ (9000 ft)")
print(f"Rho air (densité en kg/m³)          : {rho_SL * 515.3788:.2f} kg/m³ (niveau de la mer), {rho_5000 * 515.3788:.2f} kg/m³ (5000 ft), {rho_9000 * 515.3788:.2f} kg/m³ (9000 ft)")
print(f"Allongement (AR)               : {AR}")
print(f"Efficacité d'Oswald (e)        : {e}")
print(f"k (facteur de traînée induite)      : {k:.6f}")
print(f"C_Lmax Décollage (avec DEP)    : {C_Lmax_TO}")
print(f"C_Lmax Atterrissage (avec DEP) : {C_Lmax_LDG}")
print(f"C_Dmin (Traînée parasite base) : {C_Dmin}")
print(f"C_DTO (Traînée au décollage)   : {C_DTO}")
print(f"C_LTO (Portance au roulage)    : {C_LTO}")
print(f"Friction roulage (mu)          : {mu}")
print(f"Friction freinage (mu_brakes)  : {mu_brakes}")
print(f"Inversion de poussée (T_rev/W) : {T_rev_W}")
print(f"Distance de roulage T-O (S_G)  : {S_G} ft")
print(f"Pente d'approche (gamma)       : {gamma_app_deg}°")
print(f"Distance air depuis 50ft obs (landing approach)   : {S_air:.1f} ft")
print(f"Roulage atterrissage dispo     : {S_LGR:.1f} ft")
print(f"Rendements propulsifs (eta)    : TO={eta_TO}, Climb={eta_climb}, Cruise={eta_cruise}")
print(f"Vitesse croisière (V_cruise)   : {V_cruise_kt} KTAS = {V_cruise:.1f} ft/s = {V_cruise * 0.3048:.1f} m/s")
print(f"Vitesse montée (V_climb)       : {V_climb_kt} KTAS = {V_climb:.1f} ft/s = {V_climb * 0.3048:.1f} m/s")
print(f"Taux de montée (ROC)           : {ROC_fpm} fpm = {V_v:.1f} ft/s = {V_v * 0.3048:.1f} m/s")

# ==========================================
# 7. TRACÉ DU DIAGRAMME (FEASIBLE REGION)
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(WS, TW_takeoff, label='Take-off (300 ft obs limit)', color='blue', linewidth=2)
plt.plot(WS, TW_climb, label='Climb (1500 fpm)', color='green', linewidth=2)
plt.plot(WS, TW_cruise, label='Cruise (250 KTAS)', color='red', linewidth=2)
plt.axvline(x=WS_landing_max, color='magenta', linestyle='--', linewidth=2, label='Landing Limit')

# Hachurage de la Feasible Region
TW_max_constraints = np.maximum.reduce([TW_takeoff, TW_climb, TW_cruise])
plt.fill_between(WS, TW_max_constraints, 1.0, where=(WS <= WS_landing_max), 
                 color='gray', alpha=0.2, label='Feasible Region')

plt.plot(WS_opt, TW_req_at_opt, 'ko', markersize=8, label='Optimal Design Point')

plt.xlim(15, 60)
plt.ylim(0, 0.8)
plt.title("Constraint Analysis - Hybrid STOL Commuter")
plt.xlabel("Wing Loading W/S (lbf/ft²)")
plt.ylabel("Thrust-to-Weight Ratio T/W")
plt.grid(True)
plt.legend()
plt.show()