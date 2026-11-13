import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.widgets import Slider, TextBox

# ==========================================
# 1. MENUS DE DÉMARRAGE
# ==========================================
print("--- SÉLECTION DU SCÉNARIO ---")
print("1 : Atterrissage (dynamique - distance de freinage)")
print("2 : Décollage (dynamique - vario fpm >= 1500)")
print("3 : Cas standard 3° (statique - démonstration impossibilité)")
choix_scenario = input("Choisissez le scénario (1/2/3) [défaut: 1] : ").strip()
if choix_scenario not in ['1', '2', '3']:
    choix_scenario = '1'

unit_choice = input("Choisissez l'unité de mesure (ft/m) [défaut: ft] : ").strip().lower()
if unit_choice not in ['ft', 'm']:
    unit_choice = 'ft'

# Constantes d'échelle
FT_TO_M = 0.3048
unit = 'm' if unit_choice == 'm' else 'ft'
scale = FT_TO_M if unit == 'm' else 1.0

HAUTEUR = 50.0 * scale        # Obstacle 50 ft
PISTE = 300.0 * scale         # Piste totale 300 ft

# ==========================================
# SCÉNARIO 3 : CAS STANDARD 3° (STATIQUE)
# ==========================================
if choix_scenario == '3':
    base_3deg = HAUTEUR / np.tan(np.radians(3.0))  # ~954 ft
    depassement = base_3deg - PISTE

    fig, ax = plt.subplots(figsize=(11, 6))
    
    # 1. Obstacle (50 ft au seuil)
    ax.plot([0, 0], [0, HAUTEUR], color='darkred', linewidth=3.5)
    ax.scatter([0], [HAUTEUR], color='red', s=50, zorder=5)
    ax.text(-12 * scale, HAUTEUR / 2, f"Obstacle\n{HAUTEUR:.1f} {unit}", 
            ha='right', va='center', fontsize=9.5, color='darkred', weight='bold')

    # 2. Piste 300 ft
    ax.plot([0, PISTE], [0, 0], color='dimgray', linewidth=6, solid_capstyle='butt')
    ax.plot([PISTE, PISTE], [-5 * scale, 5 * scale], color='black', linewidth=2.5)
    ax.text(PISTE, 4 * scale, f"Fin piste 300 ft ({PISTE:.1f} {unit})", ha='right', va='bottom', fontsize=9, weight='bold')

    # 3. Pente 3°
    ax.plot([0, base_3deg], [HAUTEUR, 0], color='darkorange', linestyle='--', linewidth=2.2)
    ax.fill_between([0, base_3deg], [HAUTEUR, 0], color='orange', alpha=0.15)
    ax.scatter([base_3deg], [0], color='darkorange', marker='X', s=90, zorder=6)
    ax.text(base_3deg, 12 * scale, f"Touchdown 3° standard\n{base_3deg:.1f} {unit} ({base_3deg/scale:.0f} ft)", 
            ha='center', va='bottom', fontsize=9.5, weight='bold', color='darkorange',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7e6', edgecolor='darkorange'))

    # 4. Segment de dépassement
    ax.plot([PISTE, base_3deg], [0, 0], color='crimson', linewidth=6, solid_capstyle='butt')
    ax.annotate('', xy=(base_3deg, -7 * scale), xytext=(PISTE, -7 * scale),
                arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.8))
    ax.text((PISTE + base_3deg) / 2, -10 * scale, 
            f"DÉPASSEMENT EN VOL : +{depassement:.1f} {unit} (+{depassement/scale:.0f} ft)\n"
            f"L'avion touche à {base_3deg/PISTE:.1f}× la longueur de la piste !", 
            ha='center', va='top', fontsize=10, weight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee', edgecolor='red'))

    ax.set_xlim(-50 * scale, 1050 * scale)
    ax.set_ylim(-25 * scale, 75 * scale)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title("Approche Standard 3.0° — Atterrissage impossible sur piste de 300 ft", fontsize=12, weight='bold', color='firebrick')
    plt.show()

# ==========================================
# SCÉNARIOS DYNAMIQUES (1 & 2)
# ==========================================
else:
    BASE_MIN = 1.0 * scale
    BASE_MAX = 299.0 * scale
    BASE_INIT = 150.0 * scale
    ANGLE_MIN = np.degrees(np.arctan(HAUTEUR / BASE_MAX))
    ANGLE_MAX = 88.0
    ANGLE_INIT = np.degrees(np.arctan(HAUTEUR / BASE_INIT))
    VITESSE_INIT = 50.0  # kt

    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Marge inférieure généreuse pour empiler sliders et textboxes
    marge_bas = 0.40 if choix_scenario == '2' else 0.32
    plt.subplots_adjust(bottom=marge_bas, left=0.20, right=0.92)

    updating = False

    def dessiner(base, angle_deg, vitesse_kt):
        ax.clear()

        # -----------------------------------------------------------------
        # SCÉNARIO 1 : ATTERRISSAGE
        # -----------------------------------------------------------------
        if choix_scenario == '1':
            freinage = PISTE - base

            # Obstacle au seuil (x=0)
            ax.plot([0, 0], [0, HAUTEUR], color='darkred', linewidth=3.5)
            ax.scatter([0], [HAUTEUR], color='red', s=45, zorder=5)
            ax.text(-8 * scale, HAUTEUR / 2, f"Obstacle\n50 ft", ha='right', va='center', fontsize=9, color='darkred', weight='bold')

            # Trajectoire de descente
            ax.plot([0, base], [HAUTEUR, 0], 'b--', linewidth=2)
            ax.fill_between([0, base], [HAUTEUR, 0], color='skyblue', alpha=0.3)
            ax.scatter([base], [0], color='blue', s=60, zorder=6)
            ax.annotate("Touchdown", xy=(base, 0), xytext=(base + 5 * scale, 8 * scale),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1.2),
                        fontsize=8.5, weight='bold', color='blue')

            # Piste (0 à 300 ft)
            ax.plot([0, PISTE], [0, 0], color='dimgray', linewidth=6, solid_capstyle='butt')
            ax.plot([PISTE, PISTE], [-4 * scale, 4 * scale], color='black', linewidth=2.5)

            # Freinage restant
            if freinage >= 0:
                ax.plot([base, PISTE], [0, 0], color='limegreen', linewidth=6, solid_capstyle='butt')
                ax.annotate('', xy=(PISTE, -5 * scale), xytext=(base, -5 * scale),
                            arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
                ax.text(base + freinage / 2, -7 * scale, 
                        f"FREINAGE DISPONIBLE : {freinage:.1f} {unit} ({freinage/scale:.0f} ft)", 
                        ha='center', va='top', fontsize=9.5, weight='bold', color='darkgreen',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8e8', edgecolor='green'))
            else:
                ax.plot([PISTE, base], [0, 0], color='crimson', linewidth=6, solid_capstyle='butt')
                ax.text(PISTE + 5 * scale, -7 * scale, 
                        f"DÉPASSEMENT DE PISTE : {abs(freinage):.1f} {unit}", 
                        ha='left', va='top', fontsize=9.5, weight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee', edgecolor='red'))

            # Arc d'angle
            if base > 10.0 * scale:
                r_arc = min(base * 0.25, 25.0 * scale)
                arc = Arc((base, 0), width=r_arc * 2, height=r_arc * 2, angle=0, theta1=180 - angle_deg, theta2=180, color='green', linewidth=1.5)
                ax.add_patch(arc)
                rad = np.radians(180 - angle_deg / 2)
                ax.text(base + r_arc * 1.35 * np.cos(rad), max(r_arc * 1.35 * np.sin(rad), 2 * scale), 
                        f"θ = {angle_deg:.1f}°", ha='right', va='bottom', fontsize=9, weight='bold', color='green')

            ax.text(base / 2, -14 * scale, f"Distance Touchdown = {base:.1f} {unit} ({base/scale:.0f} ft)", 
                    ha='center', va='top', fontsize=9, color='darkblue')
            ax.set_title("ATTERRISSAGE — Franchissement 50 ft & Distance de freinage restante", fontsize=11.5, weight='bold', pad=15)

        # -----------------------------------------------------------------
        # SCÉNARIO 2 : DÉCOLLAGE
        # -----------------------------------------------------------------
        else:
            vz_fpm = vitesse_kt * 101.2686 * np.sin(np.radians(angle_deg))
            vz_ms = vz_fpm * 0.00508
            x_rot = PISTE - base  # Point de décollage (Leave Ground)

            # Obstacle de 50 ft en bout de piste (x = 300 ft)
            ax.plot([PISTE, PISTE], [0, HAUTEUR], color='darkred', linewidth=3.5)
            ax.scatter([PISTE], [HAUTEUR], color='red', s=45, zorder=6)
            ax.text(PISTE + 6 * scale, HAUTEUR / 2, f"Obstacle\n50 ft", ha='left', va='center', fontsize=9, color='darkred', weight='bold')

            # Piste totale (0 à 300 ft)
            ax.plot([0, PISTE], [0, 0], color='dimgray', linewidth=6, solid_capstyle='butt')
            ax.plot([0, 0], [-4 * scale, 4 * scale], color='black', linewidth=2.5)

            if x_rot >= 0:
                # Course au sol (jaune)
                ax.plot([0, x_rot], [0, 0], color='gold', linewidth=6, solid_capstyle='butt')
                ax.scatter([x_rot], [0], color='blue', s=60, zorder=6)
                ax.annotate("Leave ground", xy=(x_rot, 0), xytext=(x_rot - 5 * scale, 9 * scale),
                            arrowprops=dict(arrowstyle="->", color="blue", lw=1.2),
                            fontsize=8.5, weight='bold', color='blue', ha='right')

                # Montée (bleu)
                ax.plot([x_rot, PISTE], [0, HAUTEUR], color='navy', linestyle='--', linewidth=2)
                ax.fill_between([x_rot, PISTE], [0, HAUTEUR], color='dodgerblue', alpha=0.25)

                # Arc d'angle de montée
                if base > 10.0 * scale:
                    r_arc = min(base * 0.25, 25.0 * scale)
                    arc = Arc((x_rot, 0), width=r_arc * 2, height=r_arc * 2, angle=0, theta1=0, theta2=angle_deg, color='green', linewidth=1.5)
                    ax.add_patch(arc)
                    rad = np.radians(angle_deg / 2)
                    ax.text(x_rot + r_arc * 1.35 * np.cos(rad), max(r_arc * 1.35 * np.sin(rad), 2 * scale), 
                            f"θ = {angle_deg:.1f}°", ha='left', va='bottom', fontsize=9, weight='bold', color='green')

                # Cotes de roulement et montée
                if x_rot > 2 * scale:
                    ax.text(x_rot / 2, -6 * scale, f"Leave ground à : {x_rot:.1f} {unit} ({x_rot/scale:.0f} ft)", 
                            ha='center', va='top', fontsize=9, weight='bold', color='#8a6d00',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffbe6', edgecolor='gold'))
                
                ax.annotate('', xy=(PISTE, -13 * scale), xytext=(x_rot, -13 * scale),
                            arrowprops=dict(arrowstyle='<->', color='navy', lw=1.5))
                ax.text((x_rot + PISTE) / 2, -15 * scale, f"Distance vol montée : {base:.1f} {unit} ({base/scale:.0f} ft)", 
                        ha='center', va='top', fontsize=8.5, color='navy')
            else:
                ax.plot([0, PISTE], [0, 0], color='crimson', linewidth=6)
                ax.text(PISTE / 2, 20 * scale, "PENTE INSUFFISANTE !\nLa montée requiert plus de 300 ft de piste.", 
                        ha='center', va='center', fontsize=10, weight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fee', edgecolor='red'))

            # Cartouche d'exigence Vz
            conforme = vz_fpm >= 1500.0
            c_vz = 'darkgreen' if conforme else 'firebrick'
            bg_vz = '#eaffea' if conforme else '#ffebeb'
            statut_vz = "CONFORME (≥ 1500 fpm)" if conforme else "NON CONFORME (< 1500 fpm)"

            ax.text(10 * scale, 62 * scale, 
                    f"Vitesse : {vitesse_kt:.0f} kt\n = {vitesse_kt * 0.51444:.1f} m/s\n"
                    f"Vz : {vz_fpm:.0f} ft/min ({vz_ms:.1f} m/s)\n"
                    f"Statut : {statut_vz}", 
                    ha='left', va='top', fontsize=9.5, weight='bold', color=c_vz,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_vz, edgecolor=c_vz, lw=1.5))

            ax.set_title("DÉCOLLAGE — Course sol & Montée avec contrôle Vz ≥ 1500 ft/min", fontsize=11.5, weight='bold', pad=15)

        ax.set_xlim(-35 * scale, 350 * scale)
        ax.set_ylim(-25 * scale, 75 * scale)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.canvas.draw_idle()

    # ==========================================
    # DISPOSITION DES CONTRÔLES (ESPACÉS SANS CHEVAUCHEMENT)
    # ==========================================
    label_dist = "Touchdown" if choix_scenario == '1' else "Dist. Vol"
    label_box = f"Touchdown ({unit}) :" if choix_scenario == '1' else f"Leave ground ({unit}) :"

    if choix_scenario == '1':
        # 1. Sliders (haut)
        ax_s_base = plt.axes([0.22, 0.20, 0.68, 0.03])
        ax_s_angle = plt.axes([0.22, 0.13, 0.68, 0.03])
        slider_vit = None

        # 2. TextBoxes (en bas, bien séparées à gauche et à droite)
        ax_box_base = plt.axes([0.30, 0.04, 0.16, 0.04])
        ax_box_angle = plt.axes([0.72, 0.04, 0.16, 0.04])

    else:
        # Décollage : 3 sliders puis la ligne de textboxes
        ax_s_base = plt.axes([0.22, 0.26, 0.68, 0.025])
        ax_s_angle = plt.axes([0.22, 0.19, 0.68, 0.025])
        ax_s_vit = plt.axes([0.22, 0.12, 0.68, 0.025])
        slider_vit = Slider(ax=ax_s_vit, label="Vitesse (kt) ", valmin=30.0, valmax=120.0, valinit=VITESSE_INIT, valfmt='%1.0f kt', color='sandybrown')

        # TextBoxes alignées tout en bas
        ax_box_base = plt.axes([0.30, 0.03, 0.16, 0.04])
        ax_box_angle = plt.axes([0.72, 0.03, 0.16, 0.04])

    # Création des widgets
    slider_base = Slider(ax=ax_s_base, label=f'{label_dist} ({unit}) ', valmin=BASE_MIN, valmax=BASE_MAX, valinit=BASE_INIT, valfmt=f'%1.1f {unit}', color='skyblue')
    slider_angle = Slider(ax=ax_s_angle, label="Pente (°) ", valmin=ANGLE_MIN, valmax=ANGLE_MAX, valinit=ANGLE_INIT, valfmt='%1.1f°', color='lightgreen')

    val_init_box = BASE_INIT if choix_scenario == '1' else (PISTE - BASE_INIT)
    box_base = TextBox(ax=ax_box_base, label=label_box, initial=f"{val_init_box:.1f}")
    box_angle = TextBox(ax=ax_box_angle, label="Angle (°) :", initial=f"{ANGLE_INIT:.1f}")

    # --- SYNCHRONISATION UNIFIÉE ---
    def synchroniser(val_base, val_ang):
        global updating
        if updating: return
        updating = True

        slider_base.set_val(val_base)
        slider_angle.set_val(val_ang)

        val_box_txt = val_base if choix_scenario == '1' else (PISTE - val_base)
        box_base.set_val(f"{val_box_txt:.1f}")
        box_angle.set_val(f"{val_ang:.1f}")

        v = slider_vit.val if slider_vit else VITESSE_INIT
        dessiner(val_base, val_ang, v)
        updating = False

    def sync_from_slider_base(val):
        if updating: return
        val = max(BASE_MIN, min(val, BASE_MAX))
        ang = np.degrees(np.arctan(HAUTEUR / val))
        synchroniser(val, ang)

    def sync_from_slider_angle(ang):
        if updating: return
        ang = max(ANGLE_MIN, min(ang, ANGLE_MAX))
        val = HAUTEUR / np.tan(np.radians(ang))
        synchroniser(val, ang)

    def submit_box_base(texte):
        try:
            val_entree = float(texte)
            if choix_scenario == '1':
                b = max(BASE_MIN, min(val_entree, BASE_MAX))
            else:
                lg = max(0.0, min(val_entree, PISTE - BASE_MIN))
                b = PISTE - lg
            
            ang = np.degrees(np.arctan(HAUTEUR / b))
            synchroniser(b, ang)
        except ValueError:
            val_reset = slider_base.val if choix_scenario == '1' else (PISTE - slider_base.val)
            box_base.set_val(f"{val_reset:.1f}")

    def submit_box_angle(texte):
        try:
            ang = float(texte)
            ang = max(ANGLE_MIN, min(ang, ANGLE_MAX))
            b = HAUTEUR / np.tan(np.radians(ang))
            synchroniser(b, ang)
        except ValueError:
            box_angle.set_val(f"{slider_angle.val:.1f}")

    # Écouteurs
    slider_base.on_changed(sync_from_slider_base)
    slider_angle.on_changed(sync_from_slider_angle)
    box_base.on_submit(submit_box_base)
    box_angle.on_submit(submit_box_angle)

    if slider_vit:
        slider_vit.on_changed(lambda v: dessiner(slider_base.val, slider_angle.val, v))

    dessiner(BASE_INIT, ANGLE_INIT, VITESSE_INIT)
    plt.show()