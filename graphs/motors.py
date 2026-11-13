import matplotlib.pyplot as plt
import numpy as np

# Motor Data (Name, Weight [kg], Cont. Power [kW], Cooling Type)
# Safran weights are approximated based on their stated continuous power and density metrics.
# Evolito D-series continuous power is estimated at ~75% of stated peak power for plotting consistency.
motors = [
    ("magniAIR", 43, 130, "Air-cooled"),
    ("Siemens SP90G", 13, 65, "Air-cooled"),
    ("Siemens SP260D", 50, 261, "Air-cooled"),
    ("Calnetix Turbo Gen", 7.48, 100, "Air-cooled"),
    ("Calnetix Starter Gen", 12.43, 120, "Air-cooled"),
    ("Safran ENGINeUS 100", 100/3.5, 100, "Air-cooled"), 
    ("Safran GENeUS", 300/8.0, 300, "Air-cooled"), 
    ("magni350", 92.3, 165, "Liquid-cooled"),
    ("magni650", 144, 515, "Liquid-cooled"),
    ("Evolito E800", 28.3, 110, "Liquid-cooled"),
    ("Evolito E1400", 48, 130, "Liquid-cooled"),
    ("Evolito D250", 14.5, 180 * 0.75, "Liquid-cooled"),
    ("Evolito D500", 30.0, 275 * 0.75, "Liquid-cooled")
]

weights_air = [m[1] for m in motors if m[3] == "Air-cooled"]
powers_air = [m[2] for m in motors if m[3] == "Air-cooled"]
labels_air = [m[0] for m in motors if m[3] == "Air-cooled"]

weights_liq = [m[1] for m in motors if m[3] == "Liquid-cooled"]
powers_liq = [m[2] for m in motors if m[3] == "Liquid-cooled"]
labels_liq = [m[0] for m in motors if m[3] == "Liquid-cooled"]

plt.figure(figsize=(10, 7))

# Plot contours of constant power-to-weight ratio (kW/kg)
w_range = np.linspace(0, 160, 400)
for ptw in [2, 4, 6, 8, 10, 12, 14, 16]:
    plt.plot(w_range, w_range * ptw, color='gray', linestyle='--', alpha=0.3)
    if w_range[-1]*ptw < 600:
        plt.text(150, 150*ptw + 5, f'{ptw} kW/kg', color='gray', alpha=0.7, 
                 fontsize=9, rotation=np.degrees(np.arctan(ptw/3.5)))

# Scatter plots
plt.scatter(weights_air, powers_air, color='blue', s=80, marker='o', 
            label='Air-cooled', edgecolors='black', zorder=5)
plt.scatter(weights_liq, powers_liq, color='red', s=80, marker='s', 
            label='Liquid-cooled', edgecolors='black', zorder=5)

# Annotations
for w, p, l in zip(weights_air, powers_air, labels_air):
    xytext = (8, -5)
    if l == "Calnetix Starter Gen": xytext = (8, 5)
    plt.annotate(l, (w, p), textcoords="offset points", xytext=xytext, ha='left', fontsize=9)

for w, p, l in zip(weights_liq, powers_liq, labels_liq):
    xytext = (8, -5)
    ha = 'left'
    if l == "magni650": 
        xytext = (-8, 10)
        ha = 'right'
    elif l == "Evolito D250":
        xytext = (-8, 8)
        ha = 'right'
    elif l == "Evolito D500":
        xytext = (-8, 8)
        ha = 'right'
    plt.annotate(l, (w, p), textcoords="offset points", xytext=xytext, ha=ha, fontsize=9)

plt.title('Electric Motor Selection: Continuous Power vs. Dry Weight')
plt.xlabel('Dry Weight [kg]')
plt.ylabel('Continuous Rated Power [kW]')
plt.xlim(0, 160)
plt.ylim(0, 600)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('motor_selection_graph_updated.pdf')