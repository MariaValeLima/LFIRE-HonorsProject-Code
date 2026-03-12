"""
step2_o2m.py
Run this in the base environment (Python 3.13) after step1_propagate.py:
    conda activate base
    cd "C:/projects/Honors Project/TUDAT shtuff/O2M_project"
    python step2_o2m.py

Loads the saved TUDAT state history, computes osculating and mean orbital
elements for each satellite, and plots the results.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
O2M_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(O2M_DIR))

EGM96   = O2M_DIR / "egm96_degree360.ascii"
NPZ     = O2M_DIR / "tudat_states.npz"

# ── O2M imports ───────────────────────────────────────────────────────────────
from osculating2mean import (
    rv_to_oe_osc,
    oe_osc_to_oe_mean_eu,
    oe_osc_to_oe_mean_euk,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DEGREE  = 10    # spherical harmonic degree for Kaula
RUN_EUK = True  # set False to skip the slow EU+Kaula step (~10-20 min)

OE_LABELS = ["$a$ (m)", "$u$ (rad)", "$e_x$", "$e_y$", "$i$ (rad)", r"$\Omega$ (rad)"]

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {NPZ.name}...")
data      = np.load(str(NPZ), allow_pickle=True)
t         = data["t"]
Ts        = float(data["Ts"])
sat_names = list(data["sat_names"])
N         = len(t)
t_hours   = (t - t[0]) / 3600.0
mothership = sat_names[0]
deputies   = sat_names[1:]
print(f"  {N} timesteps, dt={Ts:.1f} s, duration={t_hours[-1]:.2f} h")
print(f"  Satellites: {sat_names}")

# ── O2M conversion ────────────────────────────────────────────────────────────
OE_osc      = {}
OE_mean_EU  = {}
OE_mean_EUK = {}

for name in sat_names:
    r_arr = data[f"r_{name}"]   # (N, 3)
    v_arr = data[f"v_{name}"]   # (N, 3)
    print(f"\nProcessing {name}...")

    osc = np.zeros((6, N))
    eu  = np.zeros((6, N))
    euk = np.zeros((6, N)) if RUN_EUK else None

    print("  Osculating OE...")
    for k in range(N):
        osc[:, k] = rv_to_oe_osc(np.concatenate([r_arr[k], v_arr[k]]))

    print("  EU mean OE...")
    for k in range(N):
        eu[:, k] = oe_osc_to_oe_mean_eu(osc[:, k])

    if RUN_EUK:
        print("  EU+Kaula mean OE (slow)...")
        for k in range(N):
            euk[:, k] = oe_osc_to_oe_mean_euk(k * Ts, osc[:, k], DEGREE, str(EGM96))

    OE_osc[name]     = osc
    OE_mean_EU[name] = eu
    if RUN_EUK:
        OE_mean_EUK[name] = euk

print("\nDone. Plotting...")

# ── Plotting ──────────────────────────────────────────────────────────────────
colors = plt.cm.tab10(np.linspace(0, 1, len(sat_names)))

# --- Per-satellite: all 6 OEs, osculating vs EU vs EUK ---
for name, color in zip(sat_names, colors):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"{name} — Osculating vs Mean OE", fontsize=13)
    for j, ax in enumerate(axes.flatten()):
        ax.plot(t_hours, OE_osc[name][j, :],     label="Osculating", alpha=0.5)
        ax.plot(t_hours, OE_mean_EU[name][j, :],  label="EU mean",    linewidth=1.5)
        if RUN_EUK:
            ax.plot(t_hours, OE_mean_EUK[name][j, :], label="EUK mean",
                    linewidth=1.5, linestyle="--")
        ax.set_xlabel("$t$ (h)")
        ax.set_ylabel(OE_LABELS[j])
        ax.legend(fontsize=7)
    plt.tight_layout()

# --- Semi-major axis: all satellites on one plot ---
fig, ax = plt.subplots(figsize=(10, 5))
for name, color in zip(sat_names, colors):
    ax.plot(t_hours, OE_mean_EU[name][0, :], label=name, color=color)
ax.set_xlabel("$t$ (h)")
ax.set_ylabel("$a$ (m)")
ax.set_title("Mean semi-major axis — all satellites (EU)")
ax.legend()
plt.tight_layout()

# --- Inclination: all satellites on one plot ---
fig, ax = plt.subplots(figsize=(10, 5))
for name, color in zip(sat_names, colors):
    ax.plot(t_hours, OE_mean_EU[name][4, :], label=name, color=color)
ax.set_xlabel("$t$ (h)")
ax.set_ylabel("$i$ (rad)")
ax.set_title("Mean inclination — all satellites (EU)")
ax.legend()
plt.tight_layout()

# --- Delta semi-major axis: deputies minus mothership ---
fig, ax = plt.subplots(figsize=(10, 5))
for name, color in zip(deputies, colors[1:]):
    da = OE_mean_EU[name][0, :] - OE_mean_EU[mothership][0, :]
    ax.plot(t_hours, da, label=f"{name} − {mothership}", color=color)
ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
ax.set_xlabel("$t$ (h)")
ax.set_ylabel(r"$\Delta a$ (m)")
ax.set_title("Semi-major axis difference: deputies vs mothership (EU mean)")
ax.legend()
plt.tight_layout()

# --- Delta inclination: deputies minus mothership ---
fig, ax = plt.subplots(figsize=(10, 5))
for name, color in zip(deputies, colors[1:]):
    di = OE_mean_EU[name][4, :] - OE_mean_EU[mothership][4, :]
    ax.plot(t_hours, di, label=f"{name} − {mothership}", color=color)
ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
ax.set_xlabel("$t$ (h)")
ax.set_ylabel(r"$\Delta i$ (rad)")
ax.set_title("Inclination difference: deputies vs mothership (EU mean)")
ax.legend()
plt.tight_layout()

plt.show()