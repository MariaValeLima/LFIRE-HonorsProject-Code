"""
example_osculating2mean.py
Python equivalent of example_osculating2mean.m

Loads the same output.mat state vector timeseries, computes osculating and
mean orbital elements using EU and EU+Kaula theory, and reproduces all 6
plots from the MATLAB script.

Usage (from your project root):
    python example_osculating2mean.py
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# ── Make sure the package is importable ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from osculating2mean import (
    rv_to_oe_osc,
    oe_osc_to_oe_mean_eu,
    oe_osc_to_oe_mean_euk,
)

# ── Configuration ─────────────────────────────────────────────────────────────
MAT_FILE  = ROOT / "data" / "output.mat"
EGM96     = ROOT / "egm96_degree360.ascii"   # in project root
DEGREE    = 10    # spherical harmonic degree (same as MATLAB example)
Ts        = 10.0  # sample period (s)

# ── Load state vector timeseries ─────────────────────────────────────────────
print("Loading data...")
mat = scipy.io.loadmat(str(MAT_FILE))
x = mat["x"]          # shape (6, N)  -- ECI position/velocity in metres
N = x.shape[1]
t_vec = np.arange(N) * Ts   # time vector in seconds
print(f"  {N} timesteps, {N*Ts:.0f} s total ({N*Ts/3600:.2f} h)")

# ── Osculating orbital elements ───────────────────────────────────────────────
print("Computing osculating OE...")
OE_osc = np.zeros((6, N))
for k in range(N):
    OE_osc[:, k] = rv_to_oe_osc(x[:, k])

# ── Mean OE — Eckstein-Ustinov (J2 only, no Fortran) ─────────────────────────
print("Computing EU mean OE...")
OE_mean_EU = np.zeros((6, N))
for k in range(N):
    OE_mean_EU[:, k] = oe_osc_to_oe_mean_eu(OE_osc[:, k])

# ── Mean OE — Eckstein-Ustinov + Kaula ───────────────────────────────────────
print("Computing EU+Kaula mean OE  (this takes a moment)...")
OE_mean_EUK = np.zeros((6, N))
for k in range(N):
    t_tdb = k * Ts
    OE_mean_EUK[:, k] = oe_osc_to_oe_mean_euk(
        t_tdb, OE_osc[:, k], DEGREE, str(EGM96)
    )
print("Done.\n")

# ── Secular drift rates (same formulas as MATLAB) ────────────────────────────
mu    = 3.986004418e14   # m^3/s^2
RE    = 6378.137e3       # m
J2    = 1082.6267e-6

a_mean   = np.mean(OE_osc[0, :])
inc_mean = np.mean(OE_osc[4, :])
n        = np.sqrt(mu / a_mean**3)
gamma    = (J2 / 2) * (RE / a_mean)**2

Omega_dot = -3 * gamma * n * np.cos(inc_mean)
omega_dot = (3/2) * gamma * n * (5 * np.cos(inc_mean)**2 - 1)
M_dot     = (3/2) * gamma * n * (3 * np.cos(inc_mean)**2 - 1)
u_dot     = n + M_dot + omega_dot

# ── Helper: wrap angles to [-pi, pi] ─────────────────────────────────────────
def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ── Plotting ──────────────────────────────────────────────────────────────────
labels   = ["Osculating", "Eckstein-Ustinov", "EU+Kaula"]
colors   = ["tab:blue", "tab:orange", "tab:green"]

# --- Figure 1: semi-major axis (full range) ---
fig, ax = plt.subplots()
ax.plot(t_vec, OE_osc[0, :],      label=labels[0], color=colors[0])
ax.plot(t_vec, OE_mean_EU[0, :],  label=labels[1], color=colors[1])
ax.plot(t_vec, OE_mean_EUK[0, :], label=labels[2], color=colors[2])
ax.set_xlabel("$t$ (s)")
ax.set_ylabel("$a$ (m)")
ax.legend()
ax.set_title("Semi-major axis")
plt.tight_layout()

# --- Figure 2: semi-major axis (zoomed to mean range) ---
fig, ax = plt.subplots()
ax.plot(t_vec, OE_osc[0, :],      label=labels[0], color=colors[0])
ax.plot(t_vec, OE_mean_EU[0, :],  label=labels[1], color=colors[1])
ax.plot(t_vec, OE_mean_EUK[0, :], label=labels[2], color=colors[2])
ax.set_ylim(OE_mean_EU[0, :].min(), OE_mean_EU[0, :].max())
ax.set_xlabel("$t$ (s)")
ax.set_ylabel("$a$ (m)")
ax.legend()
ax.set_title("Semi-major axis (zoomed)")
plt.tight_layout()

# --- Figure 3: u - u_dot*t ---
drift_u = u_dot * np.arange(N) * Ts
aux = np.zeros((3, N))
aux[0, :] = wrap_pi(OE_osc[1, :]    - drift_u)
aux[1, :] = wrap_pi(OE_mean_EU[1, :]  - drift_u)
aux[2, :] = wrap_pi(OE_mean_EUK[1, :] - drift_u)

fig, ax = plt.subplots()
for i in range(3):
    ax.plot(t_vec, aux[i, :], label=labels[i], color=colors[i])
ax.set_xlabel("$t$ (s)")
ax.set_ylabel(r"$u - \dot{u}t$ (rad)")
ax.legend()
ax.set_title("Mean argument of latitude (detrended)")
plt.tight_layout()

# --- Figure 4: ex ---
fig, ax = plt.subplots()
ax.plot(t_vec, OE_osc[2, :],      label=labels[0], color=colors[0])
ax.plot(t_vec, OE_mean_EU[2, :],  label=labels[1], color=colors[1])
ax.plot(t_vec, OE_mean_EUK[2, :], label=labels[2], color=colors[2])
ax.set_xlabel("$t$ (s)")
ax.set_ylabel("$e_x$")
ax.legend()
ax.set_title("Eccentricity vector $e_x$")
plt.tight_layout()

# --- Figure 5: ey ---
fig, ax = plt.subplots()
ax.plot(t_vec, OE_osc[3, :],      label=labels[0], color=colors[0])
ax.plot(t_vec, OE_mean_EU[3, :],  label=labels[1], color=colors[1])
ax.plot(t_vec, OE_mean_EUK[3, :], label=labels[2], color=colors[2])
ax.set_xlabel("$t$ (s)")
ax.set_ylabel("$e_y$")
ax.legend()
ax.set_title("Eccentricity vector $e_y$")
plt.tight_layout()

# --- Figure 6: inclination ---
fig, ax = plt.subplots()
ax.plot(t_vec, OE_osc[4, :],      label=labels[0], color=colors[0])
ax.plot(t_vec, OE_mean_EU[4, :],  label=labels[1], color=colors[1])
ax.plot(t_vec, OE_mean_EUK[4, :], label=labels[2], color=colors[2])
ax.set_xlabel("$t$ (s)")
ax.set_ylabel("$i$ (rad)")
ax.legend()
ax.set_title("Inclination")
plt.tight_layout()

# --- Figure 7: Omega - Omega_dot*t ---
drift_O = Omega_dot * np.arange(N) * Ts
aux2 = np.zeros((3, N))
aux2[0, :] = wrap_pi(OE_osc[5, :]    - drift_O)
aux2[1, :] = wrap_pi(OE_mean_EU[5, :]  - drift_O)
aux2[2, :] = wrap_pi(OE_mean_EUK[5, :] - drift_O)

fig, ax = plt.subplots()
for i in range(3):
    ax.plot(t_vec, aux2[i, :], label=labels[i], color=colors[i])
ax.set_xlabel("$t$ (s)")
ax.set_ylabel(r"$\Omega - \dot{\Omega}t$ (rad)")
ax.legend()
ax.set_title("RAAN (detrended)")
plt.tight_layout()

plt.show()