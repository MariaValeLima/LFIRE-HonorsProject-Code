
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT        = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from tudatpy.interface import spice
from tudatpy.astro.time_representation import DateTime
from Functions.sim_setup import make_bodies, make_propagator_settings_twobody, run_simulation
from Functions.postprocess import extract_time_arrays, extract_rv
from Functions.formation_frames import lvlh_dcm_from_rv, eci_to_lvlh

# ── Environment ───────────────────────────────────────────────────────────────
spice.load_standard_kernels()

simulation_start_epoch = DateTime(2026, 1, 14).to_epoch()
simulation_end_epoch   = simulation_start_epoch + 24 * 3600   # 60 hours

mothership_name = "LFIRE-0"
deputy_name     = "LFIRE-1"
sat_names       = [mothership_name, deputy_name]

bodies   = make_bodies(sat_names)
mu_earth = bodies.get("Earth").gravitational_parameter

# ── Chief (mothership) Keplerian elements ────────────────────────────────────
base_kepler = dict(
    semi_major_axis             = 6.99276221e+06,
    eccentricity                = 0.004,             # circular for clean CW motion
    inclination                 = 1.71065169e+00,
    argument_of_periapsis       = 1.31226971e+00,
    longitude_of_ascending_node = 3.82958313e-01,
    true_anomaly                = 3.07018490e+00,
)

# ── Chief ECI state from Keplerian elements ──────────────────────────────────
from tudatpy.astro import element_conversion

state0 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter         = mu_earth,
    **base_kepler,
)
r0_eci = np.array(state0[:3])
v0_eci = np.array(state0[3:])

# ── CW / PCO deputy initial relative state in LVLH ───────────────────────────
# Bounded CW + cross-track:  R(t) = (A/2)sin(nt),  S(t) = A cos(nt),  W(t) = A sin(nt)
# IC at t=0:  rho = [0, A, 0],  rho_dot = [An/2, 0, An]
A = 15e3  # 15 km

n = np.sqrt(mu_earth / base_kepler["semi_major_axis"]**3)

rho_lvlh     = np.array([0.0,          A,   0.0])  # [R, S, W]
rho_dot_lvlh = np.array([0.5 * A * n, 0.0, A * n]) # [Rdot, Sdot, Wdot]

# ── Convert LVLH relative state to ECI absolute state ────────────────────────
C_to_eci, _ = lvlh_dcm_from_rv(r0_eci, v0_eci)

# LVLH angular velocity (scalar, about W-hat)
omega = np.linalg.norm(np.cross(r0_eci, v0_eci)) / np.linalg.norm(r0_eci)**2
omega_lvlh = np.array([0.0, 0.0, omega])

r_dep_eci = r0_eci + C_to_eci @ rho_lvlh
v_dep_eci = v0_eci + C_to_eci @ (
    rho_dot_lvlh + np.cross(omega_lvlh, rho_lvlh)
)

# ── Stack initial state vector [chief | deputy] ───────────────────────────────
initial_state = np.concatenate([r0_eci, v0_eci, r_dep_eci, v_dep_eci])

print(f"Chief  r0 [km]: {r0_eci/1e3}")
print(f"Chief  v0 [m/s]: {v0_eci}")
print(f"Deputy r0 [km]: {r_dep_eci/1e3}")
print(f"Deputy v0 [m/s]: {v_dep_eci}")
print(f"Initial separation [km]: {np.linalg.norm(r_dep_eci - r0_eci)/1e3:.3f}")

# ── Propagate (two-body, point-mass Earth only) ───────────────────────────────
propagator_settings = make_propagator_settings_twobody(
    bodies=bodies,
    sat_names=sat_names,
    initial_state=initial_state,
    t0=simulation_start_epoch,
    tf=simulation_end_epoch,
    dt=10.0,
)

print("Propagating...")
states_array = run_simulation(bodies, propagator_settings)
t, t_hours   = extract_time_arrays(states_array)
rv           = extract_rv(states_array, sat_names)
N            = len(t)
print(f"  {N} timesteps,  duration={t_hours[-1]:.2f} h")

# ── Compute LVLH relative trajectory ─────────────────────────────────────────
r_chief, v_chief = rv[mothership_name]
r_dep,   _       = rv[deputy_name]

lvlh_rel = np.zeros((N, 3))
for k in range(N):
    C_eci, _ = lvlh_dcm_from_rv(r_chief[k], v_chief[k])
    dr_eci   = r_dep[k] - r_chief[k]
    lvlh_rel[k] = eci_to_lvlh(C_eci.T, dr_eci)

R_km = lvlh_rel[:, 0] / 1e3
S_km = lvlh_rel[:, 1] / 1e3
W_km = lvlh_rel[:, 2] / 1e3
dist_km = np.linalg.norm(lvlh_rel, axis=1) / 1e3

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. LVLH components vs time
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=125)
axs[0].plot(t_hours, R_km); axs[0].set_ylabel("R [km]"); axs[0].set_title("LVLH components vs time")
axs[1].plot(t_hours, S_km); axs[1].set_ylabel("S [km]")
axs[2].plot(t_hours, W_km); axs[2].set_ylabel("W [km]"); axs[2].set_xlabel("Time [hr]")
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show(block=False)

# 2. S-W projection
fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=125)
ax2.plot(S_km, W_km)
ax2.set_xlabel("S [km]  (along-track)")
ax2.set_ylabel("W [km]  (cross-track)")
ax2.set_title("S-W projection (circular oscillation)")
ax2.set_aspect("equal")
ax2.grid(True)
plt.tight_layout()
plt.show(block=False)

# 3. 3D LVLH trajectory
fig3 = plt.figure(figsize=(7, 6), dpi=125)
ax3 = fig3.add_subplot(111, projection="3d")
ax3.plot(R_km, S_km, W_km, lw=0.8)
ax3.scatter([0], [0], [0], color="red", s=40, label="Chief")
ax3.set_xlabel("R [km]"); ax3.set_ylabel("S [km]"); ax3.set_zlabel("W [km]")
ax3.set_title("3D LVLH relative trajectory")
ax3.legend()
plt.tight_layout()
plt.show(block=False)

# 4. Distance from mothership vs time
fig4, ax4 = plt.subplots(figsize=(10, 4), dpi=125)
ax4.plot(t_hours, dist_km)
ax4.axhline(A / 1e3, color="red", linestyle="--", label=f"Target {A/1e3:.0f} km")
ax4.set_xlabel("Time [hr]"); ax4.set_ylabel("Distance [km]")
ax4.set_title("Distance from LFIRE-0 vs time")
ax4.legend(); ax4.grid(True)
plt.tight_layout()
plt.show(block=False)

print(f"\nLVLH stats over simulation:")
print(f"  R: min={R_km.min():.3f} km,  max={R_km.max():.3f} km")
print(f"  S: min={S_km.min():.3f} km,  max={S_km.max():.3f} km")
print(f"  W: min={W_km.min():.3f} km,  max={W_km.max():.3f} km")
print(f"  dist: min={dist_km.min():.3f} km,  max={dist_km.max():.3f} km")

plt.show()
