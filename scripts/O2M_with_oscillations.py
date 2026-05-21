import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from tudatpy.interface import spice
from tudatpy.astro.time_representation import DateTime
from tudatpy.astro import element_conversion

from Functions.sim_setup import (
    make_bodies,
    make_propagator_settings_twobody,
    run_simulation,
)
from Functions.postprocess import extract_time_arrays, extract_rv
from Functions.formation_frames import lvlh_dcm_from_rv, eci_to_lvlh
from Functions.plotting import plot_relative_lvlh


# ── Environment ───────────────────────────────────────────────────────────────
spice.load_standard_kernels()

simulation_start_epoch = DateTime(2026, 1, 14).to_epoch()
simulation_end_epoch = simulation_start_epoch + 24 * 3600

mothership_name = "LFIRE-0"
deputy_name = "LFIRE-1"
sat_names = [mothership_name, deputy_name]

bodies = make_bodies(sat_names)
mu_earth = bodies.get("Earth").gravitational_parameter


# ── Chief / mothership Keplerian elements ─────────────────────────────────────
base_kepler = dict(
    semi_major_axis=6.99276221e06,
    eccentricity=0.00,
    inclination=1.71065169e00,
    argument_of_periapsis=1.31226971e00,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e00,
)


# ── Chief ECI state ───────────────────────────────────────────────────────────
chief_state_eci = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=mu_earth,
    **base_kepler,
)

r0_eci = np.array(chief_state_eci[:3])
v0_eci = np.array(chief_state_eci[3:])


# ── Deputy initial relative state in LVLH ─────────────────────────────────────
# Desired bounded CW-style motion:
#
#   R(t) = A/2 sin(nt)
#   S(t) = A cos(nt)
#   W(t) = A sin(nt)
#
# Initial condition at t = 0:
#
#   rho      = [0, A, 0]
#   rho_dot  = [A n / 2, 0, A n]
#
# LVLH axes:
#   R = radial
#   S = along-track
#   W = cross-track

A = 15e3  # m
n = np.sqrt(mu_earth / base_kepler["semi_major_axis"] ** 3)

rho_lvlh = np.array([
    0.0,
    A,
    0.0,
])

rho_dot_lvlh = np.array([
    0.5 * A * n,
    0.0,
    A * n,
])


# ── LVLH to ECI conversion WITH velocity correction ───────────────────────────
C_lvlh_to_eci, _ = lvlh_dcm_from_rv(r0_eci, v0_eci)

omega = np.linalg.norm(np.cross(r0_eci, v0_eci)) / np.linalg.norm(r0_eci) ** 2
omega_lvlh = np.array([0.0, 0.0, omega])

r_dep_eci = r0_eci + C_lvlh_to_eci @ rho_lvlh

# Important:
# This is the velocity-corrected version.
#
# The absolute deputy velocity is not just:
#
#   v0_eci + C @ rho_dot_lvlh
#
# It must include the rotating-frame correction:
#
#   omega x rho
#
v_dep_eci = v0_eci + C_lvlh_to_eci @ (
    rho_dot_lvlh + np.cross(omega_lvlh, rho_lvlh)
)


# ── Print initial orbital elements ────────────────────────────────────────────
oe_chief = np.array([
    base_kepler["semi_major_axis"],
    base_kepler["eccentricity"],
    base_kepler["inclination"],
    base_kepler["longitude_of_ascending_node"],
    base_kepler["argument_of_periapsis"],
    base_kepler["true_anomaly"],
])

oe_deputy = element_conversion.cartesian_to_keplerian(
    np.concatenate([r_dep_eci, v_dep_eci]),
    mu_earth,
)


def print_oe(label, oe):
    a, e, i, raan, aop, ta = oe
    T = 2 * np.pi * np.sqrt(a**3 / mu_earth)

    print(f"  {label}")
    print(f"    a    = {a / 1e3:.3f} km  |  T = {T / 60:.3f} min")
    print(f"    e    = {e:.6f}")
    print(f"    i    = {np.degrees(i):.4f} deg")
    print(f"    RAAN = {np.degrees(raan):.4f} deg")
    print(f"    AoP  = {np.degrees(aop):.4f} deg")
    print(f"    TA   = {np.degrees(ta):.4f} deg")


print("\n── Initial Orbital Elements ───────────────────────────────────────────────")
print_oe("Mothership", oe_chief)
print()
print_oe("Deputy, velocity-corrected", oe_deputy)

initial_separation = np.linalg.norm(r_dep_eci - r0_eci) / 1e3
print(f"\nInitial separation: {initial_separation:.3f} km")


# ── Stack initial state vector [chief | deputy] ───────────────────────────────
initial_state = np.concatenate([
    r0_eci,
    v0_eci,
    r_dep_eci,
    v_dep_eci,
])


# ── Propagate ─────────────────────────────────────────────────────────────────
propagator_settings = make_propagator_settings_twobody(
    bodies=bodies,
    sat_names=sat_names,
    initial_state=initial_state,
    t0=simulation_start_epoch,
    tf=simulation_end_epoch,
    dt=10.0,
)

print("\nPropagating...")
states_array = run_simulation(bodies, propagator_settings)

t, t_hours = extract_time_arrays(states_array)
rv = extract_rv(states_array, sat_names)

N = len(t)
print(f"  {N} timesteps, duration = {t_hours[-1]:.2f} h")


# ── Compute relative trajectory in LVLH ───────────────────────────────────────
r_chief, v_chief = rv[mothership_name]
r_dep, v_dep = rv[deputy_name]

lvlh_rel = np.zeros((N, 3))

for k in range(N):
    C_lvlh_to_eci_k, _ = lvlh_dcm_from_rv(r_chief[k], v_chief[k])

    dr_eci = r_dep[k] - r_chief[k]

    # eci_to_lvlh expects the ECI-to-LVLH rotation,
    # so we pass the transpose of C_lvlh_to_eci.
    lvlh_rel[k] = eci_to_lvlh(C_lvlh_to_eci_k.T, dr_eci)


R_km = lvlh_rel[:, 0] / 1e3
S_km = lvlh_rel[:, 1] / 1e3
W_km = lvlh_rel[:, 2] / 1e3
dist_km = np.linalg.norm(lvlh_rel, axis=1) / 1e3


# ── Plot LVLH components ──────────────────────────────────────────────────────
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=125)

axs[0].plot(t_hours, R_km)
axs[0].set_ylabel("R [km]")
axs[0].set_title("LVLH relative motion, velocity-corrected initialization")

axs[1].plot(t_hours, S_km)
axs[1].set_ylabel("S [km]")

axs[2].plot(t_hours, W_km)
axs[2].set_ylabel("W [km]")
axs[2].set_xlabel("Time [hr]")

for ax in axs:
    ax.grid(True)

plt.tight_layout()
plt.show(block=False)


# ── Plot S-W projection ───────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=125)

ax2.plot(S_km, W_km)
ax2.scatter([0], [0], color="red", s=40, label="Mothership")

ax2.set_xlabel("S [km]  along-track")
ax2.set_ylabel("W [km]  cross-track")
ax2.set_title("S-W projection")
ax2.set_aspect("equal")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show(block=False)


# ── Plot 3D LVLH trajectory ───────────────────────────────────────────────────
fig3 = plt.figure(figsize=(7, 6), dpi=125)
ax3 = fig3.add_subplot(111, projection="3d")

ax3.plot(R_km, S_km, W_km, lw=0.8)
ax3.scatter([0], [0], [0], color="red", s=40, label="Mothership")

ax3.set_xlabel("R [km]")
ax3.set_ylabel("S [km]")
ax3.set_zlabel("W [km]")
ax3.set_title("3D LVLH relative trajectory")
ax3.legend()

plt.tight_layout()
plt.show(block=False)


# ── Plot distance from mothership ─────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 4), dpi=125)

ax4.plot(t_hours, dist_km)
ax4.axhline(A / 1e3, color="red", linestyle="--", label=f"Target {A / 1e3:.0f} km")

ax4.set_xlabel("Time [hr]")
ax4.set_ylabel("Distance [km]")
ax4.set_title("Distance from mothership")
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show(block=False)


# ── Optional: all LVLH projection plots using your existing function ───────────
plot_relative_lvlh(
    rv,
    mothership_name,
    deputy_name,
    t_hours,
    plane="all",
)


# ── Final stats ───────────────────────────────────────────────────────────────
print("\nLVLH stats over simulation:")
print(f"  R:    min = {R_km.min():.3f} km, max = {R_km.max():.3f} km")
print(f"  S:    min = {S_km.min():.3f} km, max = {S_km.max():.3f} km")
print(f"  W:    min = {W_km.min():.3f} km, max = {W_km.max():.3f} km")
print(f"  dist: min = {dist_km.min():.3f} km, max = {dist_km.max():.3f} km")

plt.show()