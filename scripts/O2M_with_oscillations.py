
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
    eccentricity                = 0.00,             # circular for clean CW motion
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

# ── Drifting deputy LVLH ICs (R(0)=A, Ṡ(0)=0 → violates Eq. 5.23) ──────────
rho_drift     = np.array([A,   0.0, 0.0])
rho_dot_drift = np.array([0.0, 0.0, A*n])

r_dep_drift_eci = r0_eci + C_to_eci @ rho_drift
v_dep_drift_eci = v0_eci + C_to_eci @ (rho_dot_drift + np.cross(omega_lvlh, rho_drift))

# ── Convert ECI states to Keplerian elements ──────────────────────────────────
oe_chief = np.array([
    base_kepler["semi_major_axis"], base_kepler["eccentricity"],
    base_kepler["inclination"],     base_kepler["longitude_of_ascending_node"],
    base_kepler["argument_of_periapsis"], base_kepler["true_anomaly"],
])
oe_dep_cw    = element_conversion.cartesian_to_keplerian(
    np.concatenate([r_dep_eci,       v_dep_eci      ]), mu_earth)
oe_dep_drift = element_conversion.cartesian_to_keplerian(
    np.concatenate([r_dep_drift_eci, v_dep_drift_eci]), mu_earth)

def _print_oe(label, oe):
    a, e, i, raan, aop, ta = oe
    T = 2*np.pi * np.sqrt(a**3 / mu_earth)
    print(f"  {label}")
    print(f"    a    = {a/1e3:.3f} km  |  T = {T/60:.3f} min")
    print(f"    e    = {e:.6f}")
    print(f"    i    = {np.degrees(i):.4f} deg")
    print(f"    RAAN = {np.degrees(raan):.4f} deg")
    print(f"    AoP  = {np.degrees(aop):.4f} deg")
    print(f"    TA   = {np.degrees(ta):.4f} deg")

print("\n── Initial Orbital Elements (before propagation) ──────────────────────────")
_print_oe("Mothership  (LFIRE-0)", oe_chief)
print()
_print_oe("Deputy CW   (LFIRE-1)  Eq. 5.23 satisfied — bounded", oe_dep_cw)
print()
_print_oe("Deputy drift (no Eq. 5.23)  R(0)=A, Ṡ(0)=0 — drifting", oe_dep_drift)
print(f"\n  Initial separation (CW deputy) [km]: {np.linalg.norm(r_dep_eci - r0_eci)/1e3:.3f}")

# ── Stack initial state vector [chief | deputy] ───────────────────────────────
initial_state = np.concatenate([r0_eci, v0_eci, r_dep_eci, v_dep_eci])

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

# ── CW Analytical Solution (Eq. 5.18) ────────────────────────────────────────
def cw_solution(t_arr, x0, y0, z0, xd0, yd0, zd0, n_orb):
    """CW closed-form (Eq. 5.18).  x↔R, y↔S, z↔W."""
    nt = n_orb * t_arr
    R = ((4 - 3*np.cos(nt)) * x0
         + np.sin(nt) * xd0 / n_orb
         + 2*(1 - np.cos(nt)) * yd0 / n_orb)
    S = (6*(np.sin(nt) - nt) * x0
         + y0
         - 2*(1 - np.cos(nt)) * xd0 / n_orb
         + (4*np.sin(nt) - 3*nt) * yd0 / n_orb)
    W = z0 * np.cos(nt) + zd0 * np.sin(nt) / n_orb
    return R, S, W

# Unpack primary ICs
x0, y0, z0    = rho_lvlh
xd0, yd0, zd0 = rho_dot_lvlh
tau = t - t[0]  # CW measures time from t=0, not absolute epoch

R_cw_m, S_cw_m, W_cw_m = cw_solution(tau, x0, y0, z0, xd0, yd0, zd0, n)
R_cw_km = R_cw_m / 1e3
S_cw_km = S_cw_m / 1e3
W_cw_km = W_cw_m / 1e3

# ── Drift condition check (Eq. 5.23) ─────────────────────────────────────────
drift_coeff     = 6*n*x0 + 3*yd0        # coefficient of the secular -t term in S
S_dot0_required = -2 * n * x0           # Eq. 5.23 no-drift condition
drift_satisfied = np.isclose(drift_coeff, 0.0, atol=1e-6)

print("\n── CW Drift Analysis (Eq. 5.23) ──────────────────────────────────────────")
print(f"  No-drift requires  Ṡ(0) = -2n·R(0) = {S_dot0_required:.6f} m/s")
print(f"  Actual             Ṡ(0)             = {yd0:.6f} m/s")
print(f"  Drift coefficient  6n·R(0)+3Ṡ(0)   = {drift_coeff:.6e} m/s")
print(f"  Bounded orbit: {'YES — no secular drift' if drift_satisfied else 'NO — secular drift present'}")

# ── Drifting IC scenario (analytical only, for Plot 5) ───────────────────────
x0_d, y0_d, z0_d    = rho_drift
xd0_d, yd0_d, zd0_d = rho_dot_drift
drift_coeff_d = 6*n*x0_d + 3*yd0_d

R_drift_m, S_drift_m, _ = cw_solution(tau, x0_d, y0_d, z0_d, xd0_d, yd0_d, zd0_d, n)
R_drift_km = R_drift_m / 1e3
S_drift_km = S_drift_m / 1e3

# Bounded version: apply Eq. 5.23 correction Ṡ(0) = -2n·R(0)
yd0_bounded = -2 * n * x0_d
R_bnd_m, S_bnd_m, _ = cw_solution(tau, x0_d, y0_d, z0_d, xd0_d, yd0_bounded, zd0_d, n)
R_bnd_km = R_bnd_m / 1e3
S_bnd_km = S_bnd_m / 1e3

print(f"\n── Drifting IC scenario ───────────────────────────────────────────────────")
print(f"  ICs: R(0)={x0_d/1e3:.1f} km, Ṡ(0)={yd0_d:.4f} m/s  (Eq. 5.23 violated)")
print(f"  Drift coeff = {drift_coeff_d:.4e} m/s  (non-zero → secular drift)")
print(f"  Bounded fix: Ṡ(0) = {yd0_bounded:.4f} m/s")

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. LVLH components vs time
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=125)
axs[0].plot(t_hours, R_km,    label="Numerical")
axs[0].plot(t_hours, R_cw_km, linestyle="--", label="CW (Eq. 5.18)")
axs[0].set_ylabel("R [km]"); axs[0].set_title("LVLH components vs time")
axs[0].legend(loc="upper right")
axs[1].plot(t_hours, S_km,    label="Numerical")
axs[1].plot(t_hours, S_cw_km, linestyle="--", label="CW (Eq. 5.18)")
axs[1].set_ylabel("S [km]"); axs[1].legend(loc="upper right")
axs[2].plot(t_hours, W_km,    label="Numerical")
axs[2].plot(t_hours, W_cw_km, linestyle="--", label="CW (Eq. 5.18)")
axs[2].set_ylabel("W [km]"); axs[2].set_xlabel("Time [hr]")
axs[2].legend(loc="upper right")
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show(block=False)

# 2. S-W projection
fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=125)
ax2.plot(S_km,    W_km,    label="Numerical")
ax2.plot(S_cw_km, W_cw_km, linestyle="--", label="CW (Eq. 5.18)")
ax2.set_xlabel("S [km]  (along-track)")
ax2.set_ylabel("W [km]  (cross-track)")
ax2.set_title("S-W projection (circular oscillation)")
ax2.set_aspect("equal")
ax2.legend()
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

# 5. Drifting vs bounded CW comparison (Eq. 5.23 illustration, analytical only)
fig5, axs5 = plt.subplots(1, 2, figsize=(14, 5), dpi=125)

axs5[0].plot(t_hours, S_bnd_km,   label=f"Bounded  [Ṡ(0)={yd0_bounded:.2f} m/s]")
axs5[0].plot(t_hours, S_drift_km, linestyle="--", label="Drifting [Ṡ(0)=0]")
axs5[0].set_xlabel("Time [hr]"); axs5[0].set_ylabel("S [km]")
axs5[0].set_title("Along-track S(t): bounded vs drifting")
axs5[0].legend(); axs5[0].grid(True)

axs5[1].plot(R_bnd_km,   S_bnd_km,   label="Bounded")
axs5[1].plot(R_drift_km, S_drift_km, linestyle="--", label="Drifting")
axs5[1].scatter([x0_d/1e3], [y0_d/1e3], color="red", zorder=5, label="IC")
axs5[1].set_xlabel("R [km]"); axs5[1].set_ylabel("S [km]")
axs5[1].set_title("R-S plane: drift from violated Eq. 5.23")
axs5[1].legend(); axs5[1].grid(True)

plt.suptitle(
    f"Drift demo  [R(0)={x0_d/1e3:.0f} km,  drift coeff={drift_coeff_d:.3e} m/s]",
    fontsize=11
)
plt.tight_layout()
plt.show(block=False)

# 6. 3D LVLH trajectory — CW analytical solution with mothership at origin
fig6 = plt.figure(figsize=(7, 6), dpi=125)
ax6 = fig6.add_subplot(111, projection="3d")
ax6.plot(R_cw_km, S_cw_km, W_cw_km, lw=0.8, label="CW (Eq. 5.18)")
ax6.scatter([0], [0], [0], color="red", s=60, zorder=5, label="Mothership")
ax6.set_xlabel("R [km]"); ax6.set_ylabel("S [km]"); ax6.set_zlabel("W [km]")
ax6.set_title("3D LVLH — CW analytical trajectory")
ax6.legend()
plt.tight_layout()
plt.show(block=False)

print(f"\nLVLH stats over simulation:")
print(f"  R: min={R_km.min():.3f} km,  max={R_km.max():.3f} km")
print(f"  S: min={S_km.min():.3f} km,  max={S_km.max():.3f} km")
print(f"  W: min={W_km.min():.3f} km,  max={W_km.max():.3f} km")
print(f"  dist: min={dist_km.min():.3f} km,  max={dist_km.max():.3f} km")

plt.show()
