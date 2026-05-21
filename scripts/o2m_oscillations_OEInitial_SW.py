
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT        = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from tudatpy.interface import spice
from tudatpy.astro.time_representation import DateTime
from tudatpy.astro import element_conversion
from Functions.sim_setup import make_bodies, make_propagator_settings_twobody, run_simulation
from Functions.postprocess import extract_time_arrays, extract_rv, rv_to_kepler
from Functions.formation_frames import lvlh_dcm_from_rv, eci_to_lvlh
from Functions.plotting import (plot_single_sat_lvlh_components,
                                plot_relative_lvlh, plot_osculating_oe)
from Functions.oe_spacing_optimizer_new import compute_roe, roe_to_lvlh

# ── Environment ───────────────────────────────────────────────────────────────
spice.load_standard_kernels()

simulation_start_epoch = DateTime(2026, 1, 14).to_epoch()
simulation_end_epoch   = simulation_start_epoch + 100 * 3600

mothership_name = "LFIRE-0"
deputy_name     = "LFIRE-1"
sat_names       = [mothership_name, deputy_name]

bodies   = make_bodies(sat_names)
mu_earth = bodies.get("Earth").gravitational_parameter

# ── Chief (mothership) Keplerian elements ────────────────────────────────────
# eccentricity = 0.00 for cleanest OE comparison; swap to 4.03e-3 later
base_kepler = dict(
    semi_major_axis             = 6.99276221e+06,
    eccentricity                = 0.00,
    inclination                 = 1.71065169e+00,
    argument_of_periapsis       = 1.31226971e+00,
    longitude_of_ascending_node = 3.82958313e-01,
    true_anomaly                = 3.07018490e+00,
)

A   = 15e3                                               # ring radius, m
eps = 0.3                                               # radial fraction (ε); S amplitude = 2·eps·A
a   = base_kepler["semi_major_axis"]
n   = np.sqrt(mu_earth / a**3)

# ── Helper: build deputy Keplerian elements from OE offsets ──────────────────
def build_deputy_from_oe_offsets(base_kepler, da=0.0, de=0.0, di=0.0,
                                 dOmega=0.0, domega=0.0, dnu=0.0):
    kep = dict(base_kepler)
    kep["semi_major_axis"]             += da
    kep["eccentricity"]                += de
    kep["inclination"]                 += di
    kep["longitude_of_ascending_node"] += dOmega
    kep["argument_of_periapsis"]       += domega
    kep["true_anomaly"]                += dnu
    return kep

# ── Deputy initialization: mostly S-W ring ────────────────────────────────────
# the desired S = A·cos(θ) cannot be achieved independently of R via OE
# differences alone (CW 2:1 coupling forces S amplitude = 2·R amplitude).

deputy_kepler = build_deputy_from_oe_offsets(
    base_kepler,
    da = 0.0,
    de = eps * A / a,
    di = A / a,          # W amp = A
)

state0  = element_conversion.keplerian_to_cartesian_elementwise(
              gravitational_parameter=mu_earth, **base_kepler)
state_d = element_conversion.keplerian_to_cartesian_elementwise(
              gravitational_parameter=mu_earth, **deputy_kepler)

r0_eci,    v0_eci    = np.array(state0[:3]),  np.array(state0[3:])
r_dep_eci, v_dep_eci = np.array(state_d[:3]), np.array(state_d[3:])

# ── ROE analysis: predict initial LVLH from OE differences ───────────────────
roe = compute_roe(base_kepler, deputy_kepler)
u_c = base_kepler["argument_of_periapsis"] + base_kepler["true_anomaly"]
rho_predicted = roe_to_lvlh(roe, a, base_kepler["eccentricity"],
                             base_kepler["inclination"], u_c)

print("\n── Relative Orbital Elements ────────────────────────────────────────────")
labels = ["δa  ", "δe_x", "δe_y", "δi  ", "δΩ  ", "δu  "]
for lbl, val in zip(labels, roe):
    print(f"  {lbl} = {val:.6e}")

print("\n── Predicted initial LVLH (linearized ROE → LVLH) ──────────────────────")
print(f"  R = {rho_predicted[0]/1e3:+.4f} km  (desired ≈ {eps*A/1e3:.4f} km)")
print(f"  S = {rho_predicted[1]/1e3:+.4f} km  (CW coupling gives S ≈ {2*eps*A/1e3:.4f} km, not {A/1e3:.1f} km)")
print(f"  W = {rho_predicted[2]/1e3:+.4f} km  (desired ≈ {A/1e3:.4f} km)")

# ── δa enforcement check (hard stop) ─────────────────────────────────────────
oe_chief = element_conversion.cartesian_to_keplerian(
               np.concatenate([r0_eci,    v0_eci   ]), mu_earth)
oe_dep   = element_conversion.cartesian_to_keplerian(
               np.concatenate([r_dep_eci, v_dep_eci]), mu_earth)

delta_a = oe_dep[0] - oe_chief[0]

print("\n── δa check ─────────────────────────────────────────────────────────────")
print(f"  chief a  = {oe_chief[0]:.6f} m   T = {2*np.pi*np.sqrt(oe_chief[0]**3/mu_earth)/60:.4f} min")
print(f"  deputy a = {oe_dep[0]:.6f} m   T = {2*np.pi*np.sqrt(oe_dep[0]**3/mu_earth)/60:.4f} min")
print(f"  δa       = {delta_a:.6e} m")

TOL_A = 1.0  # 1 m — numerical round-trip noise only
if abs(delta_a) > TOL_A:
    raise RuntimeError(
        f"Deputy has δa = {delta_a:.3e} m > {TOL_A} m; secular drift expected. Fix ICs.")

# ── CW boundedness cross-check from achieved LVLH ICs (Eq. 5.23) ─────────────
C_to_eci, _ = lvlh_dcm_from_rv(r0_eci, v0_eci)
omega_scalar = np.linalg.norm(np.cross(r0_eci, v0_eci)) / np.linalg.norm(r0_eci)**2
omega_lvlh   = np.array([0.0, 0.0, omega_scalar])

dr_eci = r_dep_eci - r0_eci
dv_eci = v_dep_eci - v0_eci
rho_lvlh     = C_to_eci.T @ dr_eci
rho_dot_lvlh = C_to_eci.T @ dv_eci - np.cross(omega_lvlh, rho_lvlh)

cw_drift = rho_dot_lvlh[1] + 2*n*rho_lvlh[0]  # Eq. 5.23: ≈ 0 for bounded orbit

print("\n── CW local check (Eq. 5.23) ────────────────────────────────────────────")
print(f"  R(0)           = {rho_lvlh[0]/1e3:.4f} km")
print(f"  S(0)           = {rho_lvlh[1]/1e3:.4f} km")
print(f"  W(0)           = {rho_lvlh[2]/1e3:.4f} km")
print(f"  Ṡ(0) + 2nR(0) = {cw_drift:.6e} m/s  (0 → bounded)")
print(f"\n  Initial separation: {np.linalg.norm(dr_eci)/1e3:.3f} km")

# ── Propagate (two-body, point-mass Earth only) ───────────────────────────────
initial_state = np.concatenate([r0_eci, v0_eci, r_dep_eci, v_dep_eci])

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
t, t_hours   = extract_time_arrays(states_array)
rv           = extract_rv(states_array, sat_names)
N            = len(t)
print(f"  {N} timesteps,  duration={t_hours[-1]:.2f} h")

# ── Build OE_osc dict for plot_osculating_oe ─────────────────────────────────
OE_osc = {}
for name in sat_names:
    r_arr, v_arr = rv[name]
    oes = np.array([rv_to_kepler(r_arr[k], v_arr[k]) for k in range(N)])
    OE_osc[name] = oes.T  # (6, N)

delta_a_hist = OE_osc[deputy_name][0] - OE_osc[mothership_name][0]
print(f"\n── δa over simulation ───────────────────────────────────────────────────")
print(f"  mean δa  = {delta_a_hist.mean():.4e} m")
print(f"  max |δa| = {np.abs(delta_a_hist).max():.4e} m")

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. LVLH R/S/W vs time
plot_single_sat_lvlh_components(rv, deputy_name, mothership_name, t_hours)

# 2. Relative LVLH projections — all planes (RS, RW, SW)
plot_relative_lvlh(rv, mothership_name, deputy_name, t_hours, plane="all")

# 3. Osculating orbital elements — 'a' panel confirms δa=0 throughout
plot_osculating_oe(OE_osc, sat_names, t_hours)

plt.show()
