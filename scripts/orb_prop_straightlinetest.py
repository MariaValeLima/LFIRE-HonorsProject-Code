import sys
from pathlib import Path
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT        = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from tudatpy.interface import spice
from tudatpy.astro.time_representation import DateTime
from tudatpy.astro import element_conversion
from Functions.sim_setup import make_bodies, make_propagator_settings, run_simulation
from Functions.postprocess import extract_time_arrays, extract_rv, rv_to_kepler
from Functions.plotting import (
    plot_eci_3d, plot_osculating_oe,
    plot_relative_lvlh, plot_swarm_dispersion, plot_swarm_dispersion_LVLH,
    plot_neighbor_distances
)
from Functions.analysis import max_pairwise_separation
from Functions.sim_setup import make_bodies, make_propagator_settings_twobody, run_simulation


# ── Environment ───────────────────────────────────────────────────────────────
spice.load_standard_kernels()

simulation_start_epoch = DateTime(2026, 1, 14).to_epoch()
simulation_end_epoch   = simulation_start_epoch + 60 * 3600

mothership_name = "LFIRE-0"
deputy_names    = ["LFIRE-B", "LFIRE-C"]
sat_names       = [mothership_name] + deputy_names

bodies   = make_bodies(sat_names)
mu_earth = bodies.get("Earth").gravitational_parameter

base_kepler = dict(
    semi_major_axis             = 6.99276221e+06,
    eccentricity                = 4.03294322e-03,
    inclination                 = 1.71065169e+00,
    argument_of_periapsis       = 1.31226971e+00,
    longitude_of_ascending_node = 3.82958313e-01,
    true_anomaly                = 3.07018490e+00,
)

# Straight-line formation: 1 deputy ±15 km along-track, initialized directly in OE
# All elements identical to mothership; only true_anomaly differs by δν = 15 km / a
a   = base_kepler['semi_major_axis']
nu0 = base_kepler['true_anomaly']
delta_nu = 15e3 / a                        # rad

state_mother = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=mu_earth, **base_kepler
)
kepler_B = {**base_kepler, 'true_anomaly': nu0 - delta_nu}   # behind
state_B  = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=mu_earth, **kepler_B
)
kepler_C = {**base_kepler, 'true_anomaly': nu0 + delta_nu}   # ahead
state_C  = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=mu_earth, **kepler_C
)
initial_state = np.hstack([state_mother, state_B, state_C])

propagator_settings = make_propagator_settings_twobody(
    bodies=bodies, sat_names=sat_names,
    initial_state=initial_state,
    t0=simulation_start_epoch,
    tf=simulation_end_epoch,
    dt=10.0
)

# ── Propagate ─────────────────────────────────────────────────────────────────
print("Propagating straight-line formation...")
states_array = run_simulation(bodies, propagator_settings)
t, t_hours   = extract_time_arrays(states_array)
rv           = extract_rv(states_array, sat_names)

# ── Initial Keplerian elements diagnostic ─────────────────────────────────────
from Functions.formation_frames import lvlh_dcm_from_rv, eci_to_lvlh

print("\n--- Initial Keplerian elements [TUDAT: a, e, i, ω, Ω, ν] ---")
for name in sat_names:
    r_d, v_d = rv[name]
    oe = element_conversion.cartesian_to_keplerian(
        np.hstack((r_d[0], v_d[0])), mu_earth
    )
    print(f"  {name:<10}  a={oe[0]:.2f}  e={oe[1]:.7f}  i={oe[2]:.7f}"
          f"  ω={oe[3]:.7f}  Ω={oe[4]:.7f}  ν={oe[5]:.7f}")

r0_0, v0_0 = rv[mothership_name]
_, C0 = lvlh_dcm_from_rv(r0_0[0], v0_0[0])

print("\n--- Initial LVLH relative position and velocity ---")
print(f"  {'name':<10}  {'dR (m)':>10}  {'dS (m)':>10}  {'dW (m)':>10}"
      f"  {'dvR (m/s)':>10}  {'dvS (m/s)':>10}  {'dvW (m/s)':>10}")

for name in deputy_names:
    r_d, v_d = rv[name]
    dr = eci_to_lvlh(C0, r_d[0] - r0_0[0])
    dv = eci_to_lvlh(C0, v_d[0] - v0_0[0])
    print(f"  {name:<10}  {dr[0]:>10.2f}  {dr[1]:>10.2f}  {dr[2]:>10.2f}"
          f"  {dv[0]:>10.5f}  {dv[1]:>10.5f}  {dv[2]:>10.5f}")

print()

N  = len(t)
Ts = float(t[1] - t[0])
print(f"  {N} timesteps,  dt={Ts:.1f} s,  duration={t_hours[-1]:.2f} h")

# ── Orbital closure check ─────────────────────────────────────────────────────
a0    = base_kepler['semi_major_axis']
T_orb = 2 * np.pi * np.sqrt(a0**3 / mu_earth)
idx   = int(round(T_orb / Ts))

print(f"\nOrbital period: {T_orb:.1f} s  (step {idx}, t = {t[idx]-t[0]:.1f} s)")
print(f"{'Satellite':<12}  {'|Δr| (m)':>12}  {'|Δv| (mm/s)':>14}")

for name in sat_names:
    r_arr, v_arr = rv[name]
    dr = np.linalg.norm(r_arr[idx] - r_arr[0])
    dv = np.linalg.norm(v_arr[idx] - v_arr[0]) * 1e3
    print(f"{name:<12}  {dr:>12.3f}  {dv:>14.3f}")

print()

# ── Osculating orbital elements ───────────────────────────────────────────────
OE_osc = {}

for name in sat_names:
    r_arr, v_arr = rv[name]
    oe = np.zeros((6, len(t)))
    for k in range(len(t)):
        oe[:, k] = rv_to_kepler(r_arr[k], v_arr[k])
    OE_osc[name] = oe

# ── Plots ─────────────────────────────────────────────────────────────────────
plot_eci_3d(rv, sat_names, title="Straight-line swarm trajectories (ECI)")
plot_osculating_oe(OE_osc, sat_names, t_hours)
plot_relative_lvlh(rv, mothership_name, "LFIRE-B", t_hours, plane="all")
plot_swarm_dispersion(rv, sat_names, t_hours, mothership_name)
plot_neighbor_distances(rv, sat_names, t_hours, mothership_name)
plot_swarm_dispersion_LVLH(rv, sat_names, t_hours, mothership_name)

print("Max pairwise separation [km]:", max_pairwise_separation(rv, sat_names) / 1e3)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = ROOT / "O2M_project" / "tudat_states_straightline.npz"

save_dict = {"t": t, "sat_names": sat_names, "Ts": Ts}

for name in sat_names:
    r, v = rv[name]
    save_dict[f"r_{name}"] = r
    save_dict[f"v_{name}"] = v

np.savez(str(out_path), **save_dict)
print(f"Saved to: {out_path}")