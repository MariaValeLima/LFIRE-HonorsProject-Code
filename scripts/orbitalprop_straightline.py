
import sys
from pathlib import Path
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT        = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from tudatpy.interface import spice
from tudatpy.astro.time_representation import DateTime
from Functions.sim_setup import make_bodies, make_propagator_settings, run_simulation
from Functions.initial_conditions import build_swarm_straightline
from Functions.postprocess import extract_time_arrays, extract_rv, rv_to_kepler
from Functions.plotting import (
    plot_eci_3d, plot_osculating_oe,
    plot_relative_lvlh, plot_swarm_dispersion, plot_swarm_dispersion_LVLH,
    plot_neighbor_distances
)
from Functions.analysis import max_pairwise_separation

# ── Environment ───────────────────────────────────────────────────────────────
spice.load_standard_kernels()

simulation_start_epoch = DateTime(2026, 1, 14).to_epoch()
simulation_end_epoch   = simulation_start_epoch + 60 * 3600

mothership_name = "LFIRE-0"
deputy_names    = ["LFIRE-B", "LFIRE-C", "LFIRE-D", "LFIRE-E"]
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

# Straight-line formation: deputies at -30, -15, +15, +30 km along-track (S)
# ν-shift initialization — all deputies share the same orbit, only ν differs
initial_state = build_swarm_straightline(
    mu_earth, base_kepler,
    s_offsets_m=[-30e3, -15e3, 15e3, 30e3]
)

propagator_settings = make_propagator_settings(
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
N            = len(t)
Ts           = float(t[1] - t[0])
print(f"  {N} timesteps,  dt={Ts:.1f} s,  duration={t_hours[-1]:.2f} h")

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
plot_relative_lvlh(rv, mothership_name, "LFIRE-E", t_hours, plane="all")
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
