import numpy as np
from tudatpy.interface import spice
from tudatpy.astro.time_representation import DateTime

from Functions.sim_setup import make_bodies, make_propagator_settings, run_simulation
from Functions.initial_conditions import build_swarm_initial_state
from Functions.postprocess import extract_time_arrays, extract_rv
from Functions.plotting import (
    plot_eci_3d, plot_radius_norm, plot_speed_norm, plot_components,
    plot_energy, plot_ang_momentum, plot_separation_to_mothership
)
from Functions.analysis import max_pairwise_separation

spice.load_standard_kernels()

simulation_start_epoch = DateTime(2026, 1, 14).to_epoch()
simulation_end_epoch   = DateTime(2026, 1, 15).to_epoch()

mothership_name = "LFIRE-0"
deputy_names = [f"LFIRE-{i}" for i in range(1, 5)]
sat_names = [mothership_name] + deputy_names

bodies = make_bodies(sat_names)
mu_earth = bodies.get("Earth").gravitational_parameter

base_kepler = dict(
    semi_major_axis=6.99276221e+06,
    eccentricity=4.03294322e-03,
    inclination=1.71065169e+00,
    argument_of_periapsis=1.31226971e+00,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e+00,
)

initial_state = build_swarm_initial_state(
    mu_earth=mu_earth,
    base_kepler=base_kepler,
    R=15e3,
    thetas_rad=np.deg2rad([0, 90, 180, 270])
)

propagator_settings = make_propagator_settings(
    bodies=bodies,
    sat_names=sat_names,
    initial_state=initial_state,
    t0=simulation_start_epoch,
    tf=simulation_end_epoch,
    dt=10.0
)

states_array = run_simulation(bodies, propagator_settings)

t, t_hours = extract_time_arrays(states_array)
rv = extract_rv(states_array, sat_names)

plot_eci_3d(rv, sat_names, title="Swarm trajectories (ECI)")
plot_radius_norm(rv, sat_names, t_hours)
plot_speed_norm(rv, sat_names, t_hours)
plot_components(rv, sat_names, t_hours)
plot_energy(rv, sat_names, t_hours, mu=mu_earth)
plot_ang_momentum(rv, sat_names, t_hours)
plot_separation_to_mothership(rv, sat_names, t_hours, mothership_name)

print("Max pairwise separation [km]:", max_pairwise_separation(rv, sat_names)/1e3)
