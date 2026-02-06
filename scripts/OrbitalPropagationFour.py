import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime

spice.load_standard_kernels()

# Simulation time
simulation_start_epoch = DateTime(2026, 1, 14).to_epoch()
simulation_end_epoch   = DateTime(2026, 1, 28).to_epoch()

# Environment
bodies_to_create = ["Earth"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Define swarm names
mothership_name = "LFIRE-0"
deputy_names = [f"LFIRE-{i}" for i in range(1, 5)]
sat_names = [mothership_name] + deputy_names  # total 5

# Add them to body settings
for name in sat_names:
    body_settings.add_empty_settings(name)

bodies = environment_setup.create_system_of_bodies(body_settings)

# propagation setup
bodies_to_propagate = sat_names
central_bodies = ["Earth"] * len(bodies_to_propagate)

acc_settings_each_sat = {
    "Earth": [propagation_setup.acceleration.point_mass_gravity()]
}
acceleration_settings = {name: acc_settings_each_sat for name in bodies_to_propagate}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    time_step=10.0,
    coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
)

# Mothership base orbit 
mu_earth = bodies.get("Earth").gravitational_parameter

base_kepler = dict(
    semi_major_axis=6.99276221e+06,
    eccentricity=4.03294322e-03,
    inclination=1.71065169e+00,
    argument_of_periapsis=1.31226971e+00,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e+00,
)

# Offsets to the sattelites orbits
offsets = [
    (+30e3,  0.0),
    (-30e3,  0.0),
    ( 0.0,  +np.deg2rad(0.246)),
    ( 0.0,  -np.deg2rad(0.246)),
]

#Build initial cartesian states for all sats
initial_states_list = []

# mothership first
# keplarian to catrtesian elementwise - conversts orbital elements to states
state0 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=mu_earth,
    **base_kepler
)
initial_states_list.append(state0)

# swarm - loop once per offset pair (4times), copy the mothership settings, then apply the offsets to a and nu. Then they are converted to cartesian state
for (da_m, dnu_rad) in offsets:
    kep = base_kepler.copy()
    kep["semi_major_axis"] = base_kepler["semi_major_axis"] + da_m
    kep["true_anomaly"]    = base_kepler["true_anomaly"] + dnu_rad

    state_i = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=mu_earth,
        **kep
    )
    initial_states_list.append(state_i)

# Stack into one big initial state vector
initial_state = np.hstack(initial_states_list)

# Propagate 
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings
)

dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)

states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)

#extract r and v for every sat 
t = states_array[:, 0]  # seconds
t_hours = (t - t[0]) / 3600.0

rv = {}  # name -> (r Nx3, v Nx3)
for i, name in enumerate(sat_names):
    c = 1 + 6*i
    r = states_array[:, c:c+3]
    v = states_array[:, c+3:c+6]
    rv[name] = (r, v)


#Plot: extract each satellite position block
fig = plt.figure(figsize=(6, 6), dpi=125)
ax = fig.add_subplot(111, projection="3d")
ax.set_title("5-sat swarm trajectories around Earth")

for i, name in enumerate(sat_names):
    # Each sat has 6 states: [x,y,z,vx,vy,vz]
    col0 = 1 + 6*i
    r = states_array[:, col0:col0+3]
    ax.plot(r[:, 0], r[:, 1], r[:, 2], label=name)

ax.scatter(0.0, 0.0, 0.0, label="Earth", marker="o")

ax.legend()
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_aspect("equal")
plt.show(block=False)
plt.figure(figsize=(10,4), dpi=125)
for name in sat_names:
    r, v = rv[name]
    rnorm_km = np.linalg.norm(r, axis=1)/1e3
    plt.plot(t_hours, rnorm_km, label=name)
plt.xlabel("Time [hr]")
plt.ylabel("||r|| [km]")
plt.title("Radius magnitude vs time")
plt.legend(ncol=2)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(10,4), dpi=125)
for name in sat_names:
    r, v = rv[name]
    vnorm = np.linalg.norm(v, axis=1)
    plt.plot(t_hours, vnorm, label=name)
plt.xlabel("Time [hr]")
plt.ylabel("||v|| [m/s]")
plt.title("Speed magnitude vs time")
plt.legend(ncol=2)
plt.tight_layout()
plt.show(block=False)

fig, axs = plt.subplots(2, 3, figsize=(12,7), sharex=True, dpi=125)

for name in sat_names:
    r, v = rv[name]
    axs[0,0].plot(t_hours, r[:,0]/1e3, label=name)
    axs[0,1].plot(t_hours, r[:,1]/1e3)
    axs[0,2].plot(t_hours, r[:,2]/1e3)

    axs[1,0].plot(t_hours, v[:,0], label=name)
    axs[1,1].plot(t_hours, v[:,1])
    axs[1,2].plot(t_hours, v[:,2])

axs[0,0].set_ylabel("x [km]"); axs[0,1].set_ylabel("y [km]"); axs[0,2].set_ylabel("z [km]")
axs[1,0].set_ylabel("vx [m/s]"); axs[1,1].set_ylabel("vy [m/s]"); axs[1,2].set_ylabel("vz [m/s]")
axs[1,0].set_xlabel("Time [hr]"); axs[1,1].set_xlabel("Time [hr]"); axs[1,2].set_xlabel("Time [hr]")

axs[0,0].legend(ncol=2)
plt.tight_layout()
plt.show(block=False)

mu = mu_earth

plt.figure(figsize=(10,4), dpi=125)
for name in sat_names:
    r, v = rv[name]
    rnorm = np.linalg.norm(r, axis=1)
    vnorm = np.linalg.norm(v, axis=1)
    eps = 0.5*vnorm**2 - mu/rnorm
    plt.plot(t_hours, eps, label=name)
plt.xlabel("Time [hr]")
plt.ylabel("Specific energy ε [J/kg]")
plt.title("Specific mechanical energy (should be constant in 2-body)")
plt.legend(ncol=2)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(10,4), dpi=125)
for name in sat_names:
    r, v = rv[name]
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h, axis=1)
    plt.plot(t_hours, hnorm, label=name)
plt.xlabel("Time [hr]")
plt.ylabel("||h|| [m²/s]")
plt.title("Specific angular momentum magnitude (should be constant in 2-body)")
plt.legend(ncol=2)
plt.tight_layout()
plt.show(block=False)

r0, v0 = rv[mothership_name]
r0n = np.linalg.norm(r0, axis=1)
h0  = np.cross(r0, v0)
h0n = np.linalg.norm(h0, axis=1)

Rhat = r0 / r0n[:,None]
Chat = h0 / h0n[:,None]
Ihat = np.cross(Chat, Rhat)

def project(vec):
    dR = np.einsum("ij,ij->i", vec, Rhat)
    dI = np.einsum("ij,ij->i", vec, Ihat)
    dC = np.einsum("ij,ij->i", vec, Chat)
    return dR, dI, dC

fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True, dpi=125)

for name in sat_names:
    if name == mothership_name:
        continue
    r, v = rv[name]
    dr = r - r0
    dR, dI, dC = project(dr)
    axs[0].plot(t_hours, dR/1e3, label=name)
    axs[1].plot(t_hours, dI/1e3, label=name)
    axs[2].plot(t_hours, dC/1e3, label=name)

axs[0].set_ylabel("ΔR [km] (radial)")
axs[1].set_ylabel("ΔI [km] (in-track)")
axs[2].set_ylabel("ΔC [km] (cross-track)")
axs[2].set_xlabel("Time [hr]")
axs[0].set_title("Deputy relative position in RIC frame")
axs[0].legend(ncol=2)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(6,6), dpi=125)
for name in sat_names:
    if name == mothership_name:
        continue
    r, v = rv[name]
    dr = r - r0
    dR, dI, dC = project(dr)
    plt.plot(dR/1e3, dI/1e3, label=name)

plt.xlabel("ΔR [km]")
plt.ylabel("ΔI [km]")
plt.title("In-plane relative motion (ΔI vs ΔR)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

