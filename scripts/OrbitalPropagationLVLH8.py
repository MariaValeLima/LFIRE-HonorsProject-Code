import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime

from Functions.formation_frames import apply_relative_state_lvlh


spice.load_standard_kernels()

# Simulation time
simulation_start_epoch = DateTime(2026, 1, 14).to_epoch()
simulation_end_epoch   = DateTime(2026, 1, 15).to_epoch()

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


# STEP 1: mothership orbit -> ECI

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

# Build initial cartesian states list (for now: ONLY has the mothership)
initial_states_list = []

# Mothership first (Keplerian -> ECI Cartesian)
state0 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=mu_earth,
    **base_kepler
)
initial_states_list.append(state0)

#split mothership state into r0, v0
r0 = state0[:3]
v0 = state0[3:]

# Optional quick sanity print
print("Mothership |r0| [km] =", np.linalg.norm(r0)/1e3)
print("Mothership |v0| [m/s] =", np.linalg.norm(v0))


R = 15e3  # meters
thetas = np.deg2rad([0, 90, 180, 270])

initial_states_list = [state0]

for th in thetas:
    dr_lvlh = np.array([R*np.cos(th), R*np.sin(th), 0.0])
    dv_lvlh = np.array([0.0, 0.0, 0.0])  # start simple

    r_i, v_i = apply_relative_state_lvlh(r0, v0, dr_lvlh, dv_lvlh)
    state_i = np.hstack((r_i, v_i))
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

mu = mu_earth 
# ---------- 3D Trajectories in ECI ----------
fig = plt.figure(figsize=(6, 6), dpi=125)
ax = fig.add_subplot(111, projection="3d")
ax.set_title("5-sat swarm trajectories around Earth (ECI)")

for name in sat_names:
    r, v = rv[name]
    ax.plot(r[:, 0], r[:, 1], r[:, 2], label=name)

ax.scatter(0.0, 0.0, 0.0, label="Earth", marker="o")
ax.legend()
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_box_aspect((1, 1, 1))  # better than set_aspect("equal") for 3D
plt.tight_layout()
plt.show(block=False)

# ---------- Radius magnitude ----------
plt.figure(figsize=(10,4), dpi=125)
for name in sat_names:
    r, v = rv[name]
    rnorm_km = np.linalg.norm(r, axis=1) / 1e3
    plt.plot(t_hours, rnorm_km, label=name)
plt.xlabel("Time [hr]")
plt.ylabel("||r|| [km]")
plt.title("Radius magnitude vs time")
plt.legend(ncol=2)
plt.tight_layout()
plt.show(block=False)

# ---------- Speed magnitude ----------
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

# ---------- Components of r and v ----------
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

# ---------- Energy and angular momentum ----------
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

# ---------- Formation separation to mothership ----------
rC, vC = rv[mothership_name]
plt.figure(figsize=(10,4), dpi=125)
max_sep = 0.0
for name in sat_names:
    if name == mothership_name:
        continue
    r, v = rv[name]
    sep_km = np.linalg.norm(r - rC, axis=1) / 1e3
    max_sep = max(max_sep, np.max(sep_km))
    plt.plot(t_hours, sep_km, label=name)

plt.xlabel("Time [hr]")
plt.ylabel("Separation to mothership [km]")
plt.title(f"Deputy separation to mothership (max observed ~ {max_sep:.2f} km)")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# ---------- Pairwise max separation ----------
pairwise_max = 0.0
for i, ni in enumerate(sat_names):
    ri, vi = rv[ni]
    for j in range(i+1, len(sat_names)):
        nj = sat_names[j]
        rj, vj = rv[nj]
        pairwise_max = max(pairwise_max, np.max(np.linalg.norm(ri - rj, axis=1)))

print(f"Max pairwise separation over run: {pairwise_max/1e3:.3f} km")