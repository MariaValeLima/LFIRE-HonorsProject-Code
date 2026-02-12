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
simulation_end_epoch   = DateTime(2026, 1, 15).to_epoch()


# Environment
bodies_to_create = ["Earth"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Swarm
mothership_name = "LFIRE-0"
deputy_names = [f"LFIRE-{i}" for i in range(1, 5)]
sat_names = [mothership_name] + deputy_names  # total 5

# Add satellites to body settings
for name in sat_names:
    body_settings.add_empty_settings(name)

bodies = environment_setup.create_system_of_bodies(body_settings)

# Propagation setup (2-body point mass gravity)
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

# Mothership base orbit (Keplerian)
mu_earth = bodies.get("Earth").gravitational_parameter

base_kepler = dict(
    semi_major_axis=6.99276221e+06,
    eccentricity=4.03294322e-03,
    inclination=1.71065169e+00,
    argument_of_periapsis=1.31226971e+00,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e+00,
)

# Mothership cartesian state
state0 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=mu_earth,
    **base_kepler
)

r0 = state0[:3]
v0 = state0[3:]

# CW/Hill bounded relative-orbit initialization in RIC/LVLH
r0n = np.linalg.norm(r0)
h0 = np.cross(r0, v0)
h0n = np.linalg.norm(h0)

Rhat = r0 / r0n
Chat = h0 / h0n
Ihat = np.cross(Chat, Rhat)

# LVLH angular velocity vector (inertial) at t0
omega_vec = h0 / (r0n**2)

# Mean motion based on semi-major axis (CW assumption: near-circular chief)
a0 = base_kepler["semi_major_axis"]
n = np.sqrt(mu_earth / a0**3)

d = 15e3  # 15 km

# Deputies defined in RIC at t0: (x, y, z, xdot, ydot, zdot)
# x=radial, y=in-track, z=cross-track
deputy_ric_states = [
    (+d, 0.0, 0.0, 0.0, -2*n*(+d), 0.0),  # radial out, bounded
    (-d, 0.0, 0.0, 0.0, -2*n*(-d), 0.0),  # radial in, bounded
    (0.0, +d, 0.0, (n/2)*(+d), 0.0, 0.0), # along-track ahead, bounded
    (0.0, -d, 0.0, (n/2)*(-d), 0.0, 0.0), # along-track behind, bounded
]

def ric_to_inertial_state(x, y, z, xdot, ydot, zdot):
    # Position offset in inertial frame
    dr = x*Rhat + y*Ihat + z*Chat

    # Velocity offset in inertial frame:
    # v_inertial = v_chief + (v_rel in LVLH expressed inertial) + ω×dr
    dvlvlh = xdot*Rhat + ydot*Ihat + zdot*Chat
    dv = dvlvlh + np.cross(omega_vec, dr)

    ri = r0 + dr
    vi = v0 + dv
    return np.hstack((ri, vi))

# Build initial states
initial_states_list = [state0]
for x0, y0, z0, xdot0, ydot0, zdot0 in deputy_ric_states:
    initial_states_list.append(ric_to_inertial_state(x0, y0, z0, xdot0, ydot0, zdot0))

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

# Extract r and v for every sat
t = states_array[:, 0]
t_hours = (t - t[0]) / 3600.0

rv = {}
for i, name in enumerate(sat_names):
    c = 1 + 6*i
    r = states_array[:, c:c+3]
    v = states_array[:, c+3:c+6]
    rv[name] = (r, v)

# Plots
fig = plt.figure(figsize=(6, 6), dpi=125)
ax = fig.add_subplot(111, projection="3d")
ax.set_title("5-sat swarm trajectories around Earth (Option B init)")

for i, name in enumerate(sat_names):
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

# RIC relative motion plots (using mothership as reference)
r0_hist, v0_hist = rv[mothership_name]
r0n_hist = np.linalg.norm(r0_hist, axis=1)
h0_hist = np.cross(r0_hist, v0_hist)
h0n_hist = np.linalg.norm(h0_hist, axis=1)

Rhat_hist = r0_hist / r0n_hist[:, None]
Chat_hist = h0_hist / h0n_hist[:, None]
Ihat_hist = np.cross(Chat_hist, Rhat_hist)

def project(vec):
    dR = np.einsum("ij,ij->i", vec, Rhat_hist)
    dI = np.einsum("ij,ij->i", vec, Ihat_hist)
    dC = np.einsum("ij,ij->i", vec, Chat_hist)
    return dR, dI, dC

fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True, dpi=125)
for name in sat_names:
    if name == mothership_name:
        continue
    r, v = rv[name]
    dr = r - r0_hist
    dR, dI, dC = project(dr)
    axs[0].plot(t_hours, dR/1e3, label=name)
    axs[1].plot(t_hours, dI/1e3, label=name)
    axs[2].plot(t_hours, dC/1e3, label=name)

axs[0].set_ylabel("ΔR [km]")
axs[1].set_ylabel("ΔI [km]")
axs[2].set_ylabel("ΔC [km]")
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
    dr = r - r0_hist
    dR, dI, dC = project(dr)
    plt.plot(dR/1e3, dI/1e3, label=name)

plt.xlabel("ΔR [km]")
plt.ylabel("ΔI [km]")
plt.title("In-plane relative motion (ΔI vs ΔR)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
