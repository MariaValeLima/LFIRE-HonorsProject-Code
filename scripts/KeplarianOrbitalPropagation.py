##Imports here 
import numpy as np
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
# Load spice kernels
spice.load_standard_kernels()


##Now we start configuring how long we want to propagate it for
simulation_start_epoch = DateTime(2026, 1, 14).to_epoch()
simulation_end_epoch   = DateTime(2026, 1, 28).to_epoch()

# Create earth
bodies_to_create = ["Earth"]

# Create body settings "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Add empty settings to body settings
body_settings.add_empty_settings("LFIRE-A")
body_settings.add_empty_settings("LFIRE-B")
# Create system of bodies (in this case only Earth)
bodies = environment_setup.create_system_of_bodies(body_settings)

# Define bodies that are propagated
bodies_to_propagate = ["LFIRE-A", "LFIRE-B"]

# Define central bodies of propagation
central_bodies = ["Earth", "Earth"] #Because this is your reference point for every sat

# Define accelerations acting on sat
acc_settings_each_sat = {
    "Earth": [propagation_setup.acceleration.point_mass_gravity()]
}

acceleration_settings = {
    "LFIRE-A": acc_settings_each_sat,
    "LFIRE-B": acc_settings_each_sat
}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)
# Create termination settings
termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    time_step = 10.0,
    coefficient_set = propagation_setup.integrator.CoefficientSets.rk_4 )


# Set initial conditions for the satellite that will be
# propagated in this simulation. The initial conditions are given in
# Keplerian elements and later on converted to Cartesian elements
earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
initial_stateA0 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=earth_gravitational_parameter,
    semi_major_axis=6.99276221e+06,
    eccentricity=4.03294322e-03,
    inclination=1.71065169e+00,
    argument_of_periapsis=1.31226971e+00,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e+00,
)

initial_stateB0 = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=earth_gravitational_parameter,
    semi_major_axis=6.99276221e+06+300e3,
    eccentricity=4.03294322e-03,
    inclination=1.71065169e+00,
    argument_of_periapsis=1.31226971e+00,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e+00 +0.05,
)

initial_state = np.hstack((initial_stateA0, initial_stateB0))
# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings
)

# Create simulation object and propagate the dynamics
dynamics_simulator = simulator.create_dynamics_simulator(
    bodies, propagator_settings
)
# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)


rA = states_array[:, 1:4]
rB = states_array[:, 7:10]

print(
    f"""
Single Earth-Orbiting Satellite Example.
The initial position vector of LFIRE is [km]: \n{
    states[simulation_start_epoch][:3] / 1E3}
The initial velocity vector of LFIRE is [km/s]: \n{
    states[simulation_start_epoch][3:] / 1E3}
\nAfter {simulation_end_epoch} seconds the position vector of LFIRE is [km]: \n{
    states[simulation_end_epoch][:3] / 1E3}
And the velocity vector of LFIRE is [km/s]: \n{
    states[simulation_end_epoch][3:] / 1E3}
    """
)

# Define a 3D figure using pyplot
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'LFIRE trajectory around Earth')

# Plot the positional state history
ax.plot(rA[:,0], rA[:,1], rA[:,2], label="LFIRE-A")
ax.plot(rB[:,0], rB[:,1], rB[:,2], label="LFIRE-B")

ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')

# Add the legend and labels, then show the plot
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_aspect('equal')
plt.show()
