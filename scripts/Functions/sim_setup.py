# Functions/sim_setup.py
from tudatpy.dynamics import environment_setup, propagation_setup, simulator

def make_bodies(sat_names: list[str]):
    body_settings = environment_setup.get_default_body_settings(
        ["Earth"], "Earth", "J2000"
    )
    for name in sat_names:
        body_settings.add_empty_settings(name)
    return environment_setup.create_system_of_bodies(body_settings)

def make_propagator_settings(bodies, sat_names, initial_state, t0, tf, dt):
    bodies_to_propagate = sat_names
    central_bodies = ["Earth"] * len(bodies_to_propagate)

    acc_settings_each_sat = {"Earth": [propagation_setup.acceleration.point_mass_gravity()]}
    acceleration_settings = {name: acc_settings_each_sat for name in bodies_to_propagate}

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    termination_settings = propagation_setup.propagator.time_termination(tf)

    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=dt,
        coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    return propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        t0,
        integrator_settings,
        termination_settings
    )

def run_simulation(bodies, propagator_settings):
    dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)
    states = dynamics_simulator.propagation_results.state_history
    from tudatpy.util import result2array
    return result2array(states)
