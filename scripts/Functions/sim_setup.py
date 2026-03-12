# Functions/sim_setup.py
from tudatpy.dynamics import environment_setup, propagation_setup, simulator

# Satellite physical properties (adjust these to match your actual satellites)
SAT_MASS           = 10.0   # kg
SAT_REF_AREA       = 0.1    # m^2
SAT_DRAG_COEFF     = 1.2    # dimensionless
SAT_RAD_COEFF      = 1.2    # dimensionless


def make_bodies(sat_names: list[str]):
    # Include all third bodies needed for perturbations
    bodies_to_create = ["Earth", "Sun", "Moon", "Mars", "Venus"]

    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, "Earth", "J2000"
    )

    for name in sat_names:
        body_settings.add_empty_settings(name)

        # Mass
        body_settings.get(name).constant_mass = SAT_MASS

        # Aerodynamic coefficients (constant drag model)
        body_settings.get(name).aerodynamic_coefficient_settings = (
            environment_setup.aerodynamic_coefficients.constant(
                SAT_REF_AREA,
                [SAT_DRAG_COEFF, 0.0, 0.0]
            )
        )

        


    return environment_setup.create_system_of_bodies(body_settings)


def make_propagator_settings(bodies, sat_names, initial_state, t0, tf, dt):
    bodies_to_propagate = sat_names
    central_bodies = ["Earth"] * len(bodies_to_propagate)

    acc_settings_each_sat = {
        "Earth": [
            propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
            propagation_setup.acceleration.aerodynamic(),
        ],
        "Sun": [
            propagation_setup.acceleration.point_mass_gravity()
        ],
        "Moon":  [propagation_setup.acceleration.point_mass_gravity()],
        "Mars":  [propagation_setup.acceleration.point_mass_gravity()],
        "Venus": [propagation_setup.acceleration.point_mass_gravity()],
    }
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