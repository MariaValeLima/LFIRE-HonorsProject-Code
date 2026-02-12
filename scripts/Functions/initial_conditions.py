# Functions/initial_conditions.py
import numpy as np
from tudatpy.astro import element_conversion
from Functions.formation_frames import apply_relative_state_lvlh

def build_swarm_initial_state(mu_earth: float, base_kepler: dict, R: float, thetas_rad: np.ndarray):
    state0 = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=mu_earth,
        **base_kepler
    )
    r0, v0 = state0[:3], state0[3:]

    initial_states_list = [state0]

    for th in thetas_rad:
        dr_lvlh = np.array([R*np.cos(th), R*np.sin(th), 0.0])
        dv_lvlh = np.zeros(3)
        r_i, v_i = apply_relative_state_lvlh(r0, v0, dr_lvlh, dv_lvlh)
        initial_states_list.append(np.hstack((r_i, v_i)))

    return np.hstack(initial_states_list)
