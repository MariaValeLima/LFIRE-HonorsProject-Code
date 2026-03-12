# Functions/initial_conditions.py
import numpy as np
from tudatpy.astro import element_conversion
from Functions.formation_frames import apply_relative_state_lvlh

def build_swarm_initial_state(
    mu_earth,
    base_kepler,
    radius,
    thetas_rad,
    plane="RS",      # "RS", "SW", or "RW"
    dv_lvlh=None
):
    # Mothership
    state0 = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=mu_earth,
        **base_kepler
    )

    r0 = state0[:3]
    v0 = state0[3:]

    if dv_lvlh is None:
        dv_lvlh = np.zeros(3)

    initial_states_list = [state0]

    for th in thetas_rad:

        # Default zero
        dr_lvlh = np.zeros(3)

        if plane == "RS":
            dr_lvlh[0] = radius * np.cos(th)   # R
            dr_lvlh[1] = radius * np.sin(th)   # S

        elif plane == "SW":
            dr_lvlh[1] = radius * np.cos(th)   # S
            dr_lvlh[2] = radius * np.sin(th)   # W

        elif plane == "RW":
            dr_lvlh[0] = radius * np.cos(th)   # R
            dr_lvlh[2] = radius * np.sin(th)   # W

        else:
            raise ValueError("plane must be 'RS', 'SW', or 'RW'")

        r_i, v_i = apply_relative_state_lvlh(r0, v0, dr_lvlh, dv_lvlh)
        initial_states_list.append(np.hstack((r_i, v_i)))

    return np.hstack(initial_states_list)
