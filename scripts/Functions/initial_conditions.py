# Functions/initial_conditions.py
import numpy as np
from tudatpy.astro import element_conversion
from Functions.formation_frames import apply_relative_state_lvlh
from Functions.oe_spacing_optimizer import solve_delta_omega_M_for_lvlh_target

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

        # Apply LVLH offset → ECI Cartesian state
        r_i, v_i = apply_relative_state_lvlh(r0, v0, dr_lvlh, dv_lvlh)

        # TUDAT array ordering: [0]=a  [1]=e  [2]=i  [3]=ω  [4]=Ω  [5]=ν
        # ω (arg. periapsis) is at index 3, Ω (RAAN) at index 4 — opposite of standard convention.
        # Only index 0 (SMA) is modified here, so the swap does not affect this correction.
        oe_i    = element_conversion.cartesian_to_keplerian(
                      np.hstack((r_i, v_i)), mu_earth)
        oe_i[0] = base_kepler['semi_major_axis']
        state_i = element_conversion.keplerian_to_cartesian(oe_i, mu_earth)

        initial_states_list.append(state_i)

    return np.hstack(initial_states_list)


def build_swarm_straightline(mu_earth, base_kepler, s_offsets_m):
    """
    Build initial state for a straight-line along-track formation.
    s_offsets_m: list of along-track offsets in metres (e.g. [-30e3, -15e3, 15e3, 30e3])
    Each deputy is placed at ν₀ + δν on the exact same Keplerian orbit as the mothership.
    δν = δs / a  (first-order arc-length approximation)
    """
    # Mothership
    state0 = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=mu_earth,
        **base_kepler
    )
    initial_states_list = [state0]

    a   = base_kepler['semi_major_axis']
    nu0 = base_kepler['true_anomaly']

    for ds in s_offsets_m:
        delta_nu = ds / a                       # radians
        kepler_i = dict(**base_kepler)
        kepler_i['true_anomaly'] = nu0 + delta_nu
        state_i = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=mu_earth,
            **kepler_i
        )
        initial_states_list.append(state_i)

    return np.hstack(initial_states_list)
def build_swarm_initial_state_optimized_oe(
    mu_earth,
    base_kepler,
    desired_lvlh_offsets,
    weights=None,
    beta_e=0.0,
    return_optimizer_results=False
):
    """
    New method:
    desired LVLH offsets -> optimize delta omega and delta M
    -> build deputy states from orbital elements.

    Keeps a,e,i,Omega fixed and changes only omega and M.
    """

    state0 = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=mu_earth,
        **base_kepler
    )

    initial_states_list = [state0]
    opt_results = []

    for rho_desired in desired_lvlh_offsets:
        state_i, info = solve_delta_omega_M_for_lvlh_target(
            mu_earth=mu_earth,
            base_kepler=base_kepler,
            rho_desired=np.asarray(rho_desired, dtype=float),
            weights=weights,
            beta_e=beta_e
        )

        initial_states_list.append(state_i)
        opt_results.append(info)

    initial_state = np.hstack(initial_states_list)

    if return_optimizer_results:
        return initial_state, opt_results

    return initial_state