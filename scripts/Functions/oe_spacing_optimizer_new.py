import numpy as np
from scipy.optimize import minimize
from tudatpy.astro import element_conversion


# ── Anomaly utilities ─────────────────────────────────────────────────────────

def wrap_to_2pi(angle):
    return np.mod(angle, 2.0 * np.pi)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def true_to_eccentric_anomaly(nu, e):
    return 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(nu / 2.0),
        np.sqrt(1.0 + e) * np.cos(nu / 2.0)
    )


def eccentric_to_mean_anomaly(E, e):
    return E - e * np.sin(E)


def mean_to_eccentric_anomaly(M, e, tol=1e-9, max_iter=50):
    M = wrap_to_pi(M)
    E = M

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE

        if abs(dE) < tol:
            break

    return E


def eccentric_to_true_anomaly(E, e):
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0)
    )


# ── Build deputy orbit using delta Omega and delta M ─────────────────────────

def kepler_with_Omega_M(base_kepler, delta_Omega, delta_M):
    """
    Build deputy Keplerian elements by keeping a, e, i, omega fixed
    and perturbing only RAAN Omega and mean anomaly M.
    """

    e = base_kepler["eccentricity"]
    nu0 = base_kepler["true_anomaly"]

    E0 = true_to_eccentric_anomaly(nu0, e)
    M0 = eccentric_to_mean_anomaly(E0, e)

    Mi = M0 + delta_M
    Ei = mean_to_eccentric_anomaly(Mi, e)
    nui = eccentric_to_true_anomaly(Ei, e)

    deputy_kepler = dict(base_kepler)

    deputy_kepler["longitude_of_ascending_node"] = wrap_to_2pi(
        base_kepler["longitude_of_ascending_node"] + delta_Omega
    )

    deputy_kepler["true_anomaly"] = wrap_to_2pi(nui)

    return deputy_kepler


def state_from_kepler(mu_earth, kepler):
    return element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=mu_earth,
        **kepler
    )


# ── Relative Orbital Elements ────────────────────────────────────────────────

def compute_roe(base_kepler, deputy_kepler):
    """
    Compute absolute Relative Orbital Elements:

        [Delta a, Delta e_x, Delta e_y, Delta i, Delta Omega, Delta u]

    where:

        Delta e_x = e_d cos(omega_d) - e_c cos(omega_c)
        Delta e_y = e_d sin(omega_d) - e_c sin(omega_c)

    and:

        u = omega + nu
    """

    Da = (
        deputy_kepler["semi_major_axis"]
        - base_kepler["semi_major_axis"]
    )

    De_x = (
        deputy_kepler["eccentricity"]
        * np.cos(deputy_kepler["argument_of_periapsis"])
        - base_kepler["eccentricity"]
        * np.cos(base_kepler["argument_of_periapsis"])
    )

    De_y = (
        deputy_kepler["eccentricity"]
        * np.sin(deputy_kepler["argument_of_periapsis"])
        - base_kepler["eccentricity"]
        * np.sin(base_kepler["argument_of_periapsis"])
    )

    Di = (
        deputy_kepler["inclination"]
        - base_kepler["inclination"]
    )

    DOmega = wrap_to_pi(
        deputy_kepler["longitude_of_ascending_node"]
        - base_kepler["longitude_of_ascending_node"]
    )

    Du = wrap_to_pi(
        deputy_kepler["argument_of_periapsis"]
        + deputy_kepler["true_anomaly"]
        - base_kepler["argument_of_periapsis"]
        - base_kepler["true_anomaly"]
    )

    return np.array([Da, De_x, De_y, Di, DOmega, Du])


def roe_to_lvlh(roe, a_c, e_c, i_c, u_c):
    """
    Linearized instantaneous ROE -> LVLH mapping.

    Valid for small relative separations and near-circular chief orbits.

    Parameters
    ----------
    roe : array-like
        [Delta a, Delta e_x, Delta e_y, Delta i, Delta Omega, Delta u]

    a_c : float
        Chief semi-major axis.

    e_c : float
        Chief eccentricity. Currently unused except kept for interface consistency.

    i_c : float
        Chief inclination.

    u_c : float
        Chief argument of latitude, omega_c + nu_c.

    Returns
    -------
    np.ndarray
        [delta R, delta S, delta W] in metres.
    """

    Da, De_x, De_y, Di, DOmega, Du = roe

    dR = Da - a_c * (
        De_x * np.cos(u_c)
        + De_y * np.sin(u_c)
    )

    dS = a_c * (Du + DOmega * np.cos(i_c))

    dW = a_c * (
        Di * np.sin(u_c)
        - DOmega * np.sin(i_c) * np.cos(u_c)
    )

    return np.array([dR, dS, dW])


# ── ROE-based relative LVLH using delta Omega and delta M ────────────────────

def relative_lvlh_from_delta_Omega_M(
    mu_earth,
    base_kepler,
    delta_Omega,
    delta_M
):
    """
    Deputy LVLH position relative to the chief via:

        (delta Omega, delta M)
            -> deputy OE
            -> ROE
            -> linearized LVLH

    The ECI Cartesian state is still returned for propagation.
    """

    deputy_kepler = kepler_with_Omega_M(
        base_kepler,
        delta_Omega,
        delta_M
    )

    state_i = state_from_kepler(mu_earth, deputy_kepler)

    roe = compute_roe(base_kepler, deputy_kepler)

    a_c = base_kepler["semi_major_axis"]
    e_c = base_kepler["eccentricity"]
    i_c = base_kepler["inclination"]
    u_c = (
        base_kepler["argument_of_periapsis"]
        + base_kepler["true_anomaly"]
    )

    rho_lvlh = roe_to_lvlh(
        roe,
        a_c,
        e_c,
        i_c,
        u_c
    )

    return rho_lvlh, state_i, deputy_kepler


# ── Optimizer ────────────────────────────────────────────────────────────────

def objective_delta_Omega_M(
    x,
    mu_earth,
    base_kepler,
    rho_desired,
    weights=None
):
    """
    Objective function for optimizing delta Omega and delta M.
    """

    delta_Omega = x[0]
    delta_M = x[1]

    rho_lvlh, _, _ = relative_lvlh_from_delta_Omega_M(
        mu_earth,
        base_kepler,
        delta_Omega,
        delta_M
    )

    if weights is None:
        weights = np.ones(3)

    err = weights * (rho_lvlh - rho_desired)

    cost = float(err @ err)

    return cost


def solve_delta_Omega_M_for_lvlh_target(
    mu_earth,
    base_kepler,
    rho_desired,
    weights=None,
    bounds=None
):
    """
    Solve for (delta Omega, delta M) that best reproduces a desired LVLH offset.

    This version optimizes:

        delta Omega : RAAN difference
        delta M     : mean anomaly difference

    and keeps:

        a, e, i, omega

    fixed relative to the chief.
    """

    a = base_kepler["semi_major_axis"]

    # Initial guess:
    # along-track distance S is approximately a * delta_M
    x0 = np.array([
        0.0,
        rho_desired[1] / a
    ])

    if bounds is None:
        bounds = [
            (-0.25, 0.25),   # delta Omega, rad
            (-0.25, 0.25),   # delta M, rad
        ]

    result = minimize(
        objective_delta_Omega_M,
        x0,
        args=(
            mu_earth,
            base_kepler,
            rho_desired,
            weights
        ),
        method="L-BFGS-B",
        bounds=bounds
    )

    delta_Omega = result.x[0]
    delta_M = result.x[1]

    rho_lvlh, state_i, deputy_kepler = relative_lvlh_from_delta_Omega_M(
        mu_earth,
        base_kepler,
        delta_Omega,
        delta_M
    )

    info = {
        "success": result.success,
        "message": result.message,
        "cost": result.fun,
        "delta_Omega": delta_Omega,
        "delta_M": delta_M,
        "rho_desired": rho_desired,
        "rho_achieved": rho_lvlh,
        "rho_error": rho_lvlh - rho_desired,
        "deputy_kepler": deputy_kepler,
    }

    return state_i, info