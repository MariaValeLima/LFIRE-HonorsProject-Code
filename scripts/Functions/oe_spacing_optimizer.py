import numpy as np
from scipy.optimize import minimize
from tudatpy.astro import element_conversion

from Functions.formation_frames import lvlh_dcm_from_rv, eci_to_lvlh


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


def kepler_with_omega_M(base_kepler, delta_omega, delta_M):
    """
    Build deputy Keplerian elements by keeping a,e,i,Omega fixed
    and changing only omega and mean anomaly M.
    """

    e = base_kepler["eccentricity"]
    nu0 = base_kepler["true_anomaly"]

    E0 = true_to_eccentric_anomaly(nu0, e)
    M0 = eccentric_to_mean_anomaly(E0, e)

    Mi = M0 + delta_M
    Ei = mean_to_eccentric_anomaly(Mi, e)
    nui = eccentric_to_true_anomaly(Ei, e)

    deputy_kepler = dict(base_kepler)
    deputy_kepler["argument_of_periapsis"] = wrap_to_2pi(
        base_kepler["argument_of_periapsis"] + delta_omega
    )
    deputy_kepler["true_anomaly"] = wrap_to_2pi(nui)

    return deputy_kepler


def state_from_kepler(mu_earth, kepler):
    return element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=mu_earth,
        **kepler
    )


def relative_lvlh_from_delta_omega_M(
    mu_earth,
    base_kepler,
    delta_omega,
    delta_M
):
    """
    Returns the deputy LVLH position relative to LFIRE-0
    produced by delta_omega and delta_M.
    """

    state0 = state_from_kepler(mu_earth, base_kepler)
    r0 = state0[:3]
    v0 = state0[3:]

    deputy_kepler = kepler_with_omega_M(base_kepler, delta_omega, delta_M)
    state_i = state_from_kepler(mu_earth, deputy_kepler)
    ri = state_i[:3]

    _, C_ECI_to_LVLH = lvlh_dcm_from_rv(r0, v0)
    rho_lvlh = eci_to_lvlh(C_ECI_to_LVLH, ri - r0)

    return rho_lvlh, state_i, deputy_kepler


def relative_eccentricity_penalty(base_kepler, delta_omega):
    """
    Penalizes rotation of the eccentricity vector caused by changing omega.
    """

    e = base_kepler["eccentricity"]
    omega0 = base_kepler["argument_of_periapsis"]
    omegai = omega0 + delta_omega

    dex = e * (np.cos(omegai) - np.cos(omega0))
    dey = e * (np.sin(omegai) - np.sin(omega0))

    return dex**2 + dey**2


def objective_delta_omega_M(
    x,
    mu_earth,
    base_kepler,
    rho_desired,
    weights=None,
    beta_e=0.0
):
    delta_omega = x[0]
    delta_M = x[1]

    rho_lvlh, _, _ = relative_lvlh_from_delta_omega_M(
        mu_earth,
        base_kepler,
        delta_omega,
        delta_M
    )

    if weights is None:
        weights = np.ones(3)

    err = weights * (rho_lvlh - rho_desired)

    cost = float(err @ err)

    if beta_e > 0.0:
        cost += beta_e * relative_eccentricity_penalty(
            base_kepler,
            delta_omega
        )

    return cost


def solve_delta_omega_M_for_lvlh_target(
    mu_earth,
    base_kepler,
    rho_desired,
    weights=None,
    beta_e=0.0,
    bounds=None
):
    """
    Solve for delta_omega and delta_M that best reproduce a desired LVLH offset.
    """

    a = base_kepler["semi_major_axis"]

    # Along-track first guess: delta M ≈ S/a
    x0 = np.array([
        0.0,
        rho_desired[1] / a
    ])

    if bounds is None:
        bounds = [
            (-0.25, 0.25),   # delta omega, rad
            (-0.25, 0.25),   # delta M, rad
        ]

    result = minimize(
        objective_delta_omega_M,
        x0,
        args=(mu_earth, base_kepler, rho_desired, weights, beta_e),
        method="L-BFGS-B",
        bounds=bounds
    )

    delta_omega = result.x[0]
    delta_M = result.x[1]

    rho_lvlh, state_i, deputy_kepler = relative_lvlh_from_delta_omega_M(
        mu_earth,
        base_kepler,
        delta_omega,
        delta_M
    )

    info = {
        "success": result.success,
        "message": result.message,
        "cost": result.fun,
        "delta_omega": delta_omega,
        "delta_M": delta_M,
        "rho_desired": rho_desired,
        "rho_achieved": rho_lvlh,
        "rho_error": rho_lvlh - rho_desired,
        "deputy_kepler": deputy_kepler,
    }

    return state_i, info