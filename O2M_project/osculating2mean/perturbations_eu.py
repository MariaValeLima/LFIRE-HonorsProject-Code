# perturbations_eu.py
import numpy as np


def eckstein_ustinov_perturbations(OEMean) -> np.ndarray:
    """
    Python translation of EcksteinUstinovPerturbations.m

    Non-singular elements (same order as MATLAB):
      OEMean = [a, lambda, ex, ey, i, Omega]
        a      : semi-major axis (m)
        lambda : mean anomaly + argument of perigee (rad)
        ex     : e*cos(omega)
        ey     : e*sin(omega)
        i      : inclination (rad)
        Omega  : RAAN (rad)

    Returns:
      EUPert: np.ndarray shape (6,)
        [da, dlambda, dex, dey, di, dOmega]
    """
    OEMean = np.asarray(OEMean, dtype=float).reshape(-1)
    if OEMean.size != 6:
        raise ValueError("OEMean must have 6 elements: [a, lambda, ex, ey, i, Omega]")

    # Constants (same as MATLAB)
    # mu is defined in MATLAB but not actually used in this function; we omit it here.
    RE = 6378.137e3  # m
    J2 = 1082.6267e-6

    # Process input parameters (MATLAB notation)
    a0 = OEMean[0]
    lambda_0 = OEMean[1]
    l0 = OEMean[2]      # ex
    h0 = OEMean[3]      # ey
    i0 = OEMean[4]
    Omega_0 = OEMean[5]  # not used in equations, but part of element set

    e0 = np.sqrt(l0**2 + h0**2)  # defined in MATLAB, not used later (kept for parity)

    # Compute parameters
    G2 = -J2 * (RE / a0) ** 2
    beta_0 = np.sin(i0)
    xi_0 = np.cos(i0)
    lambda_star = 1.0 - (3.0 / 2.0) * G2 * (3.0 - 4.0 * beta_0)

    # --- Eckstein-Ustinov perturbations (direct translation) ---

    da = (
        -(3.0 / 2.0) * (a0 / lambda_star) * G2 * (
            (2.0 - (7.0 / 2.0) * beta_0**2) * l0 * np.cos(lambda_0)
            + (2.0 - (5.0 / 2.0) * beta_0**2) * h0 * np.sin(lambda_0)
            + beta_0**2 * np.cos(2.0 * lambda_0)
            + (7.0 / 2.0) * beta_0**2 * (l0 * np.cos(3.0 * lambda_0) + h0 * np.sin(3.0 * lambda_0))
        )
        + (3.0 / 4.0) * a0 * G2**2 * beta_0**2 * (
            7.0 * (2.0 - 3.0 * beta_0**2) * np.cos(2.0 * lambda_0)
            + beta_0**2 * np.cos(4.0 * lambda_0)
        )
    )

    dh = (
        -(3.0 / (2.0 * lambda_star)) * G2 * (
            (1.0 - (7.0 / 4.0) * beta_0**2) * np.sin(lambda_0)
            + (1.0 - 3.0 * beta_0**2) * l0 * np.sin(2.0 * lambda_0)
            + (-(3.0 / 2.0) + 2.0 * beta_0**2) * h0 * np.cos(2.0 * lambda_0)
            + (7.0 / 12.0) * beta_0**2 * np.sin(3.0 * lambda_0)
            + (17.0 / 8.0) * beta_0**2 * (l0 * np.sin(4.0 * lambda_0) - h0 * np.cos(4.0 * lambda_0))
        )
    )

    dl = (
        -(3.0 / (2.0 * lambda_star)) * G2 * (
            (1.0 - (5.0 / 4.0) * beta_0**2) * np.cos(lambda_0)
            + 0.5 * (3.0 - 5.0 * beta_0**2) * l0 * np.cos(2.0 * lambda_0)
            + (2.0 - (3.0 / 2.0) * beta_0**2) * h0 * np.sin(2.0 * lambda_0)
            + (7.0 / 12.0) * beta_0**2 * np.cos(3.0 * lambda_0)
            + (17.0 / 8.0) * beta_0**2 * (l0 * np.cos(4.0 * lambda_0) + h0 * np.sin(4.0 * lambda_0))
        )
    )

    di = (
        -(3.0 / (4.0 * lambda_star)) * G2 * beta_0 * xi_0 * (
            -l0 * np.cos(lambda_0)
            + h0 * np.sin(lambda_0)
            + np.cos(2.0 * lambda_0)
            + (7.0 / 3.0) * l0 * np.cos(3.0 * lambda_0)
            + (7.0 / 3.0) * h0 * np.sin(3.0 * lambda_0)
        )
    )

    dOmega = (
        (3.0 / (2.0 * lambda_star)) * G2 * xi_0 * (
            (7.0 / 2.0) * l0 * np.sin(lambda_0)
            - (5.0 / 2.0) * h0 * np.cos(lambda_0)
            - 0.5 * np.sin(2.0 * lambda_0)
            - (7.0 / 6.0) * l0 * np.sin(3.0 * lambda_0)
            + (7.0 / 6.0) * h0 * np.cos(3.0 * lambda_0)
        )
    )

    dlambda = (
        -(3.0 / (2.0 * lambda_star)) * G2 * (
            (10.0 - (119.0 / 8.0) * beta_0**2) * l0 * np.sin(lambda_0)
            + ((85.0 / 8.0) * beta_0**2 - 9.0) * h0 * np.cos(lambda_0)
            + (2.0 * beta_0**2 - 0.5) * np.sin(2.0 * lambda_0)
            + (-(7.0 / 6.0) + (119.0 / 24.0) * beta_0**2) * (l0 * np.sin(3.0 * lambda_0) - h0 * np.cos(3.0 * lambda_0))
            - (3.0 - (21.0 / 4.0) * beta_0**2) * l0 * np.sin(lambda_0)
            + (3.0 - (15.0 / 4.0) * beta_0**2) * h0 * np.cos(lambda_0)
            - (3.0 / 4.0) * beta_0**2 * np.sin(2.0 * lambda_0)
            - (21.0 / 12.0) * beta_0**2 * (l0 * np.sin(3.0 * lambda_0) - h0 * np.cos(3.0 * lambda_0))
        )
    )

    # Output vector in the same element order
    return np.array([da, dlambda, dl, dh, di, dOmega], dtype=float)