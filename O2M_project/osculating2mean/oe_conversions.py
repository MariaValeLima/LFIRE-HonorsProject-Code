# oe_conversions.py
import numpy as np
from .kepler import kep_eqtn_E
from .perturbations_eu import eckstein_ustinov_perturbations 

TWOPI = 2.0 * np.pi
MU_EARTH = 3.986004418e14  # (m^3/s^2) same as MATLAB


def _wrap_0_2pi_matlab(angle: float) -> float:
    """
    Wrap angle to [0, 2*pi] using the SAME floor/ceil logic as the MATLAB code.
    (Not just angle % (2*pi), to stay identical.)
    """
    a = float(angle)
    if a > TWOPI:
        a = a - np.floor(a / TWOPI) * TWOPI
    elif a < 0.0:
        a = a + np.ceil(-a / TWOPI) * TWOPI
    return a




def oe_osc_to_rv(OE, max_it: int = 100, epsl: float = 1e-5, mu: float = MU_EARTH) -> np.ndarray:
    """
    Direct translation of OEOsc2rv.m

    Input OE (MATLAB order):
      OE = [a, u, ex, ey, i, Omega]
        a: semi-major axis (m)
        u: mean anomaly + argument of perigee (rad)
        ex, ey: eccentricity vector components
        i: inclination (rad)
        Omega: RAAN (rad)

    Output:
      x: shape (6,) -> [rx, ry, rz, vx, vy, vz] in ECI
    """
    OE = np.asarray(OE, dtype=float).reshape(-1)
    if OE.size != 6:
        raise ValueError("OE must have 6 elements: [a, u, ex, ey, i, Omega]")

    a = OE[0]
    u = OE[1]
    e = np.sqrt(OE[2] ** 2 + OE[3] ** 2)
    inc = OE[4]
    Omega = OE[5]

    p = a * (1.0 - e ** 2)

    # Vallado Algorithm 6 (near-circular handling)
    if e < 1e-5:
        omega = 0.0
        nu = u
    else:
        omega = np.arctan2(OE[3], OE[2])
        M = u - omega

        # Fix angle difference (same logic as MATLAB)
        if M < -np.pi:
            M = M + (np.floor(abs(M - np.pi) / (2 * np.pi))) * 2 * np.pi
        elif M > np.pi:
            M = M - np.floor((M + np.pi) / (2 * np.pi)) * 2 * np.pi

        E = kep_eqtn_E(M, e, max_it=max_it, epsl=epsl)
        nu = 2.0 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2.0))

    rPQW = np.array([
        p * np.cos(nu) / (1 + e * np.cos(nu)),
        p * np.sin(nu) / (1 + e * np.cos(nu)),
        0.0
    ])

    vPQW = np.array([
        -np.sqrt(mu / p) * np.sin(nu),
        np.sqrt(mu / p) * (e + np.cos(nu)),
        0.0
    ])

    # Rotation matrix T (same entries as MATLAB)
    T = np.zeros((3, 3))
    T[0, 0] = np.cos(Omega) * np.cos(omega) - np.sin(Omega) * np.sin(omega) * np.cos(inc)
    T[0, 1] = -np.cos(Omega) * np.sin(omega) - np.sin(Omega) * np.cos(omega) * np.cos(inc)
    T[0, 2] = np.sin(Omega) * np.sin(inc)

    T[1, 0] = np.sin(Omega) * np.cos(omega) + np.cos(Omega) * np.sin(omega) * np.cos(inc)
    T[1, 1] = -np.sin(Omega) * np.sin(omega) + np.cos(Omega) * np.cos(omega) * np.cos(inc)
    T[1, 2] = -np.cos(Omega) * np.sin(inc)

    T[2, 0] = np.sin(omega) * np.sin(inc)
    T[2, 1] = np.cos(omega) * np.sin(inc)
    T[2, 2] = np.cos(inc)

    rECI = T @ rPQW
    vECI = T @ vPQW

    return np.concatenate([rECI, vECI])

def rv_to_oe_osc(x, mu: float = MU_EARTH) -> np.ndarray:
    """
    Python translation of rv2OEOsc.m

    Input:
      x: array-like shape (6,) or (6,1)
         [rx, ry, rz, vx, vy, vz] in ECI

    Output:
      OE: np.ndarray shape (6,)
          [a, u, ex, ey, i, Omega]
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 6:
        raise ValueError("x must have 6 elements: [rx, ry, rz, vx, vy, vz]")

    r0 = x[0:3]
    v0 = x[3:6]

    r = np.linalg.norm(r0)
    v = np.linalg.norm(v0)

    # Semi-major axis a
    a = -(mu / 2.0) / ((v**2) / 2.0 - mu / r)

    # Eccentricity vector
    e_vec = ((v**2 - mu / r) * r0 - (r0 @ v0) * v0) / mu
    e = np.linalg.norm(e_vec)

    # Angular momentum vector
    h = np.cross(r0, v0)
    h_norm = np.linalg.norm(h)

    # Line of nodes unit vector n (Vallado-style)
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        # Equatorial orbit: Omega undefined. MATLAB code would divide by 0 here.
        # We choose Omega=0 and define n along x for continuity.
        n = np.array([1.0, 0.0, 0.0])
        n_norm = 1.0
    n = n / n_norm

    n_cross_h = np.cross(h / h_norm, n)
    n_cross_h = n_cross_h / np.linalg.norm(n_cross_h)

    # Near-circular threshold: e < 1e3*eps (matches MATLAB)
    if e < 1e3 * np.finfo(float).eps:
        omega = 0.0
        # nu = acos(n'*(rhat))
        rhat = r0 / r
        nu = np.arccos(np.clip(n @ rhat, -1.0, 1.0))

        if rhat @ np.cross(h / h_norm, n) < 0.0:
            nu = 2.0 * np.pi - nu

        E = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(nu / 2.0))
        M = E - e * np.sin(E)
        u = M + omega
    else:
        # omega = acos(n'*(e_vec/e))
        omega = np.arccos(np.clip(n @ (e_vec / e), -1.0, 1.0))
        if e_vec[2] < 0.0:
            omega = 2.0 * np.pi - omega

        # nu = acos((e_vec'/e)*(rhat))
        rhat = r0 / r
        nu = np.arccos(np.clip((e_vec / e) @ rhat, -1.0, 1.0))
        if (r0 @ v0) < 0.0:
            nu = 2.0 * np.pi - nu

        E = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(nu / 2.0))
        M = E - e * np.sin(E)
        u = M + omega

    # Wrap u to [0, 2pi)
    twopi = 2.0 * np.pi
    if u > twopi:
        u = u - np.floor(u / twopi) * twopi
    elif u < 0.0:
        u = u + np.ceil(-u / twopi) * twopi

    ex = n @ e_vec
    ey = n_cross_h @ e_vec
    inc = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))

    # Omega = acos(n(1)), adjust if n(2) < 0
    Omega = np.arccos(np.clip(n[0], -1.0, 1.0))
    if n[1] < 0.0:
        Omega = twopi - Omega

    # Wrap Omega to [0, 2pi)
    if Omega > twopi:
        Omega = Omega - np.floor(Omega / twopi) * twopi
    elif Omega < 0.0:
        Omega = Omega + np.ceil(-Omega / twopi) * twopi

    return np.array([a, u, ex, ey, inc, Omega], dtype=float)

def oe_osc_to_oe_mean_eu(OEosc, MaxIt: int = 100, epslPos: float = 1e-1, epslVel: float = 1e-4):
    """
    Python translation of OEOsc2OEMeanEU.m

    Input:
      OEosc: [a, u, ex, ey, i, Omega]  (osculating)
      MaxIt: max iterations (default 100)
      epslPos: position tolerance in meters (default 1e-1)
      epslVel: velocity tolerance in m/s (default 1e-4)

    Output:
      OEMean: [a, u, ex, ey, i, Omega] (mean, Eckstein-Ustinov)
    """
    OEosc = np.asarray(OEosc, dtype=float).reshape(-1)
    if OEosc.size != 6:
        raise ValueError("OEosc must have 6 elements: [a, u, ex, ey, i, Omega]")

    # Compute position-velocity vector
    x = oe_osc_to_rv(OEosc)

    # 1.1 Initialization: Mean elements are equal to osculating elements
    OEMean = OEosc.copy()

    # status = zeros(2,MaxIt) in MATLAB (optional to keep)
    status = np.zeros((2, MaxIt), dtype=float)

    # 1.2 Iterate the Eckstein-Ustinov corrections
    for i in range(1, MaxIt + 1):  # MATLAB is 1..MaxIt
        # Compute perturbation
        EUPerturbation = eckstein_ustinov_perturbations(OEMean)

        # Update and fix angle ranges (osculating iteration)
        OEoscIt = OEMean + EUPerturbation
        OEoscIt[1] = _wrap_0_2pi_matlab(OEoscIt[1])  # u
        OEoscIt[5] = _wrap_0_2pi_matlab(OEoscIt[5])  # Omega

        # Update and fix angle ranges (mean update)
        OEMean = OEosc - EUPerturbation
        OEMean[1] = _wrap_0_2pi_matlab(OEMean[1])    # u
        OEMean[5] = _wrap_0_2pi_matlab(OEMean[5])    # Omega

        # Check stopping criterion
        xIt = oe_osc_to_rv(OEoscIt)
        status[0, i - 1] = np.linalg.norm(xIt[0:3] - x[0:3])
        status[1, i - 1] = np.linalg.norm(xIt[3:6] - x[3:6])

        if (np.linalg.norm(xIt[0:3] - x[0:3]) < epslPos) and (np.linalg.norm(xIt[3:6] - x[3:6]) < epslVel):
            break

        if i == MaxIt:
            raise RuntimeError("Maximum number of iterations reached for rv2OEMeanEcksteinUstinov.")

    return OEMean  # (optionally also return status if you want)


def oe_mean_eu_to_oe_osc(OEMean) -> np.ndarray:
    """
    Python translation of OEMeanEU2OEOsc.m

    Input:
      OEMean: [a, u, ex, ey, i, Omega] (mean, EU)

    Output:
      OEosc:  [a, u, ex, ey, i, Omega] (osculating)
    """
    OEMean = np.asarray(OEMean, dtype=float).reshape(-1)
    if OEMean.size != 6:
        raise ValueError("OEMean must have 6 elements: [a, u, ex, ey, i, Omega]")

    EUp = eckstein_ustinov_perturbations(OEMean)
    OEosc = OEMean + EUp

    # Fix angles (same indices as MATLAB: OE(2) and OE(6) -> Python [1] and [5])
    OEosc[1] = _wrap_0_2pi_matlab(OEosc[1])  # u
    OEosc[5] = _wrap_0_2pi_matlab(OEosc[5])  # Omega

    return OEosc


def oe_osc_to_oe_mean_euk(
    t_tdb: float,
    OEosc,
    degree: int,
    egm96_path: str,
    MaxIt: int = 100,
    epslPos: float = 1e-1,
    epslVel: float = 1e-4,
) -> np.ndarray:
    """
    Python translation of OEOsc2OEMeanEUK.m

    Converts osculating orbital elements to mean elements using combined
    Eckstein-Ustinov (J2) and Kaula geopotential theory.

    Input:
      t_tdb    : barycentric dynamical time since J2000 (s)
      OEosc    : [a, u, ex, ey, i, Omega] osculating elements
      degree   : maximum degree of the spherical harmonics geopotential model
      egm96_path: path to the EGM96 data file (for the Fortran Kaula module)
      MaxIt    : max iterations for the EU step (default 100)
      epslPos  : position tolerance in metres for EU step (default 0.1 m)
      epslVel  : velocity tolerance in m/s for EU step (default 1e-4 m/s)

    Output:
      OEMean: [a, u, ex, ey, i, Omega] mean elements (EU + Kaula)
    """
    from .perturbations_kaula import kaula_geopotential_perturbations

    OEosc = np.asarray(OEosc, dtype=float).reshape(-1)
    if OEosc.size != 6:
        raise ValueError("OEosc must have 6 elements: [a, u, ex, ey, i, Omega]")

    # Step 1: get J2 mean elements via Eckstein-Ustinov iteration
    OEMean_EU = oe_osc_to_oe_mean_eu(OEosc, MaxIt=MaxIt, epslPos=epslPos, epslVel=epslVel)

    # Step 2: compute higher-order geopotential perturbations at EU mean elements
    # dOE = [da, de, di, dOmega, dw, dM]
    dOE = kaula_geopotential_perturbations(t_tdb, OEMean_EU, degree, egm96_path)
    da     = dOE[0]
    di     = dOE[2]
    dOmega = dOE[3]
    dw     = dOE[4]
    dM     = dOE[5]

    # Perturbation in the non-singular argument u = M + omega
    du = dM + dw

    # Step 3: correct EU mean elements with Kaula perturbations.
    # ex/ey are kept from EU result (Kaula eccentricity correction has a
    # singularity for near-circular orbits — matches OEOsc2OEMeanEUK.m)
    OEMean = OEosc - np.array([da, du, 0.0, 0.0, di, dOmega])
    OEMean[2:4] = OEMean_EU[2:4]

    # Fix angle ranges
    OEMean[1] = _wrap_0_2pi_matlab(OEMean[1])   # u
    OEMean[5] = _wrap_0_2pi_matlab(OEMean[5])   # Omega

    return OEMean


def oe_mean_euk_to_oe_osc(
    t_tdb: float,
    OEMean,
    degree: int,
    egm96_path: str,
) -> np.ndarray:
    """
    Python translation of OEMeanEUK2OEOsc.m

    Converts mean orbital elements (EU + Kaula) to osculating elements.

    Input:
      t_tdb    : barycentric dynamical time since J2000 (s)
      OEMean   : [a, u, ex, ey, i, Omega] mean elements
      degree   : maximum degree of the spherical harmonics geopotential model
      egm96_path: path to the EGM96 data file (for the Fortran Kaula module)

    Output:
      OEosc: [a, u, ex, ey, i, Omega] osculating elements
    """
    from .perturbations_kaula import kaula_geopotential_perturbations

    OEMean = np.asarray(OEMean, dtype=float).reshape(-1)
    if OEMean.size != 6:
        raise ValueError("OEMean must have 6 elements: [a, u, ex, ey, i, Omega]")

    # Step 1: J2 (Eckstein-Ustinov) perturbations
    EUPerturbation = eckstein_ustinov_perturbations(OEMean)

    # Step 2: higher-order geopotential (Kaula) perturbations
    # dOE = [da, de, di, dOmega, dw, dM]
    dOE = kaula_geopotential_perturbations(t_tdb, OEMean, degree, egm96_path)
    da     = dOE[0]
    di     = dOE[2]
    dOmega = dOE[3]
    dw     = dOE[4]
    dM     = dOE[5]

    # Perturbation in the non-singular argument u = M + omega
    du = dM + dw

    # Step 3: apply perturbations to get osculating elements.
    # ex/ey use only the EU perturbation (Kaula has a singularity for near-circular orbits)
    OEosc = OEMean + np.array([da, du, 0.0, 0.0, di, dOmega])
    OEosc[2:4] = OEMean[2:4] + EUPerturbation[2:4]

    # Fix angle ranges
    OEosc[1] = _wrap_0_2pi_matlab(OEosc[1])   # u
    OEosc[5] = _wrap_0_2pi_matlab(OEosc[5])   # Omega

    return OEosc