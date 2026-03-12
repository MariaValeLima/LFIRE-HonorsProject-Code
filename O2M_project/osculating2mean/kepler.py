# kepler.py
import numpy as np
import warnings
TWOPI = 2.0 * np.pi

def kep_eqtn_E(M: float, e: float, max_it: int = 100, epsl: float = 1e-5) -> float:
    """
    Solve Kepler's equation for eccentric anomaly E from mean anomaly M and eccentricity e.
    Direct translation of KepEqtnE in OEOsc2rv.m (Vallado, 1997).
    """
    if ((M > -np.pi) and (M < 0.0)) or (M > np.pi):
        E_n1 = M - e
    else:
        E_n1 = M + e

    count = 0
    while True:
        E_n = E_n1
        E_n1 = E_n + (M - E_n + e * np.sin(E_n)) / (1.0 - e * np.cos(E_n))

        if abs(E_n1 - E_n) < epsl:
            return float(E_n1)

        count += 1
        if count >= max_it:
            warnings.warn("Maximum number of iterations for kep_eqtn_E reached.")
            return float(E_n1)
        


