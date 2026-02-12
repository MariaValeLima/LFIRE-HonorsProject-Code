# Functions/analysis.py
import numpy as np

def specific_energy(r: np.ndarray, v: np.ndarray, mu: float):
    rnorm = np.linalg.norm(r, axis=1)
    vnorm = np.linalg.norm(v, axis=1)
    return 0.5*vnorm**2 - mu/rnorm

def ang_momentum_mag(r: np.ndarray, v: np.ndarray):
    h = np.cross(r, v)
    return np.linalg.norm(h, axis=1)

def max_pairwise_separation(rv: dict, sat_names: list[str]) -> float:
    pairwise_max = 0.0
    for i, ni in enumerate(sat_names):
        ri, _ = rv[ni]
        for j in range(i+1, len(sat_names)):
            rj, _ = rv[sat_names[j]]
            pairwise_max = max(pairwise_max, float(np.max(np.linalg.norm(ri - rj, axis=1))))
    return pairwise_max
