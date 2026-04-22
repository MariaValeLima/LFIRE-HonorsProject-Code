# Functions/analysis.py
import numpy as np
from Functions.formation_frames import lvlh_dcm_from_rv, eci_to_lvlh

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

def swarm_dispersion(rv, sat_names, mothership_name):
    r0, _ = rv[mothership_name]
    deputies = [n for n in sat_names if n != mothership_name]
    dispersion = np.zeros(r0.shape[0])
    for name in deputies:
        r_i, _ = rv[name]
        dispersion += np.sum((r_i - r0)**2, axis=1)
    return dispersion

def satellite_distances_from_mothership(rv, sat_names, mothership_name):
    """Return {deputy_name: ndarray(N)} of distance [m] to mothership at each timestep."""
    r0, _ = rv[mothership_name]
    deputies = [n for n in sat_names if n != mothership_name]
    return {name: np.linalg.norm(rv[name][0] - r0, axis=1) for name in deputies}

def swarm_dispersion_lvlh(rv, sat_names, mothership_name):
    r0, v0 = rv[mothership_name]
    deputies = [n for n in sat_names if n != mothership_name]
    dispersion = np.zeros(r0.shape[0])
    for name in deputies:
        r_i, _ = rv[name]
        for k in range(r0.shape[0]):
            _, C = lvlh_dcm_from_rv(r0[k], v0[k])
            dr_lvlh = eci_to_lvlh(C, r_i[k] - r0[k])
            dispersion[k] += np.dot(dr_lvlh, dr_lvlh)
    return dispersion