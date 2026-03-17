# Functions/postprocess.py
import numpy as np

def extract_time_arrays(states_array: np.ndarray):
    t = states_array[:, 0]
    t_hours = (t - t[0]) / 3600.0
    return t, t_hours

def extract_rv(states_array: np.ndarray, sat_names: list[str]):
    rv = {}
    for i, name in enumerate(sat_names):
        c = 1 + 6*i
        r = states_array[:, c:c+3]
        v = states_array[:, c+3:c+6]
        rv[name] = (r, v)
    return rv

def rv_to_kepler(r, v, mu=3.986004418e14):
    """Convert ECI state vector to Keplerian orbital elements."""
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    
    e_vec = np.cross(v, h) / mu - r / r_norm
    e = np.linalg.norm(e_vec)
    
    a = 1 / (2/r_norm - v_norm**2/mu)
    i = np.arccos(h[2] / h_norm)
    
    n = np.array([0, 0, 1])
    node = np.cross(n, h)
    node_norm = np.linalg.norm(node)
    
    Omega = np.arccos(node[0] / node_norm)
    if node[1] < 0:
        Omega = 2*np.pi - Omega
    
    omega = np.arccos(np.dot(node, e_vec) / (node_norm * e))
    if e_vec[2] < 0:
        omega = 2*np.pi - omega
    
    nu = np.arccos(np.dot(e_vec, r) / (e * r_norm))
    if np.dot(r, v) < 0:
        nu = 2*np.pi - nu
    
    return np.array([a, e, i, Omega, omega, nu])