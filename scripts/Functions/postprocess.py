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
