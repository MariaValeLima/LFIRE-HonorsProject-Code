# formation_frames.py
import numpy as np

def unit(vec):
    n = np.linalg.norm(vec)
    if n == 0:
        raise ValueError("Zero-norm vector")
    return vec / n

def lvlh_dcm_from_rv(r_eci, v_eci):
    r = np.asarray(r_eci).reshape(3)
    v = np.asarray(v_eci).reshape(3)

    Rhat = unit(r)
    h = np.cross(r, v)
    Chat = unit(h)
    Ihat = unit(np.cross(Chat, Rhat))

    C_LVLH_to_ECI = np.column_stack((Rhat, Ihat, Chat))
    C_ECI_to_LVLH = C_LVLH_to_ECI.T

    return C_LVLH_to_ECI, C_ECI_to_LVLH

def lvlh_to_eci(C_LVLH_to_ECI, vec_lvlh):
    return C_LVLH_to_ECI @ np.asarray(vec_lvlh).reshape(3)

def eci_to_lvlh(C_ECI_to_LVLH, vec_eci):
    return C_ECI_to_LVLH @ np.asarray(vec_eci).reshape(3)

def apply_relative_state_lvlh(r0_eci, v0_eci, dr_lvlh, dv_lvlh):
    C_LVLH_to_ECI, _ = lvlh_dcm_from_rv(r0_eci, v0_eci)

    r_i = r0_eci + lvlh_to_eci(C_LVLH_to_ECI, dr_lvlh)
    v_i = v0_eci + lvlh_to_eci(C_LVLH_to_ECI, dv_lvlh)

    return r_i, v_i
