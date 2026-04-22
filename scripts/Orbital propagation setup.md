# Orbital Propagation Setup

## Overview

Propagates a 5-satellite swarm (1 mothership + 4 deputies) in a Sun-synchronous Low Earth Orbit for 60 hours using TUDAT. All satellites share the same base orbit; deputies are offset in the LVLH frame at 15 km radius.

---

## Mothership — LFIRE-0

| Element | Value |
|---|---|
| Semi-major axis | 6 992 762.21 m (6992.76 km) |
| Eccentricity | 4.033 × 10⁻³ (near-circular) |
| Inclination | 1.7107 rad = 98.04° (Sun-synchronous / near-polar) |
| Argument of periapsis ω | 1.3123 rad = 75.18° |
| RAAN Ω | 0.3830 rad = 21.94° |
| True anomaly ν | 3.0702 rad = 175.9° (near apogee) |

Orbital period: ~6085 s (~101.4 min)  
Altitude: ~621 km

---

## Deputy Formation — LFIRE-1 to LFIRE-4

All deputies are placed on a circle of radius **15 km** in the **SW plane** (along-track × cross-track) of the LVLH frame.

| Satellite | θ | S offset (along-track) | W offset (cross-track) |
|---|---|---|---|
| LFIRE-1 | 0° | +15 km | 0 |
| LFIRE-2 | 90° | 0 | +15 km |
| LFIRE-3 | 180° | −15 km | 0 |
| LFIRE-4 | 270° | 0 | −15 km |

No initial relative velocity (`dv_lvlh = 0`).

---

## LVLH Frame Definition

Defined in `Functions/formation_frames.py`:

```
R̂ = unit(r)                     — radial outward
Ĉ = unit(r × v)                 — cross-track (= angular momentum direction)
Î = unit(Ĉ × R̂)                 — along-track (≈ velocity direction)
```

Matrix: `C_LVLH_to_ECI = [R̂ | Î | Ĉ]`

Stored as LVLH index: `[0]=R, [1]=S(I), [2]=W(C)`

---

## Initial Condition Pipeline

**File:** `Functions/initial_conditions.py` — `build_swarm_initial_state()`

1. Convert mothership Keplerian elements → ECI Cartesian `(r0, v0)` via TUDAT
2. For each deputy at angle θ:
   a. Compute LVLH offset `dr_lvlh`
   b. Rotate to ECI: `dr_eci = C_LVLH_to_ECI @ dr_lvlh`
   c. Apply frame-rotation velocity correction: `v_corr = ω × dr_eci`, where `ω = (r × v) / |r|²`
   d. Deputy state: `r_i = r0 + dr_eci`, `v_i = v0 + v_corr`
   e. **SMA correction**: convert `(r_i, v_i)` → Keplerian, set `a = a_mothership`, convert back

**Note on SMA correction:** Steps a–d produce a state with a slightly different SMA than the mothership (because `v_corr` changes orbital energy for S-direction offsets). The correction in step e forces all deputies to the same orbital period.

---

## Force Model

**File:** `Functions/sim_setup.py`

| Perturbing body / force | Model |
|---|---|
| Earth gravity | Spherical harmonics, degree 5 × order 5 (includes J2–J5, tesseral terms) |
| Atmospheric drag | Constant aerodynamic coefficients (Cd = 1.2, A = 0.1 m², m = 10 kg) |
| Sun | Point-mass gravity |
| Moon | Point-mass gravity |
| Mars | Point-mass gravity |
| Venus | Point-mass gravity |

---

## Numerical Integrator

| Setting | Value |
|---|---|
| Method | Runge-Kutta 4th order (RK4), fixed step |
| Time step | 10 seconds |
| Duration | 60 hours (216 000 steps) |

---

## Post-Processing Pipeline

**File:** `OrbitalPropWithO2M.py`

1. `extract_time_arrays` — time vector in seconds and hours
2. `extract_rv` — split 30-column state array into `(r, v)` per satellite
3. `rv_to_kepler` — convert ECI Cartesian → classical Keplerian elements at each timestep (custom, `postprocess.py`)
4. Compute osculating OEs over the full trajectory for all 5 satellites

---

## Output Plots

| Plot | Description |
|---|---|
| ECI 3D trajectories | All satellites in inertial frame |
| Osculating OEs | 6 classical elements vs time for all satellites |
| Relative LVLH trajectory | Deputy relative motion in R, S, W planes |
| Swarm dispersion — per satellite | Distance deviation from initial 15 km, one line per deputy |
| Swarm dispersion — aggregate | Sum of squared distance deviations |
| LVLH dispersion | Same metric computed in rotating LVLH frame |

---

## Known Issues (under investigation)

- **LFIRE-4 large oscillations**: LFIRE-4 (−W cross-track) shows ~15 km amplitude oscillations at the orbital period while LFIRE-1/2/3 show only a slow secular drift. Root cause under investigation — see `LFIRE-4 bug investigation plan.md`.
- **Secular drift**: All deputies drift ~8.5 km away from LFIRE-0 over 60 hours, likely from differential J2 precession (the deputies have slightly different inclinations due to W-offsets).
