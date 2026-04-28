# Plan: Unperturbed Simulation + Orbital Closure Check

## Context

Two debugging tools to diagnose the anomalous results in the ring/straight-line formation:
1. **Remove all perturbations** → run pure two-body dynamics so we know exactly what the "ideal" behavior should look like and can confirm whether the initialization is correct.
2. **Orbital closure check** → verify that each satellite returns to (very nearly) its initial state after exactly one orbital period. A non-zero delta reveals either a perturbation effect or an initialization problem.

---

## Part 1 — Remove All Perturbations

### What to change

**File:** `scripts/Functions/sim_setup.py`

Add a new function `make_propagator_settings_unperturbed()` alongside the existing one. Do **not** modify the existing function — keep it for the full perturbed runs.

```python
def make_propagator_settings_unperturbed(bodies, sat_names, initial_state, t0, tf, dt):
    """Pure two-body dynamics: point-mass Earth only, no drag, no third bodies."""
    bodies_to_propagate = sat_names
    central_bodies = ["Earth"] * len(bodies_to_propagate)

    acc_settings_each_sat = {
        "Earth": [
            propagation_setup.acceleration.point_mass_gravity(),  # replaces spherical harmonics
            # aerodynamic() removed
        ],
        # Sun, Moon, Mars, Venus removed
    }
    acceleration_settings = {name: acc_settings_each_sat for name in bodies_to_propagate}

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    termination_settings = propagation_setup.propagator.time_termination(tf)
    integrator_settings  = propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=dt,
        coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    return propagation_setup.propagator.translational(
        central_bodies, acceleration_models, bodies_to_propagate,
        initial_state, t0, integrator_settings, termination_settings
    )
```

Also update `make_bodies()` to not require `"Sun", "Moon", "Mars", "Venus"` when running unperturbed. Easiest fix: pass `bodies_to_create` as a parameter with a default.

```python
def make_bodies(sat_names, bodies_to_create=None):
    if bodies_to_create is None:
        bodies_to_create = ["Earth", "Sun", "Moon", "Mars", "Venus"]
    ...
```

### How to use in a script

In `OrbitalPropWithO2M.py` (or the 3-sat script), swap one line:

```python
# Perturbed (existing):
propagator_settings = make_propagator_settings(...)

# Unperturbed (new):
bodies = make_bodies(sat_names, bodies_to_create=["Earth"])
propagator_settings = make_propagator_settings_unperturbed(...)
```

### What you should see (unperturbed, two-body)

| Plot | Expected behavior |
|------|-------------------|
| **Osculating OE — a, e, i, Ω, ω** | Perfectly flat lines. Zero drift, zero oscillation. |
| **Osculating OE — ν** | Smooth monotonically increasing curve, faster near perigee. |
| **Relative LVLH (S-axis deputies)** | Small bounded oscillation at exactly the orbital period (~6085 s). Amplitude driven by orbit eccentricity (e≈0.004 → ~few km in R, ~15 km in S). No secular drift. |
| **Swarm dispersion** | Bounded, periodic. Never grows or shrinks secularly. |
| **Neighbor distances** | Periodic at ~101 min. Constant envelope — no growth. |

If **orbital elements are not flat** in the unperturbed run, the initialization or post-processing is wrong — not the perturbation model.

If **LVLH motion drifts secularly** in the unperturbed run, the SMA correction did not equalize the periods, which is the root bug.

---

## Part 2 — Orbital Closure Check

### Concept

For a closed Keplerian orbit, satellite state at `t = T_orb` must equal the state at `t = 0`. The closure error (delta) per orbit is:

```
Δr_n = r(n·T_orb) − r(0)      [m]
Δv_n = v(n·T_orb) − v(0)      [m/s]
```

For unperturbed two-body motion: Δr_n ≈ 0 (numerical error only, ~cm).  
For perturbed motion: Δr_n grows each orbit due to J2, drag, third-body effects.

### What to add

**File:** `scripts/Functions/analysis.py`

Add a new function:

```python
def orbital_closure_deltas(r, v, t, mu=3.986004418e14):
    """
    Compute the per-orbit closure error for one satellite.

    Parameters
    ----------
    r : (N, 3) array  — ECI position in metres
    v : (N, 3) array  — ECI velocity in m/s
    t : (N,)   array  — time in seconds (absolute or relative, zero-indexed)
    mu : float        — gravitational parameter

    Returns
    -------
    orbit_times : list of float   — epoch of each orbit start (s)
    delta_r_km  : list of float   — |Δr| at each orbit boundary (km)
    delta_v_mps : list of float   — |Δv| at each orbit boundary (m/s)
    """
    import numpy as np

    a0 = 1.0 / (2.0 / np.linalg.norm(r[0]) - np.dot(v[0], v[0]) / mu)
    T_orb = 2 * np.pi * np.sqrt(a0**3 / mu)   # orbital period in seconds

    dt = float(t[1] - t[0])
    steps_per_orbit = int(round(T_orb / dt))

    orbit_times, delta_r_km, delta_v_mps = [], [], []

    n = 1
    while n * steps_per_orbit < len(t):
        idx = n * steps_per_orbit
        delta_r = np.linalg.norm(r[idx] - r[0]) / 1e3   # km
        delta_v = np.linalg.norm(v[idx] - v[0])          # m/s
        orbit_times.append(t[idx] - t[0])
        delta_r_km.append(delta_r)
        delta_v_mps.append(delta_v)
        n += 1

    return orbit_times, delta_r_km, delta_v_mps
```

### How to call it in the simulation script

At the end of any simulation script:

```python
from Functions.analysis import orbital_closure_deltas
import matplotlib.pyplot as plt

for name in sat_names:
    r_arr, v_arr = rv[name]
    times, dr_km, dv_mps = orbital_closure_deltas(r_arr, v_arr, t)
    orbit_numbers = list(range(1, len(times) + 1))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle(f"Orbital closure error — {name}")
    axes[0].plot(orbit_numbers, dr_km, 'o-')
    axes[0].set_ylabel("|Δr| (km)")
    axes[1].plot(orbit_numbers, dv_mps, 'o-')
    axes[1].set_ylabel("|Δv| (m/s)")
    axes[1].set_xlabel("Orbit number")
    plt.tight_layout()
    plt.show()
```

### What you should see

| Condition | |Δr| per orbit |
|-----------|--------------|
| **Unperturbed two-body** | < 1 m (numerical integration error only) |
| **Perturbed (full model)** | ~0.1–1 km per orbit, growing over time |
| **Initialization bug (mismatched SMA)** | Large Δr on orbit 1 itself (~km), not growing — steady offset |

A closure error that is **large on orbit 1 and doesn't grow** points to an initialization problem (wrong period), not perturbations.

A closure error that **grows linearly** per orbit points to a secular perturbation (drag, J2 RAAN drift).

---

## Files to Modify / Create

| Action | File |
|--------|------|
| **Add** `make_propagator_settings_unperturbed()` | `scripts/Functions/sim_setup.py` |
| **Modify** `make_bodies()` to accept `bodies_to_create` param | `scripts/Functions/sim_setup.py` |
| **Add** `orbital_closure_deltas()` | `scripts/Functions/analysis.py` |
| **Create** new runner script using unperturbed settings | `scripts/orbitalprop_unperturbed.py` (copy from `OrbitalPropWithO2M.py`, swap function calls) |

---

## Recommended Debugging Order

1. Run the unperturbed simulation. Check that OEs are flat. If not, the bug is in initialization.
2. Check the closure delta for the mothership (LFIRE-0) first. It should be near zero for unperturbed.
3. Check closure deltas for LFIRE-1 vs LFIRE-3 in unperturbed. If LFIRE-3 has a larger Δr, the SMA correction is not working symmetrically.
4. Then run the same closure check on the perturbed simulation to see how much each perturbation adds per orbit.
