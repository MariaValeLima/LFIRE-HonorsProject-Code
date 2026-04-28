# Plan: 3-Satellite Straight-Line Formation (Ring-Initialized)

## Context

The full ring formation (4 deputies in the SW plane) is producing unexpected results. To isolate whether the issue lies in the ring initialization logic itself vs the cross-track (W) placement, the user wants a minimal test case: **3 satellites total** (mothership + 2 deputies), arranged in a straight line along the S (along-track) axis, but initialized using the **same LVLH-offset + SMA-correction pipeline** as the ring formation — not the ν-shift method used by `build_swarm_straightline()`.

The two deputies correspond to θ=0° and θ=180° in the SW ring — i.e., pure along-track offsets of +15 km and −15 km. This lets us compare against the ring results while eliminating the cross-track component.

---

## What to Do

### Create one new script

**File:** `tudat_code/scripts/orbitalprop_straightline_ringinit.py`

Base it on `tudat_code/scripts/OrbitalPropWithO2M.py`, with these changes:

#### 1. Satellite names — 3 sats only
```python
mothership_name = "LFIRE-0"
deputy_names    = ["LFIRE-1", "LFIRE-3"]   # θ=0° and θ=180° from the ring
sat_names       = [mothership_name] + deputy_names
```
Naming as LFIRE-1/LFIRE-3 (not 1/2) preserves correspondence to the ring formation angles so plots are directly comparable.

#### 2. Initialization — ring method, S-axis only
```python
initial_state = build_swarm_initial_state(
    mu_earth, base_kepler,
    radius=15e3,
    thetas_rad=np.deg2rad([0, 180]),   # only S+ and S− positions
    plane="SW"
)
```
This calls the existing `build_swarm_initial_state()` from `Functions/initial_conditions.py` unchanged. No modifications to library code needed.

At θ=0°:   `dr_lvlh = [0, +15000, 0]`  (S+, along-track forward)  
At θ=180°: `dr_lvlh = [0, −15000, 0]`  (S−, along-track behind)

The SMA correction (lines 52–55 of `initial_conditions.py`) is still applied to both deputies.

#### 3. Plotting — drop LFIRE-4 reference, call relative LVLH per deputy
Replace the hardcoded `plot_relative_lvlh(..., "LFIRE-4", ...)` call with one call per deputy:
```python
for dep in deputy_names:
    plot_relative_lvlh(rv, mothership_name, dep, t_hours, plane="all")
```
All other plot calls (`plot_osculating_oe`, `plot_swarm_dispersion`, `plot_neighbor_distances`, `plot_swarm_dispersion_LVLH`) work automatically since they iterate `sat_names`.

#### 4. Output path — separate file, does not overwrite ring results
```python
out_path = ROOT / "O2M_project" / "tudat_states_sl_ringinit.npz"
```

---

## Critical Files

| File | Role |
|------|------|
| `scripts/OrbitalPropWithO2M.py` | Template to copy from |
| `scripts/Functions/initial_conditions.py` | `build_swarm_initial_state()` — reused unchanged |
| `scripts/Functions/sim_setup.py` | `make_bodies`, `make_propagator_settings`, `run_simulation` — reused unchanged |
| `scripts/Functions/plotting.py` | All plot functions — reused unchanged |
| `scripts/Functions/analysis.py` | `max_pairwise_separation` — reused unchanged |

No library files need to be modified.

---

## Verification

1. Run the new script — propagation header should show 3 satellites.
2. Osculating OE plot should show LFIRE-0, LFIRE-1, LFIRE-3 with **nearly identical** semi-major axes (SMA correction applied), same inclination/RAAN.
3. Relative LVLH plot for LFIRE-1 should show the satellite oscillating in S near +15 km, with small R and W drift.
4. Relative LVLH plot for LFIRE-3 should mirror LFIRE-1 near −15 km.
5. Compare S-axis behavior here vs the same deputies in the full ring run — any anomalous oscillation in the 3-sat run confirms the issue is in the ring initialization itself, not the W-axis placement.
