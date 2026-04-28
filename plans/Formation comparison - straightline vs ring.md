# Formation Comparison: Straight-Line vs Ring (SW Circular)

## 1. Formation geometry

### Straight-line (along-track)

All deputies are placed in the **S-direction** (along-track), ahead or behind the mothership in its own orbit. The formation lies entirely within LFIRE-0's orbital plane.

```
LVLH frame at t = 0:

          +S (along-track)
          ↑
    B     C     0     D     E
  −30km −15km  0km  +15km +30km

R (radial) and W (cross-track) offsets are all zero.
```

| Satellite | S offset | R offset | W offset |
|---|---|---|---|
| LFIRE-0 | 0 | 0 | 0 |
| LFIRE-B | −30 km | 0 | 0 |
| LFIRE-C | −15 km | 0 | 0 |
| LFIRE-D | +15 km | 0 | 0 |
| LFIRE-E | +30 km | 0 | 0 |

### Ring formation (SW plane)

Deputies are placed at 15 km radius on a **circle in the S–W plane** (along-track × cross-track). Each position has a mix of S and W components depending on the angle θ.

```
LVLH frame at t = 0:

        W (cross-track)
        ↑
   2(+W)●
        |
4(−W)●──0──●1(+S)    → S (along-track)
        |
   3(−S)●
```

| Satellite | θ | S offset | W offset |
|---|---|---|---|
| LFIRE-0 | — | 0 | 0 |
| LFIRE-1 | 0° | +15 km | 0 |
| LFIRE-2 | 90° | 0 | +15 km |
| LFIRE-3 | 180° | −15 km | 0 |
| LFIRE-4 | 270° | 0 | −15 km |

---

## 2. Orbital mechanics

### Straight-line

Because all satellites are displaced only in the S-direction, they all share **exactly the same orbital plane** as LFIRE-0 (same inclination *i* and RAAN *Ω*). An S-offset is simply a phase difference along the orbit — the satellite is a few thousandths of a degree ahead or behind in true anomaly.

- All 5 satellites are on the **same Keplerian ellipse**.
- Their separation in the along-track direction stays constant in the two-body problem.
- Any drift observed in simulation is caused by **differential perturbations** (J₂, drag), not by the initial conditions. These perturbations are very small over short timescales (< 1 km over 1.5 hours at this altitude).

### Ring formation (SW plane)

- **S-offset deputies (LFIRE-1, LFIRE-3):** Also remain in LFIRE-0's orbital plane (same *i* and *Ω*), but the initialization method matters significantly (see §3).
- **W-offset deputies (LFIRE-2, LFIRE-4):** A cross-track displacement rotates the angular momentum vector **out of LFIRE-0's orbital plane**. This tilts the inclination and RAAN by a small but non-negligible amount:

| Deputy | Δi | Effect |
|---|---|---|
| LFIRE-2 (+W) | +0.12° | Orbital plane tilted one way |
| LFIRE-4 (−W) | −0.12° | Orbital plane tilted opposite way |

Two orbital planes intersect at exactly two points (the **relative nodes**). Each time a W-offset deputy passes through a relative node, its distance to LFIRE-0 drops to near zero. This oscillation is **inherent to out-of-plane offsets** and cannot be eliminated without either keeping all satellites in the same plane or using a Projected Circular Orbit (PCO) design.

| Deputy | Distance to node at t = 0 | First close approach |
|---|---|---|
| LFIRE-2 (+W) | 270° of arc away | ~75 min |
| LFIRE-4 (−W) | 90° of arc away | ~25 min |

---

## 3. Initialization method

### Straight-line: true-anomaly shift (ν-shift)

Each deputy is initialized by copying the mothership's Keplerian elements and adjusting only the true anomaly:

```python
δν = δs / a        # arc-length to angle (first-order approximation, radians)

kepler_i = dict(**base_kepler)
kepler_i['true_anomaly'] = base_kepler['true_anomaly'] + δν
state_i = keplerian_to_cartesian_elementwise(mu, **kepler_i)
```

All 6 Keplerian elements except ν match the mothership exactly. No velocity correction and no SMA correction are needed.

**Function:** `build_swarm_straightline` in `Functions/initial_conditions.py`

### Ring formation: LVLH position offset + SMA correction

Each deputy is initialized by:
1. Computing a position offset in the LVLH frame and converting it to ECI.
2. Adding the ECI velocity of the mothership (plus a frame-rotation velocity correction for S-offset deputies).
3. Correcting the semi-major axis via a Keplerian round-trip to ensure all satellites share the same orbital period.

```python
# Step 1 – position offset in LVLH → ECI
dr_lvlh = [0, radius*cos(θ), radius*sin(θ)]   # SW plane
r_i, v_i = apply_relative_state_lvlh(r0, v0, dr_lvlh, dv_lvlh)

# Step 2 – SMA correction: override a, convert back
oe_i    = cartesian_to_keplerian(hstack(r_i, v_i), mu)
oe_i[0] = base_kepler['semi_major_axis']
state_i = keplerian_to_cartesian(oe_i, mu)
```

The frame-rotation velocity correction in `apply_relative_state_lvlh` is:

```python
v_frame_correction = ω × dr_eci        (ω = h / |r|²)
```

This adds a radial velocity component for S-offset deputies to prevent eccentricity vector misalignment (see §4).

**Function:** `build_swarm_initial_state` in `Functions/initial_conditions.py`

---

## 4. Stability analysis

| Property | Straight-line | Ring (SW) |
|---|---|---|
| Orbital planes | All identical | S-deputies: same plane; W-deputies: ±0.12° tilted |
| Long-term separation drift | Near-zero (perturbation-driven only) | S-deputies: bounded oscillation; W-deputies: periodic approach to 0 |
| Oscillation amplitude | < 1 km over 1.5 h | S-deputies: ~7 km; W-deputies: up to ±15 km |
| SMA correction needed | No | Yes (to equalize orbital periods) |
| Frame-rotation correction needed | No | Yes for S-deputies (prevents eccentricity vector rotation) |
| Sensitive to initialization errors | No | Yes (removing frame correction causes ~30 km drift for S-deputies) |

---

## 5. What happens if the frame-rotation correction is removed (ring formation only)

For S-offset deputies in the ring formation, if `v_frame_correction = ω × dr_eci` is removed:

- The deputy's velocity is set to exactly v₀ (mothership velocity).
- Because the position is displaced tangentially by 15 km, the eccentricity vector of the deputy's orbit is **rotated ~29°** relative to the mothership's.
- The SMA correction cannot fix this — it only adjusts the semi-major axis (which was already nearly correct).
- Result: two orbits with matching SMA but argument of periapsis differing by ~29° produce relative radial oscillations of amplitude ~31 km.

The straight-line formation is immune to this issue because the ν-shift sets all orbital elements identically — there is no eccentricity vector mismatch by construction.

---

## 6. Code file comparison

| Aspect | `OrbitalPropWithO2M.py` | `orbitalprop_straightline.py` |
|---|---|---|
| Formation function | `build_swarm_initial_state` | `build_swarm_straightline` |
| Satellite names | LFIRE-0…4 | LFIRE-0, LFIRE-B…E |
| Formation shape | Circle (r = 15 km) in SW plane | Line at ±15, ±30 km in S |
| Initialization | LVLH offset + SMA correction | ν-shift (Keplerian phase shift) |
| Simulation duration | 1.5 h | 1.5 h |
| Save file | `tudat_states.npz` | `tudat_states_straightline.npz` |
| All other functions | — | Identical (reused unchanged) |

---

## 7. When to use each formation

| Scenario | Recommended formation |
|---|---|
| Testing initialization methods; isolating perturbation effects | Straight-line (clean baseline) |
| Distributed sensing across the orbital track | Straight-line |
| Cross-track / out-of-plane coverage (different ground tracks) | Ring (SW) |
| Interferometry requiring cross-track baselines | Ring (SW) |
| Long-duration (multi-orbit) stable formation | Straight-line, or PCO-designed ring |
