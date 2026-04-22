# LFIRE-4 Bug Investigation Plan

## Observed Symptom

LFIRE-4 shows large (~15 km amplitude) oscillations at the orbital period (~101 min) in the "swarm dispersion per satellite" plot. LFIRE-1, LFIRE-2, and LFIRE-3 all overlap with a smooth secular drift of ~8.5 km over 60 hours. LFIRE-4 oscillates between 0 km and −15 km deviation (i.e., its distance from LFIRE-0 periodically drops to nearly 0), while sharing the same secular trend.

## Why This Is Suspicious

LFIRE-2 and LFIRE-4 are mirror images:
- LFIRE-2 (θ=90°): `dr_lvlh = [0, 0, +15km]` — pure +W (cross-track outward)
- LFIRE-4 (θ=270°): `dr_lvlh = [0, 0, −15km]` — pure −W (cross-track inward)

Both have **zero** frame-rotation velocity correction (`v_frame_correction = ω × dr_eci = 0`, because ω ∥ dr_eci for a W-offset). Physically they should be nearly symmetric. LFIRE-2 does not oscillate; LFIRE-4 does. This asymmetry demands explanation.

---

## Candidate Causes (in order of likelihood)

---

### Cause 1 — SMA correction introduces a large eccentricity for LFIRE-4

**What could go wrong:**  
The `cartesian_to_keplerian → override a[0] → keplerian_to_cartesian` round-trip is numerically sensitive. For LFIRE-4's specific state (−W offset, ν ≈ 176°, small e), the conversion could produce a state where the velocity direction shifts significantly — giving LFIRE-4 a large eccentricity and making it oscillate in and out of LFIRE-0's vicinity.

**Diagnostic:**  
Print the Keplerian elements of all deputies immediately before and after the SMA correction in `build_swarm_initial_state`. Check `oe_i` for LFIRE-4 (θ=270°) vs LFIRE-2 (θ=90°) — specifically eccentricity `oe_i[1]`.

```python
# Add to build_swarm_initial_state after oe_i[0] = a_target:
print(f"θ={np.rad2deg(th):.0f}°  a={oe_i[0]:.2f}  e={oe_i[1]:.6f}  i={np.rad2deg(oe_i[2]):.4f}  nu={np.rad2deg(oe_i[5]):.4f}")
```

**Expected:** All deputies have e ≈ 4.03e-3, very close to the mothership.  
**Bug indicator:** LFIRE-4 has a significantly different eccentricity.

---

### Cause 2 — True anomaly (ν) quadrant ambiguity in `cartesian_to_keplerian` for LFIRE-4

**What could go wrong:**  
The mothership is at ν ≈ 176° (near apogee). The Keplerian conversion uses `arccos(dot(e_vec, r) / (e * |r|))`, which requires a quadrant check (`if dot(r, v) < 0: ν = 2π − ν`). For LFIRE-4 at `r0 − 15km*Ĉ` with `v0` (same as mothership), the radial velocity `dot(r4, v4)` might sit right at the boundary between positive and negative, causing ν to flip from ~176° to ~184°. After setting `a = a_target` and converting back, the satellite would be placed on the wrong side of its orbit.

**Diagnostic:**  
Print `np.dot(r_i, v_i)` for each deputy before the SMA correction. If LFIRE-4 has a value near zero (or opposite sign to LFIRE-2), this is the cause.

```python
print(f"θ={np.rad2deg(th):.0f}°  dot(r,v) = {np.dot(r_i, v_i):.4f}")
```

---

### Cause 3 — `keplerian_to_cartesian` element order differs from `cartesian_to_keplerian`

**What could go wrong:**  
TUDAT's array-based `cartesian_to_keplerian(state, mu)` might return elements in a different order than `keplerian_to_cartesian(oe, mu)` expects. For example, if one uses `[a, e, i, Ω, ω, ν]` and the other uses `[a, e, i, ω, Ω, ν]` (swapping ω and Ω), the round-trip would silently corrupt the orbit. Since we only modify `oe_i[0]` (a), this wouldn't matter for the round-trip — UNLESS ω or Ω differ significantly between LFIRE-4 and the others, causing a different sensitivity to the ordering bug.

**Diagnostic:**  
Compare the output of `keplerian_to_cartesian_elementwise(**base_kepler)` against `keplerian_to_cartesian(cartesian_to_keplerian(state0, mu), mu)` for the mothership. If they match, ordering is consistent.

```python
state0_roundtrip = element_conversion.keplerian_to_cartesian(
    element_conversion.cartesian_to_keplerian(state0, mu_earth), mu_earth)
print("Mothership round-trip error:", np.max(np.abs(state0 - state0_roundtrip)))
```

---

### Cause 4 — Physical intersecting orbit geometry for −W displaced satellite

**What could go wrong:**  
This may not be a code bug at all. A satellite displaced purely in the W (out-of-plane) direction ends up in a slightly different orbital plane from the mothership. For a circular orbit, the two planes share two intersection points (the nodes of the relative orbit). If both satellites are propagated from a point near such a node, they will periodically pass through the intersection and the separation will oscillate between 0 and ~2 × 15 km.

For LFIRE-2 (+W) the initial position may be near the antinode (maximum separation), while for LFIRE-4 (−W) the initial position may be near the node (minimum separation) — purely due to the orbital geometry at the initial epoch (ν ≈ 176°, near apogee, i = 98°).

**Diagnostic:**  
Compute the relative inclination vector between LFIRE-4 and LFIRE-0. If the relative node angle puts the initial position near the relative equatorial crossing, LFIRE-4 will naturally oscillate toward zero separation.

Simplest check: look at the relative separation at t=0 and the first minimum — if the separation hits zero periodically, this is a physical orbit intersection, not a code bug.

---

### Cause 5 — SMA correction makes LFIRE-4's orbit worse, not better

**What could go wrong:**  
For a W-offset satellite the frame-rotation correction is exactly zero, so before the SMA fix the state already has `a ≈ a_mothership`. The SMA correction therefore finds a tiny `δa` and corrects it. But the Keplerian round-trip changes the velocity vector direction slightly (since the position also changes by ~40 m), and this tiny direction change could project significantly onto the radial direction — converting a nearly circular orbit into one with measurable eccentricity.

**Diagnostic:**  
Run the simulation twice — once with the SMA correction enabled (current code) and once with it disabled (commenting out the 4 correction lines). If LFIRE-4's oscillation disappears without the SMA correction, the fix itself is the bug.

---

### Cause 6 — `cos(270°)` / `sin(270°)` floating-point giving a small non-zero S component

**What could go wrong:**  
`np.cos(np.deg2rad(270))` is not exactly 0 in floating point — it equals approximately `−6.12e-17`. This gives LFIRE-4 a tiny S-direction offset (`dr_lvlh[1] ≈ −9.18e-13 m`) which is negligible. But after the SMA correction, this infinitesimal S offset might influence the Keplerian element computation in an unexpected way.

**Likelihood:** Very low — the numerical error is 13 orders of magnitude smaller than the 15 km offset.

---

### Cause 7 — `rv_to_kepler` custom function (postprocess.py) has a sign error specific to LFIRE-4's state

**What could go wrong:**  
The `rv_to_kepler` function is used to compute **post-propagation** Keplerian elements for plotting. It contains manual quadrant checks:
```python
if e_vec[2] < 0:
    omega = 2*np.pi - omega
if np.dot(r, v) < 0:
    nu = 2*np.pi - nu
```
These checks could produce wrong elements for LFIRE-4's specific orientation, making the OE plot misleading without affecting the actual orbit. This would only affect visualisation, not the trajectory.

**Diagnostic:**  
This does NOT explain the oscillation in the distance plot (which uses raw ECI positions, not Keplerian elements). Ruled out as a cause of the distance oscillation but could affect interpretation of the OE plots.

---

## Recommended Investigation Order

| Step | Action | Rules out / confirms |
|---|---|---|
| 1 | Disable SMA correction for one run; compare LFIRE-4 behaviour | Cause 5 |
| 2 | Print oe_i before and after correction for all deputies | Causes 1, 2 |
| 3 | Compute mothership state round-trip error | Cause 3 |
| 4 | Check if LFIRE-4 separation ever hits exactly 0 | Cause 4 |

Start with Step 1 — it requires no code change beyond commenting out 4 lines and is the fastest way to determine whether the SMA correction is the culprit.
