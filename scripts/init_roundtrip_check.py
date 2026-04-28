import sys
from pathlib import Path
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from tudatpy.astro import element_conversion

mu_earth = 3.986004418e14

base_kepler = dict(
    semi_major_axis             = 6.99276221e+06,
    eccentricity                = 4.03294322e-03,
    inclination                 = 1.71065169e+00,
    argument_of_periapsis       = 1.31226971e+00,
    longitude_of_ascending_node = 3.82958313e-01,
    true_anomaly                = 3.07018490e+00,
)

oe_in = np.array([
    base_kepler['semi_major_axis'],
    base_kepler['eccentricity'],
    base_kepler['inclination'],
    base_kepler['argument_of_periapsis'],         # TUDAT index 3 = ω
    base_kepler['longitude_of_ascending_node'],   # TUDAT index 4 = Ω
    base_kepler['true_anomaly'],
])

# OE → Cartesian → OE (no offset, mothership only)
state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=mu_earth, **base_kepler
)
oe_out = element_conversion.cartesian_to_keplerian(state, mu_earth)

labels = ["a (m)", "e", "i (rad)", "ω (rad)", "Ω (rad)", "ν (rad)"]

print(f"\n{'':25} {'Input':>20}  {'Output':>20}  {'Delta':>20}")
print("-" * 90)
for lbl, v_in, v_out in zip(labels, oe_in, oe_out):
    delta = v_out - v_in
    print(f"  {lbl:<22}  {v_in:>20.10f}  {v_out:>20.10f}  {delta:>20.6e}")
