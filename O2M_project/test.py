from pathlib import Path
from osculating2mean.perturbations_kaula import build_kaula_f2py

ROOT = Path(__file__).resolve().parent

build_kaula_f2py(
    fortran_src_path=str(ROOT / "KaulaGeopotentialPerturbations_mex.F"),
    egm96_path=str(ROOT / "egm96_degree360.ascii"),
    module_name="kaula_mex",
)
print("Built kaula_mex successfully!")

import numpy as np
from osculating2mean.oe_conversions import oe_osc_to_rv, rv_to_oe_osc, oe_osc_to_oe_mean_eu, oe_mean_eu_to_oe_osc
from osculating2mean.perturbations_eu import eckstein_ustinov_perturbations 
from osculating2mean.perturbations_kaula import kaula_geopotential_perturbations

OE = np.array([7000e3, 1.0, 0.001, 0.0, 0.3, 0.5])
x = oe_osc_to_rv(OE)
OE2 = rv_to_oe_osc(x)

print("OE :", OE)
print("OE2:", OE2)
OEMean = np.array([7000e3, 1.0, 1e-3, 0.0, 0.3, 0.5])
dOE = eckstein_ustinov_perturbations(OEMean)
print(dOE)


OEosc = np.array([7e6, 1.0, 1e-3, 0.0, 0.3, 0.5])
OEMean = oe_osc_to_oe_mean_eu(OEosc)
OEosc2 = oe_mean_eu_to_oe_osc(OEMean)

print("OEosc :", OEosc)
print("OEosc2:", OEosc2)

OEmean = np.array([7e6, 1.0, 1e-3, 0.0, 0.3, 0.5])
dOE = kaula_geopotential_perturbations(
    t_tdb=0.0,
    OEmean=OEmean,
    degree=20,
    egm96_path="egm96_degree360.ascii",
    module_name="kaula_mex",
)
print(dOE)