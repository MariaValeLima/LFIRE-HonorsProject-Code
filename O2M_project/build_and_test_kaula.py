from pathlib import Path
import numpy as np
import sys
import os

# Make sure Python can find your package
sys.path.append(str(Path(__file__).resolve().parent))

from osculating2mean.perturbations_kaula import build_kaula_f2py
from osculating2mean.perturbations_kaula import kaula_geopotential_perturbations

ROOT = Path(__file__).resolve().parent

# print("Building Kaula FORTRAN module...")

# # ---- BUILD STEP ----
# build_kaula_f2py(
#     fortran_src_path=str(ROOT / "KaulaGeopotentialPerturbations_mex.F"),
#     egm96_path=str(ROOT / "egm96_degree360.ascii"),
#     module_name="kaula_mex",
# )

# print("Build successful.\n")

# ---- IMPORT TEST ----
try:
    import kaula_mex
    print("Import successful:", kaula_mex)
except Exception as e:
    print("Import failed:", e)
    raise

# ---- SIMPLE PHYSICS TEST ----
print("\nRunning Kaula perturbation test...")

OEmean = np.array([7e6, 1.0, 1e-3, 0.0, 0.3, 0.5])

dOE = kaula_geopotential_perturbations(
    t_tdb=0.0,
    OEmean=OEmean,
    degree=20,
    egm96_path=str(ROOT / "egm96_degree360.ascii"),
    module_name="kaula_mex",
)

print("Perturbation vector:")
print(dOE)

print("\nIf you see numbers (not errors), Kaula works.")