"""
osculating2mean
===============
Python translation of the osculating2mean MATLAB package by Leonardo Pedroso.

Public API
----------
rv_to_oe_osc            : ECI state vector  → osculating OE
oe_osc_to_rv            : osculating OE     → ECI state vector
oe_osc_to_oe_mean_eu    : osculating OE     → J2 mean OE  (Eckstein-Ustinov)
oe_mean_eu_to_oe_osc    : J2 mean OE        → osculating OE
oe_osc_to_oe_mean_euk   : osculating OE     → mean OE  (EU + Kaula geopotential)
oe_mean_euk_to_oe_osc   : mean OE           → osculating OE (EU + Kaula)

build_kaula_f2py        : compile the Fortran Kaula mex into a Python extension
"""

from .oe_conversions import (
    rv_to_oe_osc,
    oe_osc_to_rv,
    oe_osc_to_oe_mean_eu,
    oe_mean_eu_to_oe_osc,
    oe_osc_to_oe_mean_euk,
    oe_mean_euk_to_oe_osc,
)
from .perturbations_kaula import build_kaula_f2py

__all__ = [
    "rv_to_oe_osc",
    "oe_osc_to_rv",
    "oe_osc_to_oe_mean_eu",
    "oe_mean_eu_to_oe_osc",
    "oe_osc_to_oe_mean_euk",
    "oe_mean_euk_to_oe_osc",
    "build_kaula_f2py",
]