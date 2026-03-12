from pathlib import Path
import importlib
import os
import subprocess
import sys
import numpy as np

# On Windows, add the MSYS2 runtime DLL directory so the compiled Fortran
# extension (kaula_mex) can find libgfortran and friends at import time.
if sys.platform == "win32":
    _msys2_bin = Path(r"C:\msys64\ucrt64\bin")
    if _msys2_bin.is_dir():
        os.add_dll_directory(str(_msys2_bin))


def _patch_fortran_data_path(src_text: str, egm96_path: str) -> str:
    """
    Patch the Fortran source for compilation with f2py (no MATLAB MEX):
      1. Remove #include "fintrf.h"  (MATLAB-only header, not needed by f2py)
      2. Remove the mexFunction subroutine block (MATLAB entry point, not needed)
      3. Replace the data path block with a hardcoded path assignment
      4. Insert f2py intent directives for perturb_t2 so z is treated as output
    """
    egm96_path = str(Path(egm96_path).resolve()).replace("\\", "/")
    lines = src_text.splitlines(True)

    out = []
    in_data_block = False
    in_mex_block = False

    for line in lines:
        low = line.lower().strip()

        # 1. Strip the MATLAB MEX header include
        if low.startswith('#include') and 'fintrf.h' in low:
            out.append(f"C     {line.rstrip()}  ! removed for f2py build\n")
            continue

        # 2. Strip the mexFunction block (MATLAB entry point)
        if low.startswith('subroutine mexfunction'):
            in_mex_block = True
            out.append("C     mexFunction removed for f2py build\n")
            continue
        if in_mex_block:
            if low.startswith('end subroutine') or low == 'end':
                in_mex_block = False
            continue

        # 3. Replace data path block
        if "c begin data path" in low:
            in_data_block = True
            out.append(line)
            out.append(f"      ifile1 = '{egm96_path}'\n")
            continue
        if "c end data path" in low:
            in_data_block = False
            out.append(line)
            continue
        if in_data_block:
            continue

        # 4. After the perturb_t2 subroutine declaration, insert f2py intent directives
        if low.startswith('subroutine perturb_t2'):
            out.append(line)
            out.append("Cf2py intent(in)  y\n")
            out.append("Cf2py intent(out) z\n")
            continue

        out.append(line)

    return "".join(out)


def build_kaula_f2py(fortran_src_path: str, egm96_path: str, module_name: str = "kaula_mex"):
    fortran_src_path = Path(fortran_src_path).resolve()
    if not fortran_src_path.exists():
        raise FileNotFoundError(f"Fortran source not found: {fortran_src_path}")

    src_text = fortran_src_path.read_text(encoding="utf-8", errors="ignore")

    if "subroutine" not in src_text.lower():
        raise RuntimeError(
            f"This file does not look like Fortran: {fortran_src_path}\n"
            f"First 200 chars:\n{src_text[:200]}"
        )

    patched_path = fortran_src_path.with_name(fortran_src_path.stem + "_patched.F")
    patched_text = _patch_fortran_data_path(src_text, egm96_path)

    if "from osculating2mean" in patched_text or "\ndef " in patched_text:
        raise RuntimeError("Patched Fortran text contains Python code. Refusing to continue.")

    patched_path.write_text(patched_text, encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "numpy.f2py",
        "-c",
        str(patched_path),
        "-m",
        module_name,
        "--backend",
        "meson",
    ]
    subprocess.check_call(cmd)


def kaula_geopotential_perturbations(
    t_tdb: float,
    OEmean,
    degree: int,
    egm96_path: str,
    module_name: str = "kaula_mex",
) -> np.ndarray:
    OEmean = np.asarray(OEmean, dtype=float).reshape(-1)
    if OEmean.size != 6:
        raise ValueError("OEmean must have 6 elements: [a, u, ex, ey, i, Omega]")

    a, u, ex, ey, inc, Omega = OEmean

    w = float(np.arctan2(ey, ex))
    M = float(u - w)

    twopi = 2.0 * np.pi
    if M > twopi:
        M = M - np.floor(M / twopi) * twopi
    elif M < 0.0:
        M = M + np.ceil(-M / twopi) * twopi

    e = float(np.sqrt(ex**2 + ey**2))
    t_mjd = float(t_tdb / 86400.0 + 51544.5)

    y = np.array([t_mjd, a, M, u, e, inc, Omega, float(degree)], dtype=np.float64).reshape(8, 1)

    mod = importlib.import_module(module_name)
    z = mod.perturb_t2(y)
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    if z.size != 6:
        raise RuntimeError(f"Expected 6 outputs from perturb_t2, got {z.size}")

    return z