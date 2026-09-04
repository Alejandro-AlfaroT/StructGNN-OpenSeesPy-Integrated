"""
Run_Naming.py
=============

Canonical analysis run-directory naming, with no OpenSees dependency.

Both the analysis entry point and the dataset scheduler need to agree on what
a run directory is called: the analysis creates it, and the scheduler has to
predict it to decide whether that run is already finished. The logic used to
live in Ground_Motion_Main, which imports OpenSees at module scope, so the
scheduler could not reuse it. Duplicating the rules in the scheduler would let
the two drift apart and make the scheduler silently re-run completed work, so
they are extracted here instead.

Unit-free; purely a string convention.
"""


def safe_name(value):
    """Reduce a label to characters that are safe in a directory name."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def variant_value(value):
    """Format a numeric variant so it survives as a path segment."""
    return (
        f"{float(value):g}"
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def analysis_run_name(
    base_name,
    *,
    x_only=False,
    scale_factor=None,
    damping_ratio=0.05,
    rayleigh_mode_i=0,
    rayleigh_mode_j=2,
    dt_factor=1.0,
):
    """Directory name for one analysis run.

    Only non-default settings contribute a suffix, so the common case stays
    readable and older run directories keep their existing names. Intensity
    scaling is part of the name, which is what lets several intensities of the
    same record pair coexist as separate runs under one case.
    """
    name = safe_name(base_name)
    suffixes = []
    if x_only and not name.endswith("__x_only"):
        suffixes.append("x_only")
    if scale_factor is not None and abs(float(scale_factor) - 1.0) > 1.0e-12:
        suffixes.append(f"sf_{variant_value(scale_factor)}")
    if abs(float(damping_ratio) - 0.05) > 1.0e-12:
        suffixes.append(f"zeta_{variant_value(damping_ratio)}")
    if int(rayleigh_mode_i) != 0 or int(rayleigh_mode_j) != 2:
        suffixes.append(f"rayleigh_{int(rayleigh_mode_i)}_{int(rayleigh_mode_j)}")
    if abs(float(dt_factor) - 1.0) > 1.0e-12:
        suffixes.append(f"dtf_{variant_value(dt_factor)}")
    if suffixes:
        name += "__" + "__".join(suffixes)
    return name
