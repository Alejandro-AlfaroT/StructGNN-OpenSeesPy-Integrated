"""Physical-face recovery from rectangular MITC4 Gauss-point resultants.

Supports uniform or nonuniform axis-aligned rectangular cells. Recover the raw
tensor within the cell containing the physical face, on its clear-span side.
Never average between elements or extend a result beyond its owning cell.
This is a numerical recovery rule, not an engineering verification assertion.
"""
from __future__ import annotations

import math
from collections import defaultdict

METHOD_VERSION = "mitc4_physical_face_clear_span_cell_v1"
FIELDS = ("mx", "my", "mxy_raw", "qx_raw", "qy_raw")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def _cell(points):
    if set(points) != {1, 2, 3, 4}:
        raise ValueError("Physical-face recovery requires all four distinct Gauss points per element")
    values = {k: {field: _finite(p[field], field) for field in ("x_in", "y_in", *FIELDS)}
              for k, p in points.items()}
    x1, x2 = values[1]["x_in"], values[2]["x_in"]
    y1, y2 = values[1]["y_in"], values[4]["y_in"]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Expected CCW axis-aligned rectangular Gauss-point ordering")
    for k, x, y in ((2,x2,y1),(3,x2,y2),(4,x1,y2)):
        if not math.isclose(values[k]["x_in"],x,rel_tol=0,abs_tol=1e-8) or not math.isclose(values[k]["y_in"],y,rel_tol=0,abs_tol=1e-8):
            raise ValueError("Nonrectangular or inconsistent Gauss-point geometry")
    dx, dy = math.sqrt(3)*(x2-x1), math.sqrt(3)*(y2-y1)
    return {"points":values, "x_lo":(x1+x2-dx)/2, "x_hi":(x1+x2+dx)/2,
            "y_lo":(y1+y2-dy)/2, "y_hi":(y1+y2+dy)/2}


def _recover(cell, x, y):
    tol=1e-8
    if not cell["x_lo"]-tol <= x <= cell["x_hi"]+tol or not cell["y_lo"]-tol <= y <= cell["y_hi"]+tol:
        raise ValueError("Refusing to extrapolate outside the owning element")
    p=cell["points"]
    tx=(x-p[1]["x_in"])/(p[2]["x_in"]-p[1]["x_in"])
    ty=(y-p[1]["y_in"])/(p[4]["y_in"]-p[1]["y_in"])
    weights={1:(1-tx)*(1-ty),2:tx*(1-ty),3:tx*ty,4:(1-tx)*ty}
    return {field: math.fsum(weights[k]*p[k][field] for k in weights) for field in FIELDS}


def recover_panel_faces(panel, geometry, beam_width_in):
    """Return both normal faces of both axes, with separate cell-side samples.

    At each face, retain each clipped cell's transverse endpoints and Gauss
    coordinates. A shared transverse endpoint appears once for each owning
    cell: neither side of a stress discontinuity is discarded or averaged.
    Samples exclude perpendicular beam widths. All four faces must be covered.
    """
    lx=_finite(geometry["bay_x_in"],"bay_x_in"); ly=_finite(geometry["bay_y_in"],"bay_y_in")
    half=_finite(beam_width_in,"beam_width_in")/2
    if half<=0 or 2*half>=min(lx,ly):
        raise ValueError("Beam width must leave a positive clear slab panel")
    pi,pj=panel["i"],panel["j"]
    if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in (pi,pj)):
        raise ValueError("Panel indices must be nonnegative integers")
    if pi>=geometry["num_bay_x"] or pj>=geometry["num_bay_y"]:
        raise ValueError("Panel is outside floor geometry")
    grouped=defaultdict(dict)
    for point in panel["gauss_point_resultants"]:
        element,gp=point["element"],point["gauss_point"]
        if isinstance(element, bool) or not isinstance(element, int) or element <= 0:
            raise ValueError("Element tags must be positive integers")
        if isinstance(gp, bool) or not isinstance(gp, int) or gp not in (1, 2, 3, 4):
            raise ValueError("Gauss-point numbers must be integers 1 through 4")
        if gp in grouped[element]:
            raise ValueError("Duplicate element Gauss point")
        grouped[element][gp]=point
    cells={element:_cell(points) for element,points in grouped.items()}
    samples=[];coverage=[];tol=1e-8
    for axis,origin,length,tangent,t_origin,t_length in (("x",pi*lx,lx,"y",pj*ly,ly),("y",pj*ly,ly,"x",pi*lx,lx)):
        for support,position in (("lower",origin+half),("upper",origin+length-half)):
            intervals=[]
            for element,cell in cells.items():
                lo,hi=cell[axis+"_lo"],cell[axis+"_hi"]
                # Half-open ownership selects the clear slab, including when a
                # face coincides with a mesh boundary. Never select the first
                # row merely because it is adjacent to the beam centerline.
                owned=(lo-tol<=position<hi-tol) if support=="lower" else (lo+tol<position<=hi+tol)
                if not owned:
                    continue
                start=max(cell[tangent+"_lo"],t_origin+half)
                end=min(cell[tangent+"_hi"],t_origin+t_length-half)
                if end-start<=tol:
                    continue
                intervals.append((start,end))
                coords={start,end}
                coords.update(p[tangent+"_in"] for p in cell["points"].values()
                              if start+tol<p[tangent+"_in"]<end-tol)
                for t in sorted(coords):
                    x,y=(position,t) if axis=="x" else (t,position)
                    samples.append({"panel_id":panel["panel_id"],"axis":axis,"support":support,
                        "element":element,"x_in":x,"y_in":y,"face_position_in":position,
                        "clear_span_side":"increasing" if support=="lower" else "decreasing",
                        "cell_bounds_in":{k:cell[k] for k in ("x_lo","x_hi","y_lo","y_hi")},
                        "raw_resultants":_recover(cell,x,y)})
            reached=t_origin+half
            for start,end in sorted(intervals):
                if start>reached+tol:
                    raise ValueError(f"Incomplete physical-face coverage at {panel['panel_id']}/{axis}/{support}")
                if start<reached-tol:
                    raise ValueError(f"Overlapping physical-face cells at {panel['panel_id']}/{axis}/{support}")
                reached=end
            if not intervals or reached<t_origin+t_length-half-tol:
                raise ValueError(f"Missing physical-face coverage at {panel['panel_id']}/{axis}/{support}")
            coverage.append({"axis":axis,"support":support,"position_in":position,
                             "transverse_clear_interval_in":[t_origin+half,t_origin+t_length-half],
                             "owning_cell_count":len(intervals)})
    return {"method":METHOD_VERSION,"samples":samples,"coverage":coverage,
            "complete":True,"engineering_verified":False}
