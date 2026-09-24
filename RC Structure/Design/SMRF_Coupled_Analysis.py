"""Monolithic elastic slab/web/column gravity diagnostic, NOT design approval.

Every floor has shared ShellMITC4 nodes at its slab midsurface. Downstand
beam webs (only Hbeam-hslab, not a second slab flange) are rigid-linked at
their actual centroid offset to those nodes. Full-height elastic columns
connect the floor grid intersections to a fixed base. Nothing is pinned at
an elevated column node and no in-plane shell DOFs are artificially fixed.

The auxiliary shell membrane supplies finite in-plane stiffness: this is
NOT the production rigid-diaphragm NTHA model. No seismic loads, P-delta,
creep/cracking iteration, solid joint region or independent engineering
qualification are claimed. The existing project's Iy+Iz torsion proxy is
retained explicitly for sensitivity work, not asserted as Saint-Venant J.

No files or global Structure_Parameters are changed. The caller must provide
an empty OpenSees domain. The diagnostic clears only its own scratch domain.

References: OpenSees ShellMITC4, elasticBeamColumn, rigidLink and Transformation:
https://opensees.berkeley.edu/OpenSees/manuals/usermanual/640.htm
https://opensees.github.io/OpenSeesDocumentation/user/manual/model/mp_constraint/rigidLink.html
https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/constraint/TransformationMethod.html
"""
from __future__ import annotations

import copy
import math

import openseespy.opensees as ops

from Design.SMRF_Floor_Analysis import _inputs, _integer, _number, MAX_SHELLS

METHOD_VERSION = "smrf_monolithic_eccentric_shell_web_column_gravity_v2_section_actions"


def _rectangle(b, h, fc, modifier, nu):
    e = 57.0 * math.sqrt(fc * 1000.)
    iy, iz = b*h**3/12., h*b**3/12.
    return {"area_in2": b*h, "ec_ksi": e, "g_ksi": e/(2*(1+nu)),
            "iy_in4": modifier*iy, "iz_in4": modifier*iz, "j_in4": modifier*(iy+iz),
            "bending_torsion_modifier": modifier, "axial_modifier": 1.,
            "torsion_basis": "Inherited project polar-inertia proxy Iy+Iz; NOT verified Saint-Venant torsion."}


def _frame_element(tag, ni, nj, section, transform):
    ops.element("elasticBeamColumn", tag, ni, nj, section["area_in2"], section["ec_ksi"],
                section["g_ksi"], section["j_in4"], section["iy_in4"], section["iz_in4"], transform)


def _vector(values, length, label):
    values = list(values)
    if len(values) != length or any(not math.isfinite(v) for v in values):
        raise RuntimeError(f"Invalid {label}: expected {length} finite components.")
    return values


def analyze_coupled_gravity(slab_record, geometry, sections, floor_loadcases,
                            mesh_per_bay=4, include_member_weight=True):
    """Solve one explicit gravity combination on all floors simultaneously.

    ``floor_loadcases`` is a list of length num_floor in ascending floor order,
    using the existing floor-analysis D/L/panel-pattern schema. Each floor's
    dead factor also scales that level's beam and underlying column weight.
    ``include_member_weight=False`` is a bounded diagnostic comparison only.
    Optional beam/column_stiffness_modifier default to 1 (gross); these scale
    bending/torsion, not axial stiffness. Maximum 8192 shells across ALL floors.
    """
    nf = _integer(geometry.get("num_floor"), "num_floor")
    sh = _number(geometry.get("story_h_in"), "story_h_in")
    if not isinstance(include_member_weight, bool):
        raise ValueError("include_member_weight must be Boolean.")
    if not isinstance(floor_loadcases, list) or len(floor_loadcases) != nf:
        raise ValueError("Provide one explicit load case per floor, in ascending order.")
    parsed = [_inputs(slab_record, geometry, sections, case, mesh_per_bay, allow_zero=True) for case in floor_loadcases]
    nx, ny, lx, ly, hs, fc, nu, mesh, _ = parsed[0]
    if nf*nx*ny*mesh*mesh > MAX_SHELLS:
        raise ValueError(f"Coupled diagnostic exceeds {MAX_SHELLS} total shells across floors.")
    b = _number(sections.get("b_beam_in"), "b_beam_in")
    hb = _number(sections.get("h_beam_in"), "h_beam_in")
    bc = _number(sections.get("b_col_in"), "b_col_in")
    hc = _number(sections.get("h_col_in"), "h_col_in")
    fcb = _number(sections.get("fc_beam_ksi"), "fc_beam_ksi")
    fcc = _number(sections.get("fc_col_ksi"), "fc_col_ksi")
    bm = _number(sections.get("beam_stiffness_modifier", 1.), "beam_stiffness_modifier")
    cm = _number(sections.get("column_stiffness_modifier", 1.), "column_stiffness_modifier")
    if not hs < hb < sh or hc >= lx or bc >= ly or b >= min(lx, ly):
        raise ValueError("Coupled geometry must leave a downstand web, clear column height and clear beam spans.")
    web = _rectangle(b, hb-hs, fcb, bm, nu)
    column = _rectangle(bc, hc, fcc, cm, nu)
    # From slab mid-plane to downstand-web centroid: -(hs/2+(hb-hs)/2).
    offset = -hb/2.
    gamma = _number(slab_record["concrete_unit_weight_kcf"], "concrete_unit_weight_kcf") / 1728.
    if ops.getNodeTags() or ops.getEleTags():
        raise RuntimeError("Coupled diagnostic requires an empty OpenSees domain; existing model was preserved.")
    ex, ey, dx, dy = nx*mesh, ny*mesh, lx/mesh, ly/mesh
    floor_nodes, web_nodes, base_nodes = {}, {}, {}
    positions, applied_nodes, shells, beams, columns, links = {}, {}, [], [], [], []
    weight_ledger = {"slab_area_load_kip": 0., "beam_drop_weight_kip": 0., "column_weight_kip": 0.}
    # Global applied force and moment about (0,0,0); all gravity forces vertical.
    applied_force = [0., 0., 0.]
    applied_moment = [0., 0., 0.]
    result = {"method_version": METHOD_VERSION, "verified": False, "applied_to_design": False,
              "status": "not_analyzed", "inputs": {"slab": copy.deepcopy(slab_record),
              "geometry": copy.deepcopy(geometry), "sections": copy.deepcopy(sections),
              "floor_loadcases": copy.deepcopy(floor_loadcases), "include_member_weight": include_member_weight},
              "mesh_per_bay": mesh, "web_section": web, "column_section": column,
              "shell_response_convention": {"gauss_order": ["p11", "p22", "p12", "m11", "m22", "m12", "q1", "q2"],
                                            "axes": "local 1=X, local 2=Y, normal=+Z; CCW node ordering",
                                            "units": "membrane/shear kip/in; moments kip-in/in",
                                            "sagging_moments": "negative raw m11 and m22",
                                            "location": "four Gauss points per cell; no face extrapolation"},
              "web_centroid_offset_from_slab_midplane_in": offset,
              "limitations": ["Static linear gravity diagnostic only, not an approved SMRF design or NTHA.",
                              "Finite elastic membrane in this auxiliary model; production rigid diaphragm unchanged.",
                              "Gross shell stiffness; optional web/column modifiers are sensitivity assumptions, not a verified composite cracking model.",
                              "Beam web excludes slab flange, but centerline joint stiffness overlaps remain unmodeled in 3D solid detail.",
                              "Member weight uses clear beam spans and slab-excluded column volumes, smeared on centerlines.",
                              "Iy+Iz torsion proxy retained; verify actual torsional stiffness, joint-face actions, cracking, creep and support-face resultants.",
                              "Web segment moments alone are NOT composite beam/slab design moments."]}

    def node(x, y, z):
        tag = len(positions)+1
        positions[tag] = (x, y, z)
        ops.node(tag, x, y, z)
        return tag

    def gravity_resultant(p, x, y):
        applied_force[2] -= p
        applied_moment[0] -= p*y
        applied_moment[1] += p*x

    try:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.section("ElasticMembranePlateSection", 1, 57.*math.sqrt(fc*1000.), nu, hs, 0.)
        ops.geomTransf("Linear", 1, 1., 0., 0.)  # column
        ops.geomTransf("Linear", 2, 0., 0., 1.)  # both beam directions
        for j in range(ny+1):
            for i in range(nx+1):
                n = node(i*lx, j*ly, 0.)
                base_nodes[i, j] = n
                ops.fix(n, 1, 1, 1, 1, 1, 1)
        for k in range(1, nf+1):
            for j in range(ey+1):
                for i in range(ex+1):
                    n = node(i*dx, j*dy, k*sh)
                    floor_nodes[k, i, j] = n
                    if i % mesh == 0 or j % mesh == 0:
                        nw = node(i*dx, j*dy, k*sh+offset)
                        web_nodes[k, i, j] = nw
                        # No constraint chains: shell node is retained only;
                        # the web node is constrained only, in one rigid link.
                        ops.rigidLink("beam", n, nw)
                        links.append((n, nw))
        tag = 0
        for k in range(1, nf+1):
            for j in range(ey):
                for i in range(ex):
                    tag += 1
                    nodes = [floor_nodes[k, i, j], floor_nodes[k, i+1, j],
                             floor_nodes[k, i+1, j+1], floor_nodes[k, i, j+1]]
                    ops.element("ShellMITC4", tag, *nodes, 1)
                    pressure = parsed[k-1][-1][i//mesh, j//mesh] / 144.
                    p = pressure*dx*dy
                    weight_ledger["slab_area_load_kip"] += p
                    gravity_resultant(p, (i+.5)*dx, (j+.5)*dy)
                    for n in nodes:
                        applied_nodes[n] = applied_nodes.get(n, 0.)+p/4
                    shells.append({"tag": tag, "floor": k, "i": i, "j": j, "nodes": nodes})
            for j in range(ny+1):
                for i in range(nx+1):
                    ni = base_nodes[i, j] if k == 1 else floor_nodes[k-1, i*mesh, j*mesh]
                    nj = floor_nodes[k, i*mesh, j*mesh]
                    tag += 1
                    _frame_element(tag, ni, nj, column, 1)
                    columns.append({"tag": tag, "story": k, "grid_i": i, "grid_j": j, "nodes": [ni, nj]})
            for axis, lines, intervals in (("x", ny+1, ex), ("y", nx+1, ey)):
                for line in range(lines):
                    for t in range(intervals):
                        i, j = (t, line*mesh) if axis == "x" else (line*mesh, t)
                        ni = web_nodes[k, i, j]
                        nj = web_nodes[k, i+int(axis == "x"), j+int(axis == "y")]
                        tag += 1
                        _frame_element(tag, ni, nj, web, 2)
                        beams.append({"tag": tag, "floor": k, "axis": axis, "line_index": line,
                                      "span_index": t//mesh, "segment_index": t % mesh, "nodes": [ni, nj],
                                      "body_load_force_kip": [0., 0., 0.]})
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for n, p in applied_nodes.items():
            ops.load(n, 0., 0., -p, 0., 0., 0.)
        slab_force, slab_moment = applied_force.copy(), applied_moment.copy()
        if include_member_weight:
            for item in columns:
                factor = floor_loadcases[item["story"]-1]["dead_factor"]
                p = factor*bc*hc*(sh-hs)*gamma
                ops.eleLoad("-ele", item["tag"], "-type", "-beamUniform", 0., 0., -p/sh)
                weight_ledger["column_weight_kip"] += p
                gravity_resultant(p, item["grid_i"]*lx, item["grid_j"]*ly)
            for item in beams:
                axis = item["axis"]
                length, clear = (lx, lx-hc) if axis == "x" else (ly, ly-bc)
                factor = floor_loadcases[item["floor"]-1]["dead_factor"]
                w = factor*b*(hb-hs)*gamma*clear/length
                ops.eleLoad("-ele", item["tag"], "-type", "-beamUniform", 0., -w, 0.)
                p = w*length/mesh
                item["body_load_force_kip"] = [0., 0., -p]
                ni, nj = item["nodes"]
                weight_ledger["beam_drop_weight_kip"] += p
                gravity_resultant(p, (positions[ni][0]+positions[nj][0])/2., (positions[ni][1]+positions[nj][1])/2.)
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.)
        ops.analysis("Static")
        code = ops.analyze(1)
        result["analysis_return_code"] = code
        if code != 0:
            result["status"] = "analysis_failed"
            return result
        ops.reactions()
        reactions, rf, rm = [], [0., 0., 0.], [0., 0., 0.]
        for (i, j), n in base_nodes.items():
            r = _vector(ops.nodeReaction(n), 6, "base reaction")
            x, y, z = positions[n]
            for d in range(3):
                rf[d] += r[d]
            rm[0] += r[3]+y*r[2]-z*r[1]
            rm[1] += r[4]+z*r[0]-x*r[2]
            rm[2] += r[5]+x*r[1]-y*r[0]
            reactions.append({"node": n, "grid_i": i, "grid_j": j, "force_moment": r})
        force_scale = max(1., abs(applied_force[2]))
        moment_scale = force_scale*max(nx*lx, ny*ly, nf*sh)
        force_error = max(abs(a+b) for a, b in zip(rf, applied_force))/force_scale
        moment_error = max(abs(a+b) for a, b in zip(rm, applied_moment))/moment_scale
        disps = {n: _vector(ops.nodeDisp(n), 6, "node displacement") for n in positions}
        link_error = 0.
        for retained, constrained in links:
            u = disps[retained]
            expected = [u[0]+offset*u[4], u[1]-offset*u[3], u[2], *u[3:]]
            link_error = max(link_error, *(abs(a-b) for a, b in zip(disps[constrained], expected)))
        # Shell -> frame actions at retained shell coordinates, all six DOFs.
        # These are an equilibrium partition at THIS coupled displacement,
        # not reusable fixed loads for a different bare frame stiffness.
        shell_forces = {n: [0.]*6 for n in floor_nodes.values()}
        panels = []
        for item in shells:
            f = _vector(ops.eleForce(item["tag"]), 24, "shell nodal force")
            for index, n in enumerate(item["nodes"]):
                for d in range(6):
                    shell_forces[n][d] += f[index*6+d]
            stresses = _vector(ops.eleResponse(item["tag"], "stresses"), 32, "shell Gauss resultants")
            q = parsed[item["floor"]-1][-1][item["i"]//mesh, item["j"]//mesh] / 144.
            panels.append({"floor": item["floor"], "panel_i": item["i"]//mesh,
                           "panel_j": item["j"]//mesh, "element": item["tag"],
                           "cell_i": item["i"], "cell_j": item["j"],
                           "bounds_xy_in": [item["i"]*dx, (item["i"]+1)*dx, item["j"]*dy, (item["j"]+1)*dy],
                           "gauss_resultants_raw": stresses,
                           "node_positions_in": [positions[n] for n in item["nodes"]],
                           "global_nodal_force_kip_kip_in": f,
                           "applied_nodal_force_kip_kip_in": [0., 0., -q*dx*dy/4., 0., 0., 0.]*4})
        interface = []
        for retained, _ in links:
            loads = [-v for v in shell_forces[retained]]
            loads[2] -= applied_nodes.get(retained, 0.)
            interface.append({"node": retained, "position_in": positions[retained], "force_moment": loads})
        interface_force, interface_moment = [0.]*3, [0.]*3
        for item in interface:
            f, (x, y, z) = item["force_moment"], item["position_in"]
            for d in range(3):
                interface_force[d] += f[d]
            interface_moment[0] += f[3]+y*f[2]-z*f[1]
            interface_moment[1] += f[4]+z*f[0]-x*f[2]
            interface_moment[2] += f[5]+x*f[1]-y*f[0]
        interface_f_error = max(abs(a-b) for a, b in zip(interface_force, slab_force))/force_scale
        interface_m_error = max(abs(a-b) for a, b in zip(interface_moment, slab_moment))/moment_scale
        for group in (beams, columns):
            for item in group:
                item["local_force_kip_kip_in"] = _vector(ops.eleResponse(item["tag"], "localForce"), 12, "member local force")
                item["global_force_kip_kip_in"] = _vector(ops.eleForce(item["tag"]), 12, "member global force")
                item["node_positions_in"] = [positions[n] for n in item["nodes"]]
                if group is beams:
                    item["body_load_position_in"] = [(a+b)/2. for a,b in zip(*item["node_positions_in"])]
        floor_boundary_actions = []
        for col in columns:
            # Column actions ON the floor are opposite the column's native
            # nodal resisting forces. Include both adjacent stories separately.
            for floor, end in ((col["story"]-1, 0), (col["story"], 1)):
                if floor > 0:
                    floor_boundary_actions.append({"floor": floor, "source_column": col["tag"],
                        "end": "i" if end == 0 else "j", "position_in": col["node_positions_in"][end],
                        "force_moment": [-v for v in col["global_force_kip_kip_in"][6*end:6*end+6]]})
        floors = []
        for k in range(1, nf+1):
            nodes = [n for (kk, i, j), n in floor_nodes.items() if kk == k]
            floors.append({"floor": k, "maximum_downward_displacement_in": max(0., -min(disps[n][2] for n in nodes)),
                           "column_joint_displacements": [{"grid_i": i, "grid_j": j,
                                "displacement_rotation": disps[floor_nodes[k, i*mesh, j*mesh]]}
                                for j in range(ny+1) for i in range(nx+1)]})
        result.update({"weight_ledger": weight_ledger, "base_reactions": reactions,
                       "equilibrium": {"applied_force_kip": applied_force, "applied_moment_kip_in": applied_moment,
                                       "base_force_kip": rf, "base_moment_kip_in": rm,
                                       "force_relative_error": force_error, "moment_relative_error": moment_error,
                                       "numerical_balance_passed": max(force_error, moment_error) < 1e-8},
                       "rigid_offset_max_residual": link_error, "shell_count": len(shells),
                       "beam_segment_count": len(beams), "column_count": len(columns),
                       "node_count": len(positions), "floors": floors, "shell_resultants": panels,
                       "web_segment_actions": beams, "column_actions": columns,
                       "floor_boundary_actions": floor_boundary_actions,
                       "section_action_schema": "native_global_actions_and_applied_loads_v1",
                       "shell_to_frame_nodal_actions": interface,
                       "shell_to_frame_equilibrium": {
                           "interface_force_kip": interface_force, "interface_moment_kip_in": interface_moment,
                           "slab_applied_force_kip": slab_force, "slab_applied_moment_kip_in": slab_moment,
                           "force_relative_error": interface_f_error, "moment_relative_error": interface_m_error,
                           "numerical_balance_passed": max(interface_f_error, interface_m_error) < 1e-8,
                           "basis": "All six interface actions at retained slab-node coordinates; member self-weight excluded."}})
        result["status"] = ("diagnostic_complete" if max(force_error, moment_error, interface_f_error, interface_m_error) < 1e-8
                            and link_error < 1e-8 else "equilibrium_failed")
        return result
    finally:
        ops.wipe()
