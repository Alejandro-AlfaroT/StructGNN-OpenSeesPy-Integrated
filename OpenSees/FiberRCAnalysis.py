import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from sympy.physics.units import angular_mil
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

# --------------------------------------------------
# Geometry
# --------------------------------------------------
numBayX = 3
numBayY = 3
numFloor = 5

bayX = 120.0   # in
bayY = 120.0   # in
storyH = 120.0 # in

def node_tag(k, i, j):
    return k * ((numBayX + 1) * (numBayY + 1)) + j * (numBayX + 1) + i + 1

# Create nodes
for k in range(numFloor + 1):
    z = k * storyH
    for j in range(numBayY + 1):
        y = j * bayY
        for i in range(numBayX + 1):
            x = i * bayX
            ops.node(node_tag(k, i, j), x, y, z)

# Fix base
for j in range(numBayY + 1):
    for i in range(numBayX + 1):
        ops.fix(node_tag(0, i, j), 1, 1, 1, 1, 1, 1)

# --------------------------------------------------
# Material / Section Properties
# --------------------------------------------------

# Concrete strengths
fc_col_ksi = 5.0      # columns, ksi
fc_beam_ksi = 4.0     # beams, ksi

fc_col_psi = fc_col_ksi * 1000.0
fc_beam_psi = fc_beam_ksi * 1000.0

# ACI approximate Ec = 57000 sqrt(fc') psi
Ec_col = 57000.0 * math.sqrt(fc_col_psi) / 1000.0   # ksi
Ec_beam = 57000.0 * math.sqrt(fc_beam_psi) / 1000.0 # ksi

Gc_col = 0.4 * Ec_col
Gc_beam = 0.4 * Ec_beam

# Columns
b_col = 18.0
h_col = 18.0
A_col = b_col * h_col
Iy_col = h_col * b_col**3 / 12.0
Iz_col = b_col * h_col**3 / 12.0
J_col = Iy_col + Iz_col

# Beams
b_beam = 12.0
h_beam = 18.0
A_beam = b_beam * h_beam
Iy_beam = h_beam * b_beam**3 / 12.0
Iz_beam = b_beam * h_beam**3 / 12.0
J_beam = Iy_beam + Iz_beam

# --------------------------------------------------
# RC Fiber Section Definitions
# --------------------------------------------------

# Material tags
cover_col_tag = 1
core_col_tag = 2
cover_beam_tag = 3
core_beam_tag = 4
steel_tag = 5

# Steel
fy = 60.0      # ksi
Es = 29000.0   # ksi
b_steel = 0.01

ops.uniaxialMaterial('Steel02', steel_tag, fy, Es, b_steel)

# Concrete02 inputs are negative in compression
# Concrete02 tag fpc epsc0 fpcu epsU lambda ft Ets

ops.uniaxialMaterial('Concrete02', cover_col_tag, -fc_col_ksi, -0.002, -0.20 * fc_col_ksi, -0.006, 0.1, 0.0, 0.0)
ops.uniaxialMaterial('Concrete02', core_col_tag,  -1.15 * fc_col_ksi, -0.0025, -0.30 * fc_col_ksi, -0.020, 0.1, 0.0, 0.0)

ops.uniaxialMaterial('Concrete02', cover_beam_tag, -fc_beam_ksi, -0.002, -0.20 * fc_beam_ksi, -0.006, 0.1, 0.0, 0.0)
ops.uniaxialMaterial('Concrete02', core_beam_tag,  -1.10 * fc_beam_ksi, -0.0025, -0.30 * fc_beam_ksi, -0.015, 0.1, 0.0, 0.0)


def make_rc_rect_section(secTag, b, h, cover, core_mat, cover_mat, steel_mat,
                         top_bars, bot_bars, bar_area, side_bars=0, GJ=1.0e8):
    """
    Rectangular RC fiber section.
    Local section coordinates:
    y = horizontal section width direction
    z = vertical section depth direction
    """

    y1 = -b / 2.0
    y2 =  b / 2.0
    z1 = -h / 2.0
    z2 =  h / 2.0

    yc1 = y1 + cover
    yc2 = y2 - cover
    zc1 = z1 + cover
    zc2 = z2 - cover

    ops.section('Fiber', secTag, '-GJ', GJ)

    # Core concrete
    ops.patch('rect', core_mat, 12, 12, yc1, zc1, yc2, zc2)

    # Cover concrete: bottom, top, left, right
    ops.patch('rect', cover_mat, 12, 2, y1, z1, y2, zc1)
    ops.patch('rect', cover_mat, 12, 2, y1, zc2, y2, z2)
    ops.patch('rect', cover_mat, 2, 12, y1, zc1, yc1, zc2)
    ops.patch('rect', cover_mat, 2, 12, yc2, zc1, y2, zc2)

    # Longitudinal rebar
    ops.layer('straight', steel_mat, top_bars, bar_area, yc1, zc2, yc2, zc2)
    ops.layer('straight', steel_mat, bot_bars, bar_area, yc1, zc1, yc2, zc1)

    if side_bars > 0:
        ops.layer('straight', steel_mat, side_bars, bar_area, yc1, zc1, yc1, zc2)
        ops.layer('straight', steel_mat, side_bars, bar_area, yc2, zc1, yc2, zc2)

# --------------------------------------------------
# Create RC Fiber Sections
# --------------------------------------------------
col_sec_tag = 101
beam_sec_tag = 102

GJ_col = Gc_col * J_col
GJ_beam = Gc_beam * J_beam

cover = 1.5  # in

Abar8 = 0.79  # #8 bar area, in^2
Abar6 = 0.44  # #6 bar area, in^2

make_rc_rect_section(
    col_sec_tag,
    b_col, h_col, cover,
    core_col_tag, cover_col_tag, steel_tag,
    top_bars=4,
    bot_bars=4,
    side_bars=2,
    bar_area=Abar8,
    GJ=GJ_col
)

make_rc_rect_section(
    beam_sec_tag,
    b_beam, h_beam, cover,
    core_beam_tag, cover_beam_tag, steel_tag,
    top_bars=2,
    bot_bars=2,
    side_bars=0,
    bar_area=Abar6,
    GJ=GJ_beam
)

# Beam integration
num_int_pts = 5
ops.beamIntegration('Lobatto', 1, col_sec_tag, num_int_pts)
ops.beamIntegration('Lobatto', 2, beam_sec_tag, num_int_pts)

# --------------------------------------------------
# Geometric Transformations
# --------------------------------------------------
ops.geomTransf('PDelta', 1, 1, 0, 0)  # columns
ops.geomTransf('Linear', 2, 0, 0, 1)  # X beams
ops.geomTransf('Linear', 3, 0, 0, 1)  # Y beams

# --------------------------------------------------
# Elements
# --------------------------------------------------
eleTag = 1

# Columns
for k in range(numFloor):
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            nI = node_tag(k, i, j)
            nJ = node_tag(k + 1, i, j)
            ops.element('forceBeamColumn', eleTag, nI, nJ, 1, 1)
            eleTag += 1

# Beams in X
for k in range(1, numFloor + 1):
    for j in range(numBayY + 1):
        for i in range(numBayX):
            nI = node_tag(k, i, j)
            nJ = node_tag(k, i + 1, j)
            ops.element('forceBeamColumn', eleTag, nI, nJ, 2, 2 )
            eleTag += 1

# Beams in Y
for k in range(1, numFloor + 1):
    for j in range(numBayY):
        for i in range(numBayX + 1):
            nI = node_tag(k, i, j)
            nJ = node_tag(k, i, j + 1)
            ops.element('forceBeamColumn', eleTag, nI, nJ,3, 2)
            eleTag += 1

# --------------------------------------------------
# Rigid Diaphragms
# --------------------------------------------------
for k in range(1, numFloor + 1):
    floor_nodes = []
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            floor_nodes.append(node_tag(k, i, j))

    master = node_tag(k, numBayX // 2, numBayY // 2)
    slave_nodes = [n for n in floor_nodes if n != master]
    ops.rigidDiaphragm(3, master, *slave_nodes)

# --------------------------------------------------
# Nodal Mass
# --------------------------------------------------
g = 386.4  # in/sec^2
Pnode = -25.0
m = abs(Pnode) / g

for k in range(1, numFloor + 1):
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            n = node_tag(k, i, j)
            ops.mass(n, m, m, 1e-8, 0.0, 0.0, 0.0)


# --------------------------------------------------
# Helper for manual load plotting
# --------------------------------------------------
def get_node_coords(tag):
    c = ops.nodeCoord(tag)
    return c[0], c[1], c[2]

def plot_loads_manual():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # draw member lines manually just for the load figure
    for ele in ops.getEleTags():
        nodes = ops.eleNodes(ele)
        x1, y1, z1 = get_node_coords(nodes[0])
        x2, y2, z2 = get_node_coords(nodes[1])
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=1.0)

    # fixed base markers
    x_fix, y_fix, z_fix = [], [], []
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            n = node_tag(0, i, j)
            x, y, z = get_node_coords(n)
            x_fix.append(x)
            y_fix.append(y)
            z_fix.append(z)
    ax.scatter(x_fix, y_fix, z_fix, c='k', marker='s', s=80)

    # gravity arrows
    grav_scale = 60
    for k in range(1, numFloor + 1):
        for j in range(numBayY + 1):
            for i in range(numBayX + 1):
                n = node_tag(k, i, j)
                x, y, z = get_node_coords(n)
                ax.quiver(x, y, z, 0, 0, -grav_scale, color='g', arrow_length_ratio=0.2)

    # lateral arrows at diaphragm masters
    lat_scale = 80
    for k in range(1, numFloor + 1):
        master = node_tag(k, numBayX // 2, numBayY // 2)
        x, y, z = get_node_coords(master)
        ax.quiver(x, y, z, lat_scale, 0, 0, color='m', arrow_length_ratio=0.2)

    ax.set_title("Applied Loads")
    ax.set_xlabel("X (in)")
    ax.set_ylabel("Y (in)")
    ax.set_zlabel("Z (in)")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

def make_prism_between_points(p1, p2, width, depth):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    axis = p2 - p1
    L = np.linalg.norm(axis)

    if L == 0:
        return []

    axis = axis / L

    # Pick a reference vector not parallel to member axis
    ref = np.array([0, 0, 1], dtype=float)

    if abs(np.dot(axis, ref)) > 0.95:
        ref = np.array([1, 0, 0], dtype=float)

    v1 = np.cross(axis, ref)
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.cross(axis, v1)
    v2 = v2 / np.linalg.norm(v2)

    hw = width / 2
    hd = depth / 2

    corners_p1 = [
        p1 + hw*v1 + hd*v2,
        p1 - hw*v1 + hd*v2,
        p1 - hw*v1 - hd*v2,
        p1 + hw*v1 - hd*v2,
    ]

    corners_p2 = [
        p2 + hw*v1 + hd*v2,
        p2 - hw*v1 + hd*v2,
        p2 - hw*v1 - hd*v2,
        p2 + hw*v1 - hd*v2,
    ]

    faces = [
        [corners_p1[0], corners_p1[1], corners_p1[2], corners_p1[3]], # end face
        [corners_p2[0], corners_p2[1], corners_p2[2], corners_p2[3]], # end face
        [corners_p1[0], corners_p1[1], corners_p2[1], corners_p2[0]],
        [corners_p1[1], corners_p1[2], corners_p2[2], corners_p2[1]],
        [corners_p1[2], corners_p1[3], corners_p2[3], corners_p2[2]],
        [corners_p1[3], corners_p1[0], corners_p2[0], corners_p2[3]],
    ]

    return faces
#----------
# Plots
#----------

# Undeformed model
def plot_extruded_structure():
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    for ele in ops.getEleTags():
        n1, n2 = ops.eleNodes(ele)

        p1 = ops.nodeCoord(n1)
        p2 = ops.nodeCoord(n2)

        x1, y1, z1 = p1
        x2, y2, z2 = p2

        # Detect member orientation
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)

        if dz > dx and dz > dy:
            # Column
            width = b_col
            depth = h_col
        else:
            # Beam
            width = b_beam
            depth = h_beam

        faces = make_prism_between_points(p1, p2, width, depth)

        prism = Poly3DCollection(
            faces,
            facecolor='lightgray',
            alpha=0.9,
            linewidths=0.4,
            edgecolor='k'
        )

        ax.add_collection3d(prism)

    ax.set_title("Extruded RC Frame")
    ax.set_xlabel("X (in)")
    ax.set_ylabel("Y (in)")
    ax.set_zlabel("Z (in)")

    ax.view_init(elev=25, azim=-45)
    ax.set_box_aspect([1, 1, 2])
    ax.set_axis_off()

    plt.tight_layout()

plot_extruded_structure()


# Applied loads (manual arrows)
plot_loads_manual()

# --------------------------------------------------
# Gravity Loads
# --------------------------------------------------
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)

for k in range(1, numFloor + 1):
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            n = node_tag(k, i, j)
            ops.load(n, 0.0, 0.0, Pnode, 0.0, 0.0, 0.0)

# --------------------------------------------------
# Static Gravity Analysis
# --------------------------------------------------
ops.system('BandGeneral')
ops.constraints('Transformation')
ops.numberer('RCM')
ops.test('NormDispIncr', 1e-8, 20)
ops.algorithm('Newton')
ops.integrator('LoadControl', 1.0)
ops.analysis('Static')

ok = ops.analyze(1)

if ok != 0:
    raise RuntimeError("Static gravity analysis failed")
else:
    print("Static gravity analysis succeeded")
    ops.loadConst('-time', 0.0)

roof_node = node_tag(numFloor, numBayX // 2, numBayY // 2)

ux_g = ops.nodeDisp(roof_node, 1)
uy_g = ops.nodeDisp(roof_node, 2)
uz_g = ops.nodeDisp(roof_node, 3)

print("\nGravity-only roof displacement:")
print(f"Ux = {ux_g:.6e} in")
print(f"Uy = {uy_g:.6e} in")
print(f"Uz = {uz_g:.6e} in")

# --------------------------------------------------
# Modal Analysis
# --------------------------------------------------
ops.wipeAnalysis()

ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
#ops.system('FullGeneral')   # use FullGeneral with -fullGenLapack

numModes = numFloor + 2

lam = ops.eigen(numModes)
#lam = ops.eigen('-fullGenLapack', numModes)

print("\nRaw eigenvalues:")
for i, x in enumerate(lam, start=1):
    print(f"Mode {i}: lambda = {x:.12e}")

periods = []
angular_freqs = []
frequencies = []
valid_modes = []

tol = 1e-8

for i, x in enumerate(lam, start=1):
    if x <= tol:
        print(f"Skipping invalid/near-zero eigenvalue in mode {i}: {x:.12e}")
        continue

    w = math.sqrt(x)
    T = 2 * math.pi / w
    f = w / (2 * math.pi)

    valid_modes.append(i)
    periods.append(T)
    angular_freqs.append(w)
    frequencies.append(f)

print("\nModal Properties:")
for mode, T, w, f in zip(valid_modes, periods, angular_freqs, frequencies):
    print(
        f"Mode {mode}: "
        f"T = {T:.6f} sec, "
        f"omega = {w:.6f} rad/sec, "
        f"f = {f:.6f} Hz"
    )

#--------------------------
# Plot Mode Shapes
#--------------------------
mode_scale = 10

for mode in valid_modes:
    opsv.plot_mode_shape(mode, sfac=mode_scale)
    plt.title(f"Mode Shape {mode} (scaled)")
    plt.tight_layout()

# --------------------------------------------------
# Lateral Load Pattern
# --------------------------------------------------
ops.timeSeries('Linear', 2)
ops.pattern('Plain', 2, 2)

Fx = 10.0  # kip lateral load at each floor master node, global X
for k in range(1, numFloor + 1):
    master = node_tag(k, numBayX // 2, numBayY // 2)
    ops.load(master, Fx, 0.0, 0.0, 0.0, 0.0, 0.0)

# --------------------------------------------------
# Gravity + Lateral Pushover Analysis
# --------------------------------------------------
ops.wipeAnalysis()
ops.system('BandGeneral')
ops.constraints('Transformation')
ops.numberer('RCM')
ops.test('NormDispIncr', 1e-6, 50)
ops.algorithm('Newton')
ops.integrator('DisplacementControl', roof_node, 1, 0.05)
ops.analysis('Static')

roof_disp = []
base_shear = []

num_steps = 400

for step in range(num_steps):
    ok2 = ops.analyze(1)

    if ok2 != 0:
        ops.test('NormDispIncr', 1e-5, 100)
        ops.algorithm('ModifiedNewton')
        ok2 = ops.analyze(1)

        ops.test('NormDispIncr', 1e-6, 50)
        ops.algorithm('Newton')

    if ok2 != 0:
        print(f"Pushover failed at step {step}")
        break

    ux = ops.nodeDisp(roof_node, 1)

    ops.reactions()
    vx = 0.0
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            base_node = node_tag(0, i, j)
            vx += ops.nodeReaction(base_node, 1)

    roof_disp.append(ux)
    base_shear.append(-vx)

print(f"Pushover completed {len(roof_disp)} steps")

plt.figure()
plt.plot(roof_disp, base_shear, '-o', markersize=2)
plt.xlabel("Roof displacement X (in)")
plt.ylabel("Base shear X (kip)")
plt.title("Pushover Curve")
plt.grid(True)
plt.tight_layout()

ux_tot = ops.nodeDisp(roof_node, 1)
uy_tot = ops.nodeDisp(roof_node, 2)
uz_tot = ops.nodeDisp(roof_node, 3)

print("\nGravity + lateral roof displacement:")
print(f"Ux = {ux_tot:.6e} in")
print(f"Uy = {uy_tot:.6e} in")
print(f"Uz = {uz_tot:.6e} in")

print("\nIncrement due to lateral load:")
print(f"dUx = {ux_tot - ux_g:.6e} in")
print(f"dUy = {uy_tot - uy_g:.6e} in")
print(f"dUz = {uz_tot - uz_g:.6e} in")

print("\nElement Forces:")

for ele in ops.getEleTags():
    forces = ops.eleForce(ele)
    print(f"Element {ele}: {forces}")


# --------------------------------------------------
# Plots using opsvis
# --------------------------------------------------


'''
# Undeformed model 
opsv.plot_model()
opsv.plot_model(node_labels=0, element_labels=0)
plt.title("Undeformed Structure")
plt.tight_layout()
opsv.plot_model(
    node_labels=0,
    element_labels=0,
    local_axes=False,
    node_supports=False
)
plt.title("Undeformed Structure")
plt.axis('off')
plt.tight_layout()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot members
for ele in ops.getEleTags():
    n1, n2 = ops.eleNodes(ele)

    x1, y1, z1 = ops.nodeCoord(n1)
    x2, y2, z2 = ops.nodeCoord(n2)

    ax.plot([x1, x2], [y1, y2], [z1, z2], color='black', linewidth=1.4)

# Plot fixed base nodes
x_fix, y_fix, z_fix = [], [], []

for j in range(numBayY + 1):
    for i in range(numBayX + 1):
        n = node_tag(0, i, j)
        x, y, z = ops.nodeCoord(n)
        x_fix.append(x)
        y_fix.append(y)
        z_fix.append(z)

ax.scatter(x_fix, y_fix, z_fix, marker='s', s=90, color='black')

ax.set_title("Undeformed Structure")
ax.set_axis_off()
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()
'''

# Deformed shape from gravity + lateral
sfac = 1
opsv.plot_defo(sfac=sfac)
plt.title(f"Deformed Shape (Gravity + Lateral, sfac: {sfac})")
plt.tight_layout()

plt.show()