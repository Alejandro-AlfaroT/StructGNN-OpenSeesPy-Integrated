import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

# --------------------------------------------------
# Geometry
# --------------------------------------------------
numBayX = 2
numBayY = 2
numFloor = 4

bayX = 240.0   # in
bayY = 240.0   # in
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
                         top_bars, bot_bars, bar_area, side_bars=0):
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

    ops.section('Fiber', secTag, '-GJ', 1.0e8)

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

    # Optional side bars, useful for columns
    if side_bars > 0:
        ops.layer('straight', steel_mat, side_bars, bar_area, yc1, zc1, yc1, zc2)
        ops.layer('straight', steel_mat, side_bars, bar_area, yc2, zc1, yc2, zc2)

# --------------------------------------------------
# Create RC Fiber Sections
# --------------------------------------------------
col_sec_tag = 101
beam_sec_tag = 102

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
    bar_area=Abar8
)

make_rc_rect_section(
    beam_sec_tag,
    b_beam, h_beam, cover,
    core_beam_tag, cover_beam_tag, steel_tag,
    top_bars=2,
    bot_bars=2,
    side_bars=0,
    bar_area=Abar6
)

# Beam integration
num_int_pts = 5
ops.beamIntegration('Lobatto', 1, col_sec_tag, num_int_pts)
ops.beamIntegration('Lobatto', 2, beam_sec_tag, num_int_pts)

# --------------------------------------------------
# Geometric Transformations
# --------------------------------------------------
ops.geomTransf('Linear', 1, 1, 0, 0)  # columns
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
m = abs(10) / g  # if you want mass consistent with your gravity load

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

# --------------------------------------------------
# Gravity Loads
# --------------------------------------------------
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)

Pnode = -25.0  # kip downward per elevated node
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
    print("Static gravity analysis failed")
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
# Lateral Load Pattern
# --------------------------------------------------
ops.timeSeries('Linear', 2)
ops.pattern('Plain', 2, 2)

Fx = 10.0  # kip lateral load at each floor master node, global X
for k in range(1, numFloor + 1):
    master = node_tag(k, numBayX // 2, numBayY // 2)
    ops.load(master, Fx, 0.0, 0.0, 0.0, 0.0, 0.0)

# --------------------------------------------------
# Gravity + Lateral Analysis
# --------------------------------------------------
ok2 = ops.analyze(1)

if ok2 != 0:
    print("Gravity + lateral analysis failed")
else:
    print("Gravity + lateral analysis succeeded")

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
# Modal Analysis
# --------------------------------------------------
lam = ops.eigen(6)
periods = [2 * math.pi / math.sqrt(x) for x in lam]

print("\nPeriods:")
for i, T in enumerate(periods, start=1):
    print(f"Mode {i}: T = {T:.6f} sec")

print("\nRoof-node eigenvector components:")
for mode in range(1, 7):
    ux_mode = ops.nodeEigenvector(roof_node, mode, 1)
    uy_mode = ops.nodeEigenvector(roof_node, mode, 2)
    uz_mode = ops.nodeEigenvector(roof_node, mode, 3)
    print(
        f"Mode {mode}: "
        f"Ux = {ux_mode:.6e}, "
        f"Uy = {uy_mode:.6e}, "
        f"Uz = {uz_mode:.6e}"
    )

# --------------------------------------------------
# Plots using opsvis
# --------------------------------------------------

# 1. Undeformed model
opsv.plot_model()
plt.title("Undeformed Structure")
plt.tight_layout()

# 2. Applied loads (manual arrows)
plot_loads_manual()

# 3. Deformed shape from gravity + lateral
opsv.plot_defo(sfac=200)
plt.title("Deformed Shape (Gravity + Lateral, scaled)")
plt.tight_layout()

# 4. Mode shapes
mode_scale = 10

for mode in range(1, 5):
    opsv.plot_mode_shape(mode, sfac=mode_scale)
    plt.title(f"Mode Shape {mode} (scaled)")
    plt.tight_layout()

plt.show()