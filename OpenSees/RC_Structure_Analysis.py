import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RC Structure'))
from RC_Design_Check import run_checks, print_summary

ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

# --------------------------------------------------
# Geometry
# --------------------------------------------------
numBayX = 3
numBayY = 3
numFloor = 6

bayX = 180.0   # in
bayY = 180.0   # in
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
# Uniaxial Materials
# --------------------------------------------------
# Concrete01: Kent-Scott-Park with linear tension softening
# Args: matTag, fpc, epsc0, fpcu, epsU  (compression negative)
#
# Column concrete (fc' = 6 ksi):
#   Confined core  — Mander approximation: f'cc ~ 7.0 ksi, epsc0 ~ -0.004, epsU ~ -0.020
#   Unconfined cover                       — f'c  = 6.0 ksi, spalls at epsU = -0.006
ops.uniaxialMaterial('Concrete01', 1, -7.0, -0.004,  -1.4, -0.020)  # col confined
ops.uniaxialMaterial('Concrete01', 2, -6.0, -0.002,   0.0, -0.006)  # col cover (no tensile strength)

# Beam concrete (fc' = 4 ksi):
#   Confined core  — f'cc ~ 4.5 ksi
#   Unconfined cover
ops.uniaxialMaterial('Concrete01', 3, -4.5, -0.004,  -0.9, -0.020)  # beam confined
ops.uniaxialMaterial('Concrete01', 4, -4.0, -0.002,   0.0, -0.006)  # beam cover

# Steel01: bilinear with strain hardening
# Args: matTag, Fy, E0, b
# Grade 60 rebar: Fy = 60 ksi, Es = 29000 ksi, b = 1% hardening
ops.uniaxialMaterial('Steel01', 5, 60.0, 29000.0, 0.01)

# --------------------------------------------------
# Fiber Sections
# --------------------------------------------------
# Section tag 1: Column (18" x 18")
#   Clear cover to stirrups = 1.5 in; #4 ties (0.5 in dia)
#   -> Core extends to ±7.25 in (to stirrup centerline)
#   -> Bar center offset = 1.5 + 0.25 + 0.5 = 2.25 in from face  ->  ±6.75 in from centroid
#   Longitudinal rebar: 8 #8 bars (Ab = 0.79 in²)
#     corners: (±6.75, ±6.75)   [4 bars]
#     midface: (±6.75,  0.0) and (0.0, ±6.75) [4 bars]

col_half  = 9.0          # half section dimension
col_core  = 7.25         # half core dimension (to stirrup CL)
col_d     = 6.75         # bar center distance from centroid
Ab_col    = 0.79         # #8 bar area, in²
nFibY_col = 8            # fiber subdivisions in Y for patches
nFibZ_col = 8            # fiber subdivisions in Z for patches

ops.section('Fiber', 1)
# Confined core
ops.patch('rect', 1, nFibY_col, nFibZ_col,
          -col_core, -col_core, col_core, col_core)
# Cover patches (4 sides)
ops.patch('rect', 2, 2, nFibZ_col,                         # bottom cover
          -col_half, -col_half, -col_core, col_half)
ops.patch('rect', 2, 2, nFibZ_col,                         # top cover
           col_core, -col_half,  col_half, col_half)
ops.patch('rect', 2, nFibY_col, 2,                         # left cover
          -col_core, -col_half,  col_core, -col_core)
ops.patch('rect', 2, nFibY_col, 2,                         # right cover
          -col_core,  col_core,  col_core,  col_half)
# Corner bars
ops.layer('straight', 5, 4, Ab_col,
           col_d, -col_d,   col_d,  col_d)  # top row
ops.layer('straight', 5, 4, Ab_col,
          -col_d, -col_d,  -col_d,  col_d)  # bottom row
# Mid-face bars (left and right faces)
ops.fiber(0.0,  col_d, Ab_col, 5)
ops.fiber(0.0, -col_d, Ab_col, 5)

# Section tag 2: Beam (12" x 18")
#   Clear cover = 1.5 in; #3 stirrups (0.375 in dia)
#   -> Core extends to ±(6 - 1.6875) = ±4.3125 in (Z) and ±(9 - 1.6875) = ±7.3125 in (Y)
#   Bar center offset = 1.5 + 0.1875 + 0.4375 = 2.125 in from face
#   Top rebar:    3 #7 (Ab = 0.60 in²) at Y = +6.875 in (from centroid)
#   Bottom rebar: 2 #7 (Ab = 0.60 in²) at Y = -6.875 in

bm_hY  = 9.0         # half beam height (18/2)
bm_hZ  = 6.0         # half beam width  (12/2)
bm_cY  = 7.3125      # half core height
bm_cZ  = 4.3125      # half core width
bm_dY  = 6.875       # bar Y-offset from centroid
Ab_bm  = 0.60        # #7 bar area, in²
nFibY_bm = 8
nFibZ_bm = 6

ops.section('Fiber', 2)
# Confined core
ops.patch('rect', 3, nFibY_bm, nFibZ_bm,
          -bm_cY, -bm_cZ,  bm_cY,  bm_cZ)
# Cover patches (4 sides)
ops.patch('rect', 4, 2, nFibZ_bm,          # bottom cover
          -bm_hY, -bm_hZ, -bm_cY,  bm_hZ)
ops.patch('rect', 4, 2, nFibZ_bm,          # top cover
           bm_cY, -bm_hZ,  bm_hY,  bm_hZ)
ops.patch('rect', 4, nFibY_bm, 2,          # left cover
          -bm_cY, -bm_hZ,  bm_cY, -bm_cZ)
ops.patch('rect', 4, nFibY_bm, 2,          # right cover
          -bm_cY,  bm_cZ,  bm_cY,  bm_hZ)
# Top rebar (3 #7 evenly spaced)
ops.layer('straight', 5, 3, Ab_bm,
           bm_dY, -bm_cZ,  bm_dY,  bm_cZ)
# Bottom rebar (2 #7)
ops.layer('straight', 5, 2, Ab_bm,
          -bm_dY, -bm_cZ, -bm_dY,  bm_cZ)

# --------------------------------------------------
# Beam Integration (Lobatto, 5 integration points)
# --------------------------------------------------
ops.beamIntegration('Lobatto', 1, 1, 5)   # intTag 1 -> column section
ops.beamIntegration('Lobatto', 2, 2, 5)   # intTag 2 -> beam section

# --------------------------------------------------
# Geometric Transformations
# --------------------------------------------------
ops.geomTransf('PDelta', 1, 1, 0, 0)  # columns (P-Delta for seismic)
ops.geomTransf('Linear', 2, 0, 0, 1)  # X beams
ops.geomTransf('Linear', 3, 0, 0, 1)  # Y beams

# --------------------------------------------------
# Elements  (dispBeamColumn with fiber sections)
# Track tags for the design checker
# --------------------------------------------------
eleTag    = 1
col_tags  = []
beam_tags = []

# Columns
for k in range(numFloor):
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            nI = node_tag(k, i, j)
            nJ = node_tag(k + 1, i, j)
            ops.element('dispBeamColumn', eleTag, nI, nJ, 1, 1)
            col_tags.append(eleTag)
            eleTag += 1

# Beams in X
for k in range(1, numFloor + 1):
    for j in range(numBayY + 1):
        for i in range(numBayX):
            nI = node_tag(k, i, j)
            nJ = node_tag(k, i + 1, j)
            ops.element('dispBeamColumn', eleTag, nI, nJ, 2, 2)
            beam_tags.append(eleTag)
            eleTag += 1

# Beams in Y
for k in range(1, numFloor + 1):
    for j in range(numBayY):
        for i in range(numBayX + 1):
            nI = node_tag(k, i, j)
            nJ = node_tag(k, i, j + 1)
            ops.element('dispBeamColumn', eleTag, nI, nJ, 3, 2)
            beam_tags.append(eleTag)
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

Pnode = -10.0  # kip downward per elevated node
for k in range(1, numFloor + 1):
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            n = node_tag(k, i, j)
            ops.load(n, 0.0, 0.0, Pnode, 0.0, 0.0, 0.0)

# --------------------------------------------------
# Static Gravity Analysis  (10 increments for nonlinear convergence)
# --------------------------------------------------
ops.system('BandGeneral')
ops.constraints('Transformation')
ops.numberer('RCM')
ops.test('NormDispIncr', 1e-6, 50)
ops.algorithm('Newton')
ops.integrator('LoadControl', 0.1)   # 10 steps of 10% load
ops.analysis('Static')

ok = ops.analyze(10)

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

# --------------------------------------------------
# ACI 318-19 Design Check  (gravity + lateral demand)
# --------------------------------------------------
design_results = run_checks(col_tags, beam_tags)
col_fail, beam_fail = print_summary(design_results, verbose=False)
# Set verbose=True above to print every element's DCR, not just failures.

# --------------------------------------------------
# Modal Analysis
# --------------------------------------------------
numModes = numFloor + 2

lam = ops.eigen(numModes)
periods = [2 * math.pi / math.sqrt(x) for x in lam]

print("\nPeriods:")
for i, T in enumerate(periods, start=1):
    print(f"Mode {i}: T = {T:.6f} sec")

print("\nRoof-node eigenvector components:")
for mode in range(1, len(lam) + 1):
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


# Deformed shape from gravity + lateral
opsv.plot_defo(sfac=1)
plt.title("Deformed Shape (Gravity + Lateral, scaled)")
plt.tight_layout()

# Mode shapes
mode_scale = 100

for mode in range(1, numFloor+3):
    opsv.plot_mode_shape(mode, sfac=mode_scale)
    plt.title(f"Mode Shape {mode} (scaled)")
    plt.tight_layout()

plt.show()