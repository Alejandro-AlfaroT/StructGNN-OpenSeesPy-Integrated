import openseespy.opensees as ops
import opsvis as opsv
from math import asin
import matplotlib.pyplot as plt
import math

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
fc_ksi = 4.0
Ec = 57.0 * math.sqrt(fc_ksi * 1000.0) / 1000.0  # ksi
Gc = 0.4 * Ec

# Columns
b_col = 24.0
h_col = 24.0
A_col = b_col * h_col
Iy_col = h_col * b_col**3 / 12.0
Iz_col = b_col * h_col**3 / 12.0
J_col = Iy_col + Iz_col

# Beams
b_beam = 16.0
h_beam = 24.0
A_beam = b_beam * h_beam
Iy_beam = h_beam * b_beam**3 / 12.0
Iz_beam = b_beam * h_beam**3 / 12.0
J_beam = Iy_beam + Iz_beam

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
beam_tags = []
all_elements = []

# Columns
for k in range(numFloor):
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            nI = node_tag(k, i, j)
            nJ = node_tag(k + 1, i, j)
            ops.element(
                'elasticBeamColumn', eleTag, nI, nJ,
                A_col, Ec, Gc, J_col, Iy_col, Iz_col, 1)
            all_elements.append((eleTag, nI, nJ))
            eleTag += 1

# Beams in X
for k in range(1, numFloor + 1):
    for j in range(numBayY + 1):
        for i in range(numBayX):
            nI = node_tag(k, i, j)
            nJ = node_tag(k, i + 1, j)
            ops.element(
                'elasticBeamColumn', eleTag, nI, nJ,
                A_beam, Ec, Gc, J_beam, Iy_beam, Iz_beam, 2)
            beam_tags.append(eleTag)
            all_elements.append((eleTag, nI, nJ))
            eleTag += 1

# Beams in Y
for k in range(1, numFloor + 1):
    for j in range(numBayY):
        for i in range(numBayX + 1):
            nI = node_tag(k, i, j)
            nJ = node_tag(k, i, j + 1)
            ops.element(
                'elasticBeamColumn', eleTag, nI, nJ,
                A_beam, Ec, Gc, J_beam, Iy_beam, Iz_beam, 3)
            beam_tags.append(eleTag)
            all_elements.append((eleTag, nI, nJ))
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
m = 5.0  # kip-sec^2/in placeholder

for k in range(1, numFloor + 1):
    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            n = node_tag(k, i, j)
            ops.mass(n, m, m, 1e-8, 0.0, 0.0, 0.0)

# --------------------------------------------------
# Plotting Definitions
# --------------------------------------------------
def get_node_coords(tag):
    c = ops.nodeCoord(tag)
    return c[0], c[1], c[2]

def plot_fixed_bases(ax, size=80):
    x_fix, y_fix, z_fix = [], [], []

    for j in range(numBayY + 1):
        for i in range(numBayX + 1):
            n = node_tag(0, i, j)  # base nodes only
            x, y, z = get_node_coords(n)
            x_fix.append(x)
            y_fix.append(y)
            z_fix.append(z)

    ax.scatter(x_fix, y_fix, z_fix, c='k', marker='s', s=size)

def plot_structure(ax, deformed=False, scale=1.0, mode=None, title="Structure"):
    for _, nI, nJ in all_elements:
        x1, y1, z1 = get_node_coords(nI)
        x2, y2, z2 = get_node_coords(nJ)

        if deformed and mode is None:
            dx1 = ops.nodeDisp(nI, 1) * scale
            dy1 = ops.nodeDisp(nI, 2) * scale
            dz1 = ops.nodeDisp(nI, 3) * scale

            dx2 = ops.nodeDisp(nJ, 1) * scale
            dy2 = ops.nodeDisp(nJ, 2) * scale
            dz2 = ops.nodeDisp(nJ, 3) * scale

        elif mode is not None:
            dx1 = ops.nodeEigenvector(nI, mode, 1) * scale
            dy1 = ops.nodeEigenvector(nI, mode, 2) * scale
            dz1 = ops.nodeEigenvector(nI, mode, 3) * scale

            dx2 = ops.nodeEigenvector(nJ, mode, 1) * scale
            dy2 = ops.nodeEigenvector(nJ, mode, 2) * scale
            dz2 = ops.nodeEigenvector(nJ, mode, 3) * scale
        else:
            dx1 = dy1 = dz1 = 0.0
            dx2 = dy2 = dz2 = 0.0

        ax.plot(
            [x1 + dx1, x2 + dx2],
            [y1 + dy1, y2 + dy2],
            [z1 + dz1, z2 + dz2],
            'b-' if not deformed and mode is None else 'r-',
            linewidth=1.2
        )
    plot_fixed_bases(ax)
    ax.set_title(title)
    ax.set_xlabel("X (in)")
    ax.set_ylabel("Y (in)")
    ax.set_zlabel("Z (in)")
    ax.set_box_aspect([1, 1, 1])

def plot_loads(ax, show_gravity=True, show_lateral=True, grav_scale=20, lat_scale=20):
    # draw undeformed structure first
    plot_structure(ax, deformed=False, title="Applied Loads")

    # gravity loads at all elevated nodes
    if show_gravity:
        for k in range(1, numFloor + 1):
            for j in range(numBayY + 1):
                for i in range(numBayX + 1):
                    n = node_tag(k, i, j)
                    x, y, z = get_node_coords(n)
                    ax.quiver(x, y, z, 0, 0, -grav_scale, color='g', arrow_length_ratio=0.2)

    # lateral loads only at diaphragm master nodes
    if show_lateral:
        for k in range(1, numFloor + 1):
            master = node_tag(k, numBayX // 2, numBayY // 2)
            x, y, z = get_node_coords(master)
            ax.quiver(x, y, z, lat_scale, 0, 0, color='m', arrow_length_ratio=0.2)

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
# Plots
# --------------------------------------------------

# 1. Undeformed structure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plot_structure(ax, title="Undeformed Structure")
plt.tight_layout()

# 2. Load display
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plot_loads(ax, show_gravity=True, show_lateral=True, grav_scale=60, lat_scale=80)
plt.tight_layout()

# 3. Gravity + lateral deformed shape
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plot_structure(ax, deformed=True, scale=10, title="Deformed Shape (Gravity + Lateral, scaled)")
plt.tight_layout()

# 4. Modal shape plots
modal_scale = 100

for mode in range(1, 7):  # of modes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_structure(ax, mode=mode, scale=modal_scale, title=f"Mode Shape {mode} (scaled)")
    plt.tight_layout()

plt.show()