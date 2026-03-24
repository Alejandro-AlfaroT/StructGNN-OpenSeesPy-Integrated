import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt

ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

# -------------------------
# GEOMETRY / PROPERTIES
# -------------------------
colL = 6.0
girL = 8.0

n_col_lines = 8     # number of vertical grid lines
n_levels = 12       # number of node levels

Acol, Agir = 2.0e-3, 6.0e-3
IzCol, IzGir = 1.6e-5, 5.4e-5
E = 200.0e9

Px = 4.0e3
Py = -3.0e3
M = -150.0
Wy = -10.0e3
Wx = 0.0

# -------------------------
# HELPER FUNCTION
# -------------------------
def node_tag(ix, iy):
    """ix = column line index, iy = level index"""
    return ix * n_levels + iy + 1

# -------------------------
# CREATE NODES
# -------------------------
for ix in range(n_col_lines):
    x = ix * girL
    for iy in range(n_levels):
        y = iy * colL
        ops.node(node_tag(ix, iy), x, y)

# supports at base nodes
for ix in range(n_col_lines):
    ops.fix(node_tag(ix, 0), 1, 1, 1)

opsv.plot_model()
plt.title('plot_model before defining elements')

# -------------------------
# GEOMETRIC TRANSFORMATION
# -------------------------
ops.geomTransf('Linear', 1)

# -------------------------
# CREATE COLUMN ELEMENTS
# -------------------------
eleTag = 1

for ix in range(n_col_lines):
    for iy in range(n_levels - 1):
        n1 = node_tag(ix, iy)
        n2 = node_tag(ix, iy + 1)
        ops.element('elasticBeamColumn', eleTag, n1, n2, Acol, E, IzCol, 1)
        eleTag += 1

# -------------------------
# CREATE BEAM ELEMENTS
# -------------------------
for iy in range(1, n_levels):   # no beam at base level
    for ix in range(n_col_lines - 1):
        n1 = node_tag(ix, iy)
        n2 = node_tag(ix + 1, iy)
        ops.element('elasticBeamColumn', eleTag, n1, n2, Agir, E, IzGir, 1)
        eleTag += 1

# -------------------------
# LOADS
# -------------------------
ops.timeSeries('Constant', 1)
ops.pattern('Plain', 1, 1)

ops.load(node_tag(0, 1), Px, 0.0, 0.0)   # node 2
ops.load(node_tag(0, 4), Px, 0.0, 0.0)   # node 5
ops.load(node_tag(0, 23), 0.0, Py, 0.0)   # node 6
ops.load(node_tag(0, 0), 0.0, 0.0, M)    # node 1
ops.load(node_tag(0, 9), Px, 0.0, 0.0)

for etag in [3, 6, 8, 10]:
    ops.eleLoad('-ele', etag, '-type', 'beamUniform', Wy, Wx)

# -------------------------
# ANALYSIS
# -------------------------
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-6, 6, 2)
ops.algorithm('Linear')
ops.integrator('LoadControl', 1)
ops.analysis('Static')
ops.analyze(1)

ops.printModel()

# -------------------------
# PLOTS
# -------------------------

opsv.plot_model()
plt.title('plot_model after defining elements')

opsv.plot_load()
plt.title("Load Scenario")

opsv.plot_defo()
plt.title("Structure Deformation")

sfacN, sfacV, sfacM = 5.0e-5, 5.0e-5, 5.0e-5

opsv.section_force_diagram_2d('N', sfacN)
plt.title("Axial force Distribution")

opsv.section_force_diagram_2d('T', sfacV)
plt.title("Shear force Distribution")

opsv.section_force_diagram_2d('M', sfacM)
plt.title("Bending Moment Distribution")

plt.show()