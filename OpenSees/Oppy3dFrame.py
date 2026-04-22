import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import numpy as np
from math import asin


# -------------------------
# INITIALIZE MODEL
# -------------------------
ops.wipe()
ops.model('Basic', '-ndm', 3, '-ndf', 6)

# properties
numBayX = 8
numBayY = 10
numFloor = 16

bayWidthX = 144.0
bayWidthY = 144.0
storyHeights = [240.0, 168.0, 156.0, 156.0, 156.0, 156.0, 156.0, 156.0, 156.0, 156.0, 156.0, 156.0, 156.0, 156.0, 156.0, 156.0]

E = 29500.0
massX = 0.49
M = 0.0
coordTransf = "Linear"
massType = "-lMass"

# -------------------------
# CREATE NODES
# -------------------------
nodeTag = 1
zLoc = 0.0

for k in range(numFloor + 1):
    xLoc = 0.0
    for i in range(numBayX + 1):
        yLoc = 0.0
        for j in range(numBayY + 1):
            ops.node(nodeTag, xLoc, yLoc, zLoc)
            ops.mass(nodeTag, massX, massX, 0.01, 1.0e-10, 1.0e-10, 1.0e-10)

            if k == 0:
                ops.fix(nodeTag, 1, 1, 1, 1, 1, 1)
            else:
                ops.mass(nodeTag, massX, massX, 0.01, 1.0e-10, 1.0e-10, 1.0e-10)

            yLoc += bayWidthY
            nodeTag += 1

        xLoc += bayWidthX

    if k < numFloor:
        zLoc += storyHeights[k]

# -------------------------
# GEOMETRIC TRANSFORMATIONS
# -------------------------
ops.geomTransf(coordTransf, 1, 1, 0, 0)
ops.geomTransf(coordTransf, 2, 0, 0, 1)

# -------------------------
# CREATE COLUMNS
# -------------------------
eleTag = 1
nodeTag1 = 1
nodesPerFloor = (numBayX + 1) * (numBayY + 1)

for k in range(numFloor):
    for i in range(numBayX + 1):
        for j in range(numBayY + 1):
            nodeTag2 = nodeTag1 + nodesPerFloor
            ops.element(
                'elasticBeamColumn',
                eleTag, nodeTag1, nodeTag2,
                50.0, E, 1000.0, 1000.0, 2150.0, 2150.0,
                1, '-mass', M, massType
            )
            eleTag += 1
            nodeTag1 += 1

# -------------------------
# CREATE BEAMS IN X DIRECTION
# -------------------------
nodeTag1 = 1 + nodesPerFloor
for floor in range(1, numFloor + 1):
    for i in range(numBayX):
        for j in range(numBayY + 1):
            nodeTag2 = nodeTag1 + (numBayY + 1)
            ops.element(
                'elasticBeamColumn',
                eleTag, nodeTag1, nodeTag2,
                50.0, E, 1000.0, 1000.0, 2150.0, 2150.0,
                2, '-mass', M, massType
            )
            eleTag += 1
            nodeTag1 += 1
    nodeTag1 += (numBayY + 1)

# -------------------------
# CREATE BEAMS IN Y DIRECTION
# -------------------------
nodeTag1 = 1 + nodesPerFloor
for floor in range(1, numFloor + 1):
    for i in range(numBayY + 1):
        for j in range(numBayX):
            nodeTag2 = nodeTag1 + 1
            ops.element(
                'elasticBeamColumn',
                eleTag, nodeTag1, nodeTag2,
                50.0, E, 1000.0, 1000.0, 2150.0, 2150.0,
                2, '-mass', M, massType
            )
            eleTag += 1
            nodeTag1 += 1
        nodeTag1 += 1

# -------------------------
# EIGENVALUE ANALYSIS
# -------------------------
numEigen = 16
eigenValues = ops.eigen(numEigen)
PI = 2 * asin(1.0)

print("Eigenvalues:", eigenValues)

# -------------------------
# PLOT MODEL
# -------------------------
opsv.plot_model(node_labels=0, element_labels=0)
plt.title("Structure")

# -------------------------
# PLOT MODE SHAPE
# -------------------------
#opsv.plot_mode_shape(4, sfac=300)
#plt.title("Mode Shape" )
numModes = 16

for mode in range(1, numModes + 1):
    #plt.figure(f"Mode Shape {mode}")
    opsv.plot_mode_shape(mode, sfac=300)
    plt.title(f"Mode Shape {mode}")

plt.show()


# -------------------------
# STATIC ANALYSIS
# -------------------------
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
ops.load(72, 1, 0.0, 0.0, 0.0, 0.0)

ops.constraints('Plain')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-8, 20)
ops.algorithm('Linear')
ops.integrator('LoadControl', 0.1)
ops.analysis('Static')

ops.analyze(16)

ops.wipe()