import comtypes.client
import os

# Set up SAP2000 helper and instance
helper = comtypes.client.CreateObject('SAP2000v1.Helper')
helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
mySapObject.ApplicationStart()
SapModel = mySapObject.SapModel

# Units: 4 = kip_ft_F
kip_ft_F = 4
SapModel.InitializeNewModel(kip_ft_F)

# Template types:
#   1 = SteelDeck, 2 = ConcreteSlab, 3 = BeamSlab (what you're using), etc.
BeamSlab = 3
# Use template to create 3D frame
# Format: New3DFrame(template_type, spansZ, LengthZ, spansX, LengthX, SpansY, LengthY)
ret = SapModel.File.New3DFrame(BeamSlab, 2, 12, 2, 20, 2, 20)

# Optionally save and close
ModelPath = os.path.join(os.getcwd(), '3DFrame.sdb')

# ------------------------------------------------------
# Add Load Pattern and Apply Point Loads at Top
# ------------------------------------------------------

LTYPE_OTHER = 8
SapModel.LoadPatterns.Add("8", LTYPE_OTHER, 0, True)

# Get all joint names (fixed line)
ret, NumberNames, JointNames = SapModel.PointObj.GetNameList()

# Identify top story joints (Z > threshold, here ~24 ft assuming 2 stories × 12 ft)
TopJointNames = []
for name in JointNames:
    ret, x, y, z = SapModel.PointObj.GetCoordCartesian(name)
    if z >= 24:
        TopJointNames.append((name, x, y))  # Keep coordinates for sorting

# Sort top joints by X-coordinate to consistently pick left/mid/right
TopJointNames.sort(key=lambda tup: tup[1])  # Sort by x

# Select 3 joints (left, center, right in X)
SelectedJoints = [tup[0] for tup in TopJointNames[:1] + TopJointNames[len(TopJointNames)//2:len(TopJointNames)//2+1] + TopJointNames[-1:]]

# Apply point load of 5 kip in +X direction to selected joints under pattern "8"
# [Force in X, Force in Y, Force in Z, Moment X, Moment Y, Moment Z]
for joint in SelectedJoints:
    SapModel.PointObj.SetLoadForce(joint, "8", [5.0, 0, 0, 0, 0, 0])

# ------------------------------------------------------

# Save and Run Analysis
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()

# Extract Results

# ADD ^^^

# Exit SAP2000
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None
