import comtypes.client
import os
import csv
import numpy as np

# Set up SAP2000 helper and instance
helper = comtypes.client.CreateObject('SAP2000v1.Helper')
mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
mySapObject.ApplicationStart()
SapModel = mySapObject.SapModel

# Units: 4 = kip_ft_F
kip_ft_F = 4
SapModel.InitializeNewModel(kip_ft_F)

#0 = OpenFrame, 1 = PerimeterFrame, 2 = BeamSlab, 3 = FlatPlate
template_type = 2
NumStory = 2
StoryHeight = 12
SpansX = 4
LengthX = 20
SpansY = 4
LengthY = 20

#Create 3D Frame
SapModel.File.New3DFrame(template_type, NumStory, StoryHeight, SpansX, LengthX, SpansY, LengthY)

# Optionally save and close
ModelPath = os.path.join(os.getcwd(), '3DFrame.sdb')

#Set Up Load information
#Add Load Pattern
#1 = Dead, 2 = SuperDead, 3 = Live, 4 = Reduced Live, 5 = Quake, 6 = Wind, 7 = Snow, 8 = Other... Check API for others
LTYPE = 8
SapModel.LoadPatterns.Add("TOP_LOAD", LTYPE)
ForceX = 90
ForceY = 150

#Apply Loads

# Automatically find top-story joints
#setup
output = SapModel.PointObj.GetNameList()
ret = output[0]
PointNames = output[1]

# Step 1: Determine top Z level
max_z = -999
for name in PointNames:
    x, y, z, ret = SapModel.PointObj.GetCoordCartesian(name)
    if z > max_z:
        max_z = z

# Step 2: Find top-story joints
top_joints = []
for name in PointNames:
    x, y, z, ret = SapModel.PointObj.GetCoordCartesian(name)
    if np.isclose(z, max_z):
        top_joints.append(name)

# Step 3: Find min Y and min X among top joints
min_y = float('inf')
min_x = float('inf')
for name in top_joints:
    x, y, z, ret = SapModel.PointObj.GetCoordCartesian(name)
    if y < min_y:
        min_y = y
    if x < min_x:
        min_x = x

# Step 4: Apply ForceX to min-X joints, and ForceY to min-y joints
for name in top_joints:
    x, y, z, ret = SapModel.PointObj.GetCoordCartesian(name)

    if np.isclose(x, min_x):
        SapModel.PointObj.SetLoadForce(name, "TOP_LOAD", [ForceX, 0, 0, 0, 0, 0])

    if np.isclose(y, min_y):
        SapModel.PointObj.SetLoadForce(name, "TOP_LOAD", [0, ForceY, 0, 0, 0, 0])

# Save and Run Analysis
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()


# Select the load case to extract results from
load_case = "TOP_LOAD"
SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
SapModel.Results.Setup.SetCaseSelectedForOutput(load_case)

# Get all joint names
joint_names = SapModel.PointObj.GetNameList()
# ^ need to convert this into something to run a function that checks every joint, not just the last one

# Create CSV file to store displacements
with open("joint_displacements.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Joint", "U1", "U2", "U3"])  # Only displacements

    for joint in joint_names:
        # Initialize variables
        joint = str(joint)

        NumberResults = 0
        Obj = []
        Elm = []
        ACase = []
        StepType = []
        StepNum = []
        U1 = []
        U2 = []
        U3 = []

        # Only displacements: ignore rotations
        [NumberResults, Obj, Elm, ACase, StepType, StepNum,
         U1, U2, U3, *_] = SapModel.Results.JointDispl(
            joint, 0, NumberResults,
            Obj, Elm, ACase, StepType, StepNum,
            U1, U2, U3, [], [], []
        )
        for i in range(NumberResults):
            writer.writerow([joint, U1[i], U2[i], U3[i]])


# Close SAP2000
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None
