import comtypes.client
import os
import csv
import numpy as np
import collections

# Set up SAP2000 helper and instance
helper = comtypes.client.CreateObject('SAP2000v1.Helper')
mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
mySapObject.ApplicationStart()
SapModel = mySapObject.SapModel

# Units: lb_in_F = 1, lb_ft_F = 2, kip_in_F = 3, kip_ft_F = 4, kN_mm_C = 5, kN_m_C = 6, kgf_mm_C = 7, kgf_m_C = 8, N_mm_C = 9, N_m_C = 10, Ton_mm_C = 11, Ton_m_C = 12, kN_cm_C = 13, kgf_cm_C = 14, N_cm_C = 15, Ton_cm_C = 16
kip_ft_F = 4
SapModel.InitializeNewModel(kip_ft_F)

#0 = OpenFrame, 1 = PerimeterFrame, 2 = BeamSlab, 3 = FlatPlate
template_type = 0
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
LType = 8
SapModel.LoadPatterns.Add("TOP_LOAD", LType)
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

#Get Displacement Results

# Select the load case to extract results from
LoadCase = "TOP_LOAD"
SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
SapModel.Results.Setup.SetCaseSelectedForOutput(LoadCase)

# Get all joint names
count, joint_names, ret = SapModel.PointObj.GetNameList()

# Create CSV file to store displacements
with open("joint_displacements.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Joint Name", "Displacement X", "Displacement Y", "Displacement Z"
    ])

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

        (
            NumberResults, Obj, Elm, ACase, StepType, StepNum,
            U1, U2, U3, *_
        ) = SapModel.Results.JointDispl(
            str(joint), 0, NumberResults,
            Obj, Elm, ACase, StepType, StepNum,
            U1, U2, U3, [], [], []
        )
        for i in range(NumberResults):
            #[Joint name, Displacement X, Displacement Y, Displacement Z]
            writer.writerow([joint, U1[i], U2[i], U3[i]])

# Get Moment Results

# --- Frame-end forces for ALL members (every I/J point), plus joint-wise aggregation ---

# 1) Make sure the load case/combination you want is selected
SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
SapModel.Results.Setup.SetCaseSelectedForOutput("TOP_LOAD")

# 2) Get all frame objects
count, frame_names, ret = SapModel.FrameObj.GetNameList()

# CSV for raw frame-end results
with open("frame_joint_forces.csv", "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow([
        "Obj","Elm","PointEnd(I/J)","Case","StepType","StepNum",
        "F1","F2","F3","M1","M2","M3"
    ])

    # Optional: joint-wise aggregator (sum of frame-end forces at each joint)
    # key = (joint_name, case, stepType, stepNum)  value = dict of summed components
    joint_sum = collections.defaultdict(lambda: {"F1":0.0,"F2":0.0,"F3":0.0,"M1":0.0,"M2":0.0,"M3":0.0})

    for frame_name in frame_names:
        # Map I/J to actual joint names for aggregation
        ret_i, ret_j, ret = SapModel.FrameObj.GetPoints(frame_name)  # returns (iPoint, jPoint, retcode)
        i_joint, j_joint = ret_i, ret_j

        # Prepare output arrays (ALL are outputs)
        NumberResults = 0
        Obj = []
        Elm = []
        PointElm = []
        ACase = []
        StepType = []
        StepNum = []
        F1 = []
        F2 = []
        F3 = []
        M1 = []
        M2 = []
        M3 = []

        (
            NumberResults, Obj, Elm, PointElm, ACase, StepType, StepNum,
            F1, F2, F3, M1, M2, M3, *_
        ) = SapModel.Results.FrameJointForce(
            frame_name,         # Name
            0,                  # ObjectElm: 0=Objects, 1=Elements
            NumberResults, Obj, Elm, PointElm,
            ACase, StepType, StepNum, F1, F2, F3, M1, M2, M3
        )

        # Write raw rows and build joint-wise sums
        for i in range(NumberResults):
            writer.writerow([
                Obj[i], Elm[i], PointElm[i], ACase[i], StepType[i], StepNum[i],
                F1[i], F2[i], F3[i], M1[i], M2[i], M3[i]
            ])

            # Map I/J end to actual joint name, then accumulate
            if PointElm[i].upper().startswith("I"):
                joint_name = i_joint
            elif PointElm[i].upper().startswith("J"):
                joint_name = j_joint
            else:
                # Unexpected label; skip aggregation but keep CSV row
                continue

            key = (joint_name, ACase[i], StepType[i], StepNum[i])
            agg = joint_sum[key]
            agg["F1"] += F1[i]
            agg["F2"] += F2[i]
            agg["F3"] += F3[i]
            agg["M1"] += M1[i]
            agg["M2"] += M2[i]
            agg["M3"] += M3[i]


# Close SAP2000
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None
