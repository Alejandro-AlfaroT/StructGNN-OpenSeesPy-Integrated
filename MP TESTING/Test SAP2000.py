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
#SetLoadForce (Joint Name, LoadType, [ForceX, ForceY, ForceZ, MomentX, MomentY, MomentZ]
SapModel.PointObj.SetLoadForce("3","TOP_LOAD", [ForceX, ForceY, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("6","TOP_LOAD", [ForceX, 0, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("9","TOP_LOAD", [ForceX, 0, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("12","TOP_LOAD", [ForceX, 0, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("15","TOP_LOAD", [ForceX, 0, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("18","TOP_LOAD", [0, ForceY, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("33","TOP_LOAD", [0, ForceY, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("48","TOP_LOAD", [0, ForceY, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("63","TOP_LOAD", [0, ForceY, 0, 0, 0, 0])

#^^Need to make a function to so it sets loads on the top floor
#Can do this by writing a function to select the top layer by finding all points with the highest Z value, then choose select the options with the smallest X/Y value so it only chooses the edges of the top floor

# Save and Run Analysis
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()


# Select the load case to extract results from
load_case = "TOP_LOAD"
SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
SapModel.Results.Setup.SetCaseSelectedForOutput(load_case)

# Get all joint names
joint_names = SapModel.PointObj.GetNameList()

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

        if NumberResults > 0:
            writer.writerow([joint, U1[0], U2[0], U3[0]])


# Close SAP2000
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None
