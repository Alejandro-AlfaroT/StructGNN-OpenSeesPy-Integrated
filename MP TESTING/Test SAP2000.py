import comtypes.client
import os

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
SpansX = 2
LengthX = 20
SpansY = 2
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
ForceX = 150
ForceY = 150

#Apply Loads
#SetLoadForce (Joint Name, LoadType, [ForceX, ForceY, ForceZ, MomentX, MomentY, MomentZ]
SapModel.PointObj.SetLoadForce("3","TOP_LOAD", [ForceX, ForceY, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("6","TOP_LOAD", [ForceX, 0, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("9","TOP_LOAD", [ForceX, 0, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("12","TOP_LOAD", [0, ForceY, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("21","TOP_LOAD", [0, ForceY, 0, 0, 0, 0])

#^^Need to make a function to so it sets loads on the top at each joint

# Save and Run Analysis
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()

# Extract Results

 SapModel.Results.JointDispl("All", GroupElm, NumberResults, Obj, Elm, Loadcase, StepType, StepNum, U1, U2, U3, R1, R2, R3)

# Exit SAP2000
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None
