import comtypes.client
import os
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

# Save and Run Analysis
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()

# Extract Results
Axial = np.zeros(25)
Reaction = np.zeros([4,3])
Displacement = np.zeros([10,3])

SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
SapModel.Results.Setup.SetCaseSelectedForOutput('[TOP_LOAD')

# Metadata and identifiers
NumberResults = 0
Obj = []          # Object names
Elm = []          # Element IDs
ACase = []        # Analysis case names
StepType = []     # Type of analysis step (e.g., Linear, TimeHistory)
StepNum = []      # Step number

# Object/element-level result control
ObjectElm = 0     # 0 = object results, 1 = element results
ObjSta = []       # Object station positions
ElmSta = []       # Element station positions

# Internal force result containers
P = []            # Axial force
V2 = []           # Shear force in local 2-direction
V3 = []           # Shear force in local 3-direction
T = []            # Torsion
M2 = []           # Moment about local 2-axis
M3 = []           # Moment about local 3-axis

# Element-level control and end force results
Element = 1       # Request element-level results
F1 = []
F2 = []
F3 = []
R1 = []
R2 = []
R3 = []

# Group-based result extraction
GroupElm = 2      # Target group ID

# Joint displacement result containers
U1 = []           # Displacement in global X
U2 = []           # Displacement in global Y
U3 = []           # Displacement in global Z
U4 = []           # Rotation about global X
U5 = []           # Rotation about global Y
U6 = []           # Rotation about global Z


# Close SAP2000
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None
