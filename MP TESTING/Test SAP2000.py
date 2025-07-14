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

#Apply Load
#Add Load Pattern
LTYPE_OTHER = 8
SapModel.LoadPatterns.Add("OTHER", LTYPE_OTHER)

#SetLoadForce (Joint Location, LoadType, [ForceX, ForceY, ForceZ, MomentX, MomentY, MomentZ]

#Apply Load in X Direction

SapModel.PointObj.SetLoadForce("3","OTHER", [150, 150, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("6","OTHER", [150, 0, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("9","OTHER", [150, 0, 0, 0, 0, 0])

#Apply Load in Y Direction
SapModel.PointObj.SetLoadForce("12","OTHER", [0, 150, 0, 0, 0, 0])
SapModel.PointObj.SetLoadForce("21","OTHER", [0, 150, 0, 0, 0, 0])



# Save and Run Analysis
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()

# Extract Results

# ADD ^^^

# Exit SAP2000
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None
