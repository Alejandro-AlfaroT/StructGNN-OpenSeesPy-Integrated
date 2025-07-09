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
# Format: New3DFrame(template_type, numX, XBay, numY, YBay, numStory, StoryHeight)
ret = SapModel.File.New3DFrame(BeamSlab, 3, 12, 3, 28, 2, 36)

# Optionally save and close
ModelPath = os.path.join(os.getcwd(), '3DFrame.sdb')

#Load case

#ADD LOAD CASE HERE

#Save and Run Analysis
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()

#Extract Results

#ADD ^^^

# Exit SAP2000
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None
