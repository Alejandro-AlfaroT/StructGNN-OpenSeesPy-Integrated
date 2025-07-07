import os
import sys
import comtypes.client
import comtypes.gen.SAP2000v1
from ctypes import c_int, POINTER, c_double, pointer, cast, c_long
from comtypes import BSTR

# Settings
AttachToInstance = False
SpecifyPath = False
ProgramPath = r'C:\Program Files\Computers and Structures\SAP2000 24\SAP2000.exe'
APIPath = r'C:\CSiAPIexample'

# Create output directory if needed
if not os.path.exists(APIPath):
    try:
        os.makedirs(APIPath)
    except OSError:
        pass

ModelPath = os.path.join(APIPath, 'API_1-001.sdb')

# Create API helper object
helper = comtypes.client.CreateObject('SAP2000v1.Helper')
helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)

# Start or attach to SAP2000
if AttachToInstance:
    try:
        mySapObject = helper.GetObject("CSI.SAP2000.API.SapObject")
    except (OSError, comtypes.COMError):
        print("No running instance of the program found or failed to attach.")
        sys.exit(-1)
else:
    if SpecifyPath:
        try:
            mySapObject = helper.CreateObject(ProgramPath)
        except (OSError, comtypes.COMError):
            print(f"Cannot start a new instance from {ProgramPath}")
            sys.exit(-1)
    else:
        try:
            mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
        except (OSError, comtypes.COMError):
            print("Cannot start a new instance of the program.")
            sys.exit(-1)

mySapObject.ApplicationStart()
SapModel = mySapObject.SapModel

# Build the model
SapModel.InitializeNewModel()
SapModel.File.NewBlank()

# Define materials and section
MATERIAL_CONCRETE = 2
SapModel.PropMaterial.SetMaterial('CONC', MATERIAL_CONCRETE)
SapModel.PropMaterial.SetMPIsotropic('CONC', 3600, 0.2, 0.0000055)
SapModel.PropFrame.SetRectangle('R1', 'CONC', 12, 12)
SapModel.PropFrame.SetModifiers('R1', [1000, 0, 0, 1, 1, 1, 1, 1])

# Set units
kip_ft_F = 4
SapModel.SetPresentUnits(kip_ft_F)

# Add frames
FrameName1 = FrameName2 = FrameName3 = ''
FrameName1, _ = SapModel.FrameObj.AddByCoord(0, 0, 0, 0, 0, 10, FrameName1, 'R1', '1', 'Global')
FrameName2, _ = SapModel.FrameObj.AddByCoord(0, 0, 10, 8, 0, 16, FrameName2, 'R1', '2', 'Global')
FrameName3, _ = SapModel.FrameObj.AddByCoord(-4, 0, 10, 0, 0, 10, FrameName3, 'R1', '3', 'Global')

# Restraints
RestraintBase = [True, True, True, True, False, False]
RestraintTop = [True, True, False, False, False, False]
PointName1, PointName2, _ = SapModel.FrameObj.GetPoints(FrameName1, '', '')
SapModel.PointObj.SetRestraint(PointName1, RestraintBase)
PointName1, PointName2, _ = SapModel.FrameObj.GetPoints(FrameName2, '', '')
SapModel.PointObj.SetRestraint(PointName2, RestraintTop)

# Refresh view
SapModel.View.RefreshView(0, False)

# Load patterns
LTYPE_OTHER = 8
for i in range(1, 8):
    SapModel.LoadPatterns.Add(str(i), LTYPE_OTHER, 1 if i == 1 else 0, True)

# Load cases
PointName1, PointName2, _ = SapModel.FrameObj.GetPoints(FrameName3, '', '')
SapModel.PointObj.SetLoadForce(PointName1, '2', [0, 0, -10, 0, 0, 0])
SapModel.FrameObj.SetLoadDistributed(FrameName3, '2', 1, 10, 0, 1, 1.8, 1.8)
SapModel.PointObj.SetLoadForce(PointName2, '3', [0, 0, -17.2, 0, -54.4, 0])
SapModel.FrameObj.SetLoadDistributed(FrameName2, '4', 1, 11, 0, 1, 2, 2)
SapModel.FrameObj.SetLoadDistributed(FrameName1, '5', 1, 2, 0, 1, 2, 2, 'Local')
SapModel.FrameObj.SetLoadDistributed(FrameName2, '5', 1, 2, 0, 1, -2, -2, 'Local')
SapModel.FrameObj.SetLoadDistributed(FrameName1, '6', 1, 2, 0, 1, 0.9984, 0.3744, 'Local')
SapModel.FrameObj.SetLoadDistributed(FrameName2, '6', 1, 2, 0, 1, -0.3744, 0, 'Local')
SapModel.FrameObj.SetLoadPoint(FrameName2, '7', 1, 2, 0.5, -15, 'Local')

# Switch to k-in
kip_in_F = 3
SapModel.SetPresentUnits(kip_in_F)

# Save and run
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()

# Extract results
SapResult = [0.0] * 7
PointName1, PointName2, _ = SapModel.FrameObj.GetPoints(FrameName2, '', '')

for i in range(0,7):
      NumberResults= 0
      Obj= []
      Elm= []
      ACase= []
      StepType= []
      StepNum= []
      U1= []
      U2= []
      U3= []
      R1= []
      R2= []
      R3= []
      ObjectElm= 0
      ret= SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
      ret= SapModel.Results.Setup.SetCaseSelectedForOutput(str(i+1))
      if i <= 3:
          [NumberResults,Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = SapModel.Results.JointDispl(PointName2,ObjectElm, NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2,U3, R1, R2, R3)
          SapResult[i]= U3[0]
      else:
          [NumberResults,Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = SapModel.Results.JointDispl(PointName1,ObjectElm, NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2,U3, R1, R2, R3)
          SapResult[i]= U1[0]
# Close
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None

# Independent values and comparison
IndResult = [-0.02639, 0.06296, 0.06296, -0.2963, 0.3125, 0.11556, 0.00651]
PercentDiff = [(SapResult[i] / IndResult[i] - 1) if IndResult[i] != 0 else 0 for i in range(7)]

# Print results
for i in range(7):
    print(f"\nLoad Case {i + 1}:")
    print(f"SAP Result   = {SapResult[i]:.5f}")
    print(f"Expected     = {IndResult[i]:.5f}")
    print(f"% Difference = {PercentDiff[i] * 100:.2f}%")
