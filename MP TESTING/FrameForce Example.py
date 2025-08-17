import comtypes.client

#dimension variables
Object = 0
NumberResults = 0
Obj      = []
ObjSta   = []
Elm      = []
ElmSta   = []
LoadCase = []
StepType = []
StepNum  = []
P        = []
V2       = []
V3       = []
T        = []
M2       = []
M3       = []

#create Sap2000 object
helper = comtypes.client.CreateObject('SAP2000v1.Helper')
SapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")


#start Sap2000 application
SapObject.ApplicationStart()

#create SapModel object
SapModel = SapObject.SapModel

#initialize model
SapModel.InitializeNewModel()

#create model from template
PortalFrame = 0
SapModel.File.New2DFrame(PortalFrame, 3, 124, 3, 200)

#run analysis
SapModel.File.Save()
SapModel.Analyze.RunAnalysis()

#clear all case and combo output selections
SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()

#set case and combo output selections
SapModel.Results.Setup.SetCaseSelectedForOutput("DEAD")

#get frame forces for line object "1"
SapModel.Results.FrameForce("1", Object, NumberResults, Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)


# Print header
print(f"{'Station':>8} {'P':>10} {'V2':>10} {'V3':>10} {'T':>10} {'M2':>10} {'M3':>10}")

# Loop through results
for i in range(len(P)):
    print(f"{ObjSta[i]:8.3f} {P[i]:10.3f} {V2[i]:10.3f} {V3[i]:10.3f} {T[i]:10.3f} {M2[i]:10.3f} {M3[i]:10.3f}")


#close Sap2000
SapObject.ApplicationExit(False)
SapModel = None
SapObject = None
