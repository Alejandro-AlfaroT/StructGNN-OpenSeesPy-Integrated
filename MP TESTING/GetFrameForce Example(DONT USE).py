import comtypes.client


##THIS ONE USES THE WRONG COMMAND, IT SHOULD BE SapObject.SapModel.Results.FrameForce ##


#dimension variables
ObjSta       = [0.0, 124.0]          # stations along the element
LoadCase     = ["EXTERNAL"]           # load case name(s)

numberStations = len(ObjSta)

P            = [5.0, 5.0]             # axial force
V2           = [-5.0, 5.0]            # shear in local 2
V3           = [0.0] * numberStations # shear in local 3
T            = [0.0] * numberStations # torsion
M2           = [0.0] * numberStations # moment about local 2
M3           = [100.0, 100.0]         # moment about local 3

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

#create load case for external results

SapModel.LoadCases.ExternalResults.SetCase("EXTERNAL")
SapModel.LoadCases.ExternalResults.SetNumberSteps("EXTERNAL", 0, 0) #Firststep=0, LastStep=0

#set cases and stations for frame external results

SapModel.ExternalAnalysisResults.PresetFrameCases("1", 1, LoadCase)
SapModel.ExternalAnalysisResults.SetFrameStations("1", ObjSta)

#set frame external result forces at case first step

SapModel.ExternalAnalysisResults.SetFrameForce("1", "EXTERNAL", 0, P, V2, V3, T, M2, M3)
SapModel.ExternalAnalysisResults.GetFrameForce("1", "EXTERNAL", 0, numberStations, P, V2, V3, T, M2, M3)


# --- read forces back ---
# NOTE: comtypes returns (ret, P, V2, V3, T, M2, M3, NumberStations)
ret, Pout, V2out, V3out, Tout, M2out, M3out, nst = SapModel.ExternalAnalysisResults.GetFrameForce(
    "1", "EXTERNAL", 0, numberStations, P, V2, V3, T, M2, M3
)

# Be defensive about nst type; fall back to array length
try:
    n = int(nst)
except (TypeError, ValueError):
    n = len(Pout)

print(f"Return code: {ret} (0 = success)")
print(f"Frame: 1   Case: EXTERNAL   Step: 0   Stations returned: {n}\n")
print(f"{'Station':>10}  {'P':>12}  {'V2':>12}  {'V3':>12}  {'T':>12}  {'M2':>12}  {'M3':>12}")

for i in range(n):
    print(f"{ObjSta[i]:10.3f}  {Pout[i]:12.3f}  {V2out[i]:12.3f}  {V3out[i]:12.3f}  "
          f"{Tout[i]:12.3f}  {M2out[i]:12.3f}  {M3out[i]:12.3f}")

#close Sap2000

SapObject.ApplicationExit(False)
SapModel = None
SapObject = None