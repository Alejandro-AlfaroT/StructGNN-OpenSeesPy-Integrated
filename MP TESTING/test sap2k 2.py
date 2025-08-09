import comtypes.client
import os
import numpy as np

# === 1. Launch and Initialize SAP2000 ===
helper = comtypes.client.CreateObject('SAP2000v1.Helper')
mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
mySapObject.ApplicationStart()
SapModel = mySapObject.SapModel

# Units: 4 = kip_ft_F
kip_ft_F = 4
SapModel.InitializeNewModel(kip_ft_F)

# === 2. Create 3D Frame Template ===
template_type = 2  # BeamSlab
NumStory = 2
StoryHeight = 12
SpansX = 4
LengthX = 20
SpansY = 4
LengthY = 20

SapModel.File.New3DFrame(template_type, NumStory, StoryHeight, SpansX, LengthX, SpansY, LengthY)

# === 3. Add Load Pattern and Apply Loads ===
LTYPE = 8  # Other
SapModel.LoadPatterns.Add("TOP_LOAD", LTYPE)
ForceX = 90
ForceY = 150

# Automatically find top-story joints
output = SapModel.PointObj.GetNameList()
ret = output[0]
PointNames = output[1]

# Determine top Z level
max_z = -999
for name in PointNames:
    ret, x, y, z = SapModel.PointObj.GetCoordCartesian(name)
    if z > max_z:
        max_z = z

top_joints = []
for name in PointNames:
    ret, x, y, z = SapModel.PointObj.GetCoordCartesian(name)
    if np.isclose(z, max_z):
        top_joints.append(name)

# Apply loads to top-story joints
for name in top_joints:
    fx = ForceX if int(name) % 2 == 0 else 0
    fy = ForceY if int(name) % 2 != 0 else 0
    SapModel.PointObj.SetLoadForce(name, "TOP_LOAD", [fx, fy, 0, 0, 0, 0])

# === 4. Save and Run Analysis ===
ModelPath = os.path.join(os.getcwd(), '3DFrame.sdb')
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()

# === 5. Extract and Print Results ===

SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
SapModel.Results.Setup.SetCaseSelectedForOutput("TOP_LOAD")

print("\n=== Joint Displacements (Top Story) ===")
for joint in top_joints:
    ret, Obj, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3 = SapModel.Results.JointDispl(joint, 0)
    if len(U1) > 0:
        print(f"Joint {joint}: UX={U1[0]:.4f}, UY={U2[0]:.4f}, UZ={U3[0]:.4f}")

print("\n=== Joint Reactions (Supports) ===")
# Get support joints (lowest Z)
output = SapModel.PointObj.GetNameList()
ret = output[0]
PointNames = output[1]

min_z = 1e6
for name in PointNames:
    ret, x, y, z = SapModel.PointObj.GetCoordCartesian(name)
    if z < min_z:
        min_z = z

support_joints = []
for name in PointNames:
    ret, x, y, z = SapModel.PointObj.GetCoordCartesian(name)
    if np.isclose(z, min_z):
        support_joints.append(name)

for joint in support_joints:
    ret, Obj, ACase, StepType, StepNum, F1, F2, F3, M1, M2, M3 = SapModel.Results.JointReact(joint, 0)
    if len(F1) > 0:
        print(f"Joint {joint}: FX={F1[0]:.2f}, FY={F2[0]:.2f}, FZ={F3[0]:.2f}")

print("\n=== Frame Internal Forces ===")
output = SapModel.FrameObj.GetNameList()
ret = output[0]
FrameNames = output[1]

for frame in FrameNames[:10]:  # Limit to first 10 for brevity
    ret, Obj, Elm, ACase, StepType, StepNum, ObjSta, ElmSta, P, V2, V3, T, M2, M3 = SapModel.Results.FrameForce(frame, 0)
    if len(P) > 0:
        print(f"Frame {frame}:")
        for i in range(len(P)):
            print(f"  Sta={ObjSta[i]:.2f} -> P={P[i]:.2f}, V2={V2[i]:.2f}, V3={V3[i]:.2f}, M3={M3[i]:.2f}")

# === 6. Clean Up and Exit ===
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None



#Extract Displacement Results
joint_names = []
ret, joint_names = SapModel.PointObj.GetNameList()

displacements = {}

for joint in joint_names:
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    SapModel.Results.Setup.SetCaseSelectedForOutput("LoadCaseName")

    NumberResults = 0
    Obj = []
    Elm = []
    ACase = []
    StepType = []
    StepNum = []
    U1 = []
    U2 = []
    U3 = []
    R1 = []
    R2 = []
    R3 = []

    [NumberResults, Obj, Elm, ACase, StepType, StepNum,
     U1, U2, U3, R1, R2, R3, ret] = SapModel.Results.JointDispl(
        joint, 0, NumberResults, Obj, Elm, ACase, StepType, StepNum,
        U1, U2, U3, R1, R2, R3
    )

    if NumberResults > 0:
        displacements[joint] = {
            'U1': U1[0], 'U2': U2[0], 'U3': U3[0],
            'R1': R1[0], 'R2': R2[0], 'R3': R3[0]
        }


'''
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
'''