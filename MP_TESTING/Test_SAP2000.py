# Test_SAP2000.py (Fully synchronized batch-safe version)
import os
import csv
import numpy as np
import collections
import comtypes.client


# ============================================================
#  START SAP2000 ONCE (called from DATA_GENERATION_BATCH.py)
# ============================================================
def start_sap(attach_to_running=False, visible=True):
    """
    Starts SAP2000 once and returns (mySapObject, SapModel)
    """
    helper = comtypes.client.CreateObject("SAP2000v1.Helper")
    mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")

    # Start SAP2000 (visible=True keeps UI visible)
    mySapObject.ApplicationStart(visible)

    SapModel = mySapObject.SapModel
    return mySapObject, SapModel


# ============================================================
#  ANALYSIS FUNCTION — RUNS ONE STRUCTURE EACH CALL
# ============================================================
def SAPAnalysis(
    SapModel,
    Units,
    template_type,
    NumStory,
    StoryHeight,
    SpansX,
    LengthX,
    SpansY,
    LengthY,
    Restraint,
    Beam,
    Column,
    LType,
    ForceX,
    ForceY
):
    """
    Batch-safe version:
    - Does NOT start SAP
    - Does NOT close SAP
    - Fully wipes model each run
    - Writes CSV outputs in old format (Displacement X/Y/Z)
    """

    # ============================================================
    # 1. FULL MODEL RESET (required for persistent SAP sessions)
    # ============================================================
    SapModel.File.NewBlank()
    SapModel.InitializeNewModel(Units)

    # ============================================================
    # 2. CREATE 3D FRAME
    # ============================================================
    SapModel.File.New3DFrame(
        template_type,
        NumStory,
        StoryHeight,
        SpansX,
        LengthX,
        SpansY,
        LengthY,
        Restraint,
        Beam,
        Column
    )

    model_path = os.path.join(os.getcwd(), "3DFrame.sdb")

    # ============================================================
    # 3. LOAD PATTERN
    # ============================================================
    SapModel.LoadPatterns.Add("TOP_LOAD", LType)

    # ============================================================
    # 4. FIND TOP-STORY JOINTS
    # ============================================================
    count, point_names, ret = SapModel.PointObj.GetNameList()

    max_z = -1e9
    for name in point_names:
        x, y, z, ret2 = SapModel.PointObj.GetCoordCartesian(name)
        if z > max_z:
            max_z = z

    top_joints = []
    for name in point_names:
        x, y, z, ret2 = SapModel.PointObj.GetCoordCartesian(name)
        if np.isclose(z, max_z):
            top_joints.append(name)

    min_x = float("inf")
    min_y = float("inf")

    for name in top_joints:
        x, y, z, _ = SapModel.PointObj.GetCoordCartesian(name)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    # APPLY LOADS
    for name in top_joints:
        x, y, z, _ = SapModel.PointObj.GetCoordCartesian(name)

        if np.isclose(x, min_x):
            SapModel.PointObj.SetLoadForce(name, "TOP_LOAD",
                                           [ForceX, 0, 0, 0, 0, 0])

        if np.isclose(y, min_y):
            SapModel.PointObj.SetLoadForce(name, "TOP_LOAD",
                                           [0, ForceY, 0, 0, 0, 0])

    # ============================================================
    # 5. RUN ANALYSIS
    # ============================================================
    SapModel.File.Save(model_path)
    SapModel.Analyze.RunAnalysis()

    # ============================================================
    # 6. JOINT DISPLACEMENTS → CSV (OLD HEADERS restored)
    # ============================================================
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    SapModel.Results.Setup.SetCaseSelectedForOutput("TOP_LOAD")

    count, joint_names, ret = SapModel.PointObj.GetNameList()

    with open("joint_displacements.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Joint Name",
                         "Displacement X",
                         "Displacement Y",
                         "Displacement Z"])

        for joint in joint_names:
            jname = str(joint)

            NumberResults = 0
            Obj = []; Elm = []; Case = []
            StepType = []; StepNum = []
            U1 = []; U2 = []; U3 = []

            (
                NumberResults, Obj, Elm, Case, StepType, StepNum,
                U1, U2, U3, *_
            ) = SapModel.Results.JointDispl(
                jname,
                0,
                NumberResults,
                Obj, Elm,
                Case, StepType, StepNum,
                U1, U2, U3,
                [], [], []
            )

            for i in range(NumberResults):
                writer.writerow([jname, U1[i], U2[i], U3[i]])

    # ============================================================
    # 7. FRAME JOINT FORCES → CSV
    # ============================================================
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    SapModel.Results.Setup.SetCaseSelectedForOutput("TOP_LOAD")

    count, frame_names, ret = SapModel.FrameObj.GetNameList()

    with open("frame_joint_forces.csv", "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "Obj", "Elm", "I/J", "Case", "StepType", "StepNum",
            "F1", "F2", "F3", "M1", "M2", "M3"
        ])

        joint_sum = collections.defaultdict(
            lambda: {"F1": 0, "F2": 0, "F3": 0,
                     "M1": 0, "M2": 0, "M3": 0}
        )

        for frame_name in frame_names:

            i_joint, j_joint, ret2 = SapModel.FrameObj.GetPoints(frame_name)

            NumberResults = 0
            Obj = []; Elm = []; PointElm = []
            Case = []; StepType = []; StepNum = []
            F1 = []; F2 = []; F3 = []
            M1 = []; M2 = []; M3 = []

            (
                NumberResults, Obj, Elm, PointElm,
                Case, StepType, StepNum,
                F1, F2, F3, M1, M2, M3, *_
            ) = SapModel.Results.FrameJointForce(
                frame_name, 0,
                NumberResults,
                Obj, Elm, PointElm,
                Case, StepType, StepNum,
                F1, F2, F3, M1, M2, M3
            )

            for i in range(NumberResults):
                writer.writerow([
                    Obj[i], Elm[i], PointElm[i],
                    Case[i], StepType[i], StepNum[i],
                    F1[i], F2[i], F3[i], M1[i], M2[i], M3[i]
                ])

                if PointElm[i].upper().startswith("I"):
                    jname = i_joint
                elif PointElm[i].upper().startswith("J"):
                    jname = j_joint
                else:
                    continue

                key = (jname, Case[i], StepType[i], StepNum[i])
                agg = joint_sum[key]

                agg["F1"] += F1[i]
                agg["F2"] += F2[i]
                agg["F3"] += F3[i]
                agg["M1"] += M1[i]
                agg["M2"] += M2[i]
                agg["M3"] += M3[i]
