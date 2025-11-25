# MP_TESTING/Test_SAP2000.py
import os
import csv
import numpy as np
import collections
import comtypes.client


# ============================================================
#  START / ATTACH TO SAP2000
# ============================================================
def start_sap(attach_to_running: bool = False, visible: bool = True):
    """
    Starts a new SAP2000 instance (or attaches) and returns (mySapObject, SapModel).
    """
    helper = comtypes.client.CreateObject("SAP2000v1.Helper")
    mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")

    # For simplicity we always start a new instance here.
    # (attach_to_running is kept for possible future use)
    mySapObject.ApplicationStart(visible)
    SapModel = mySapObject.SapModel
    return mySapObject, SapModel


# ============================================================
#  GEOMETRY PREPARATION (called ONCE per unique frame geometry)
# ============================================================
def prepare_geometry(
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
):
    """
    Builds the 3D frame for a given geometry and returns geometry info
    needed to apply loads efficiently for many load cases.

    Returns:
        geom_info: dict with keys:
            - "top_joints": list of joint names at top story
            - "min_x": float
            - "min_y": float
    """
    # Fresh blank model
    SapModel.File.NewBlank()
    SapModel.InitializeNewModel(Units)

    # Create the 3D frame
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
        Column,
    )

    # Load pattern used for all runs
    SapModel.LoadPatterns.Add("TOP_LOAD", LType)

    # ---- Find top-story joints (by max Z) ----
    count, point_names, ret = SapModel.PointObj.GetNameList()

    max_z = -1e9
    for name in point_names:
        x, y, z, _ = SapModel.PointObj.GetCoordCartesian(name)
        if z > max_z:
            max_z = z

    top_joints = []
    for name in point_names:
        x, y, z, _ = SapModel.PointObj.GetCoordCartesian(name)
        if np.isclose(z, max_z):
            top_joints.append(name)

    # Find min-x and min-y among top joints (load lines)
    min_x = float("inf")
    min_y = float("inf")
    for name in top_joints:
        x, y, z, _ = SapModel.PointObj.GetCoordCartesian(name)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    geom_info = {
        "top_joints": top_joints,
        "min_x": min_x,
        "min_y": min_y,
    }
    return geom_info


# ============================================================
#  RUN ANALYSIS FOR ONE LOAD CASE (reused many times per geometry)
# ============================================================
def run_analysis_for_loads(
    SapModel,
    geom_info,
    ForceX,
    ForceY,
):
    """
    Uses an already-built geometry (prepare_geometry) and:
      - applies ForceX/ForceY to top-line joints
      - runs analysis
      - writes joint_displacements.csv and frame_joint_forces.csv
    """
    top_joints = geom_info["top_joints"]
    min_x = geom_info["min_x"]
    min_y = geom_info["min_y"]

    # -----------------------------------------
    # Apply loads to top-story joints
    # (overwrites previous values for TOP_LOAD)
    # -----------------------------------------
    for name in top_joints:
        x, y, z, _ = SapModel.PointObj.GetCoordCartesian(name)

        # X line
        if np.isclose(x, min_x):
            SapModel.PointObj.SetLoadForce(
                name,
                "TOP_LOAD",
                [ForceX, 0.0, 0.0, 0.0, 0.0, 0.0],
            )

        # Y line
        if np.isclose(y, min_y):
            SapModel.PointObj.SetLoadForce(
                name,
                "TOP_LOAD",
                [0.0, ForceY, 0.0, 0.0, 0.0, 0.0],
            )

    # -----------------------------------------
    # Run analysis
    # -----------------------------------------
    model_path = os.path.join(os.getcwd(), "3DFrame.sdb")
    SapModel.File.Save(model_path)
    SapModel.Analyze.RunAnalysis()

    # -----------------------------------------
    # Joint displacements → joint_displacements.csv
    # -----------------------------------------
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    SapModel.Results.Setup.SetCaseSelectedForOutput("TOP_LOAD")

    count, joint_names, ret = SapModel.PointObj.GetNameList()

    with open("joint_displacements.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Joint Name",
                "Displacement X",
                "Displacement Y",
                "Displacement Z",
            ]
        )

        for joint in joint_names:
            jname = str(joint)

            NumberResults = 0
            Obj = []
            Elm = []
            Case = []
            StepType = []
            StepNum = []
            U1 = []
            U2 = []
            U3 = []

            (
                NumberResults,
                Obj,
                Elm,
                Case,
                StepType,
                StepNum,
                U1,
                U2,
                U3,
                *_,
            ) = SapModel.Results.JointDispl(
                jname,
                0,
                NumberResults,
                Obj,
                Elm,
                Case,
                StepType,
                StepNum,
                U1,
                U2,
                U3,
                [],
                [],
                [],
            )

            for i in range(NumberResults):
                writer.writerow([jname, U1[i], U2[i], U3[i]])

    # -----------------------------------------
    # Frame joint forces → frame_joint_forces.csv
    # -----------------------------------------
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    SapModel.Results.Setup.SetCaseSelectedForOutput("TOP_LOAD")

    count, frame_names, ret = SapModel.FrameObj.GetNameList()

    with open("frame_joint_forces.csv", "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "Obj",
                "Elm",
                "I/J",
                "Case",
                "StepType",
                "StepNum",
                "F1",
                "F2",
                "F3",
                "M1",
                "M2",
                "M3",
            ]
        )

        joint_sum = collections.defaultdict(
            lambda: {
                "F1": 0.0,
                "F2": 0.0,
                "F3": 0.0,
                "M1": 0.0,
                "M2": 0.0,
                "M3": 0.0,
            }
        )

        for frame_name in frame_names:

            i_joint, j_joint, ret2 = SapModel.FrameObj.GetPoints(frame_name)

            NumberResults = 0
            Obj = []
            Elm = []
            PointElm = []
            Case = []
            StepType = []
            StepNum = []
            F1 = []
            F2 = []
            F3 = []
            M1 = []
            M2 = []
            M3 = []

            (
                NumberResults,
                Obj,
                Elm,
                PointElm,
                Case,
                StepType,
                StepNum,
                F1,
                F2,
                F3,
                M1,
                M2,
                M3,
                *_,
            ) = SapModel.Results.FrameJointForce(
                frame_name,
                0,
                NumberResults,
                Obj,
                Elm,
                PointElm,
                Case,
                StepType,
                StepNum,
                F1,
                F2,
                F3,
                M1,
                M2,
                M3,
            )

            for i in range(NumberResults):
                writer.writerow(
                    [
                        Obj[i],
                        Elm[i],
                        PointElm[i],
                        Case[i],
                        StepType[i],
                        StepNum[i],
                        F1[i],
                        F2[i],
                        F3[i],
                        M1[i],
                        M2[i],
                        M3[i],
                    ]
                )

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


# ============================================================
#  COMPATIBILITY WRAPPER (single-run, like before)
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
    ForceY,
):
    """
    Backwards-compatible wrapper:
    builds geometry + runs analysis for one load case.
    """
    geom = prepare_geometry(
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
    )
    run_analysis_for_loads(SapModel, geom, ForceX, ForceY)
