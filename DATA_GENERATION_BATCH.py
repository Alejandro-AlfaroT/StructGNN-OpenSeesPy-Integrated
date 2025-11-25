# DATA_GENERATION_BATCH.py (ultra-fast geometry grouping)
import itertools
import time
import traceback
from datetime import datetime
from collections import defaultdict

from MP_TESTING import Test_SAP2000
from MP_TESTING import Del_Files
from MP_TESTING import graph_generation
from MP_TESTING import Update_Output_Features


# ------------------------------
# SAP2000 fixed settings
# ------------------------------
Units = 4                 # kip-ft-F
template_type = 0         # OpenFrame
Restraint = True
Beam = "Default"
Column = "Default"
LType = 8                 # "Other"


# ------------------------------
# Parameter sweep grid
# ------------------------------
PARAM_GRID = {
    "NumStory":   [2, 3, 4, 5, 6],
    "StoryHeight":[12, 16],        # feet
    "SpansX":     [3, 5, 6],
    "LengthX":    [20, 30],        # feet
    "SpansY":     [3, 4, 6],
    "LengthY":    [20, 40],        # feet
    "ForceX":     [60, 80, 100],   # kip
    "ForceY":     [90, 120, 160],  # kip
}


def run_one(
    structure_id: int,
    SapModel,
    geom_info: dict,
    NumStory: int,
    StoryHeight: float,
    SpansX: int,
    LengthX: float,
    SpansY: int,
    LengthY: float,
    ForceX: float,
    ForceY: float,
) -> None:
    """
    Executes a single run for a given geometry & load combo.
    Geometry is already built in SAP (geom_info).
    """
    print("\n" + "=" * 80)
    print(
        f"RUN {structure_id} | "
        f"NumStory={NumStory}, StoryHeight={StoryHeight}, "
        f"SpansX={SpansX}, LengthX={LengthX}, "
        f"SpansY={SpansY}, LengthY={LengthY}, "
        f"ForceX={ForceX}, ForceY={ForceY}"
    )
    print("=" * 80)

    # 1) Build PyTorch Geometric graph for this configuration
    graph_generation.generate_structure(
        structure_id,
        NumStory,
        SpansX,
        SpansY,
        LengthX,
        LengthY,
        StoryHeight,
        ForceX,
        ForceY,
    )

    # 2) SAP2000: apply loads + run + export CSVs (reusing geometry)
    Test_SAP2000.run_analysis_for_loads(
        SapModel,
        geom_info,
        ForceX,
        ForceY,
    )

    # 3) Update node/edge outputs inside the graph from the generated CSVs
    Update_Output_Features.update_output(structure_id=structure_id)

    # 4) Clean up CSVs if desired (no-op or minimal)
    Del_Files.cleanup_SAP2000()   # currently disabled in your Del_Files
    Del_Files.cleanup_csv()       # you disabled this too; safe to leave


def main() -> None:
    # ------------------------------
    # Start SAP2000 once
    # ------------------------------
    mySapObject, SapModel = Test_SAP2000.start_sap(
        attach_to_running=False,
        visible=True,
    )

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*(PARAM_GRID[k] for k in keys)))

    # Build a list of run configs with IDs
    runs = []
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        params["structure_id"] = i
        runs.append(params)

    print("\n" + "=" * 80)
    print(f"Starting batch run with {len(runs)} total combinations.")
    batch_start_dt = datetime.now()
    print(f"Batch started at: {batch_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # ------------------------------
    # Group runs by GEOMETRY ONLY
    # ------------------------------
    geom_groups = defaultdict(list)
    for p in runs:
        geom_key = (
            p["NumStory"],
            p["StoryHeight"],
            p["SpansX"],
            p["LengthX"],
            p["SpansY"],
            p["LengthY"],
        )
        geom_groups[geom_key].append(p)

    total_runs = 0
    start_time = time.time()

    # ------------------------------
    # Loop over each geometry group
    # ------------------------------
    for g_idx, (geom_key, group_runs) in enumerate(geom_groups.items(), start=1):
        NumStory, StoryHeight, SpansX, LengthX, SpansY, LengthY = geom_key

        print("\n" + "#" * 80)
        print(
            f"GEOMETRY GROUP {g_idx} | "
            f"NumStory={NumStory}, StoryHeight={StoryHeight}, "
            f"SpansX={SpansX}, LengthX={LengthX}, "
            f"SpansY={SpansY}, LengthY={LengthY}"
        )
        print("#" * 80)

        # --- Prepare SAP geometry ONCE for this group ---
        geom_info = Test_SAP2000.prepare_geometry(
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

        # --- Run all load combinations for this geometry ---
        for params in group_runs:
            structure_id = params["structure_id"]

            run_start = time.time()
            try:
                run_one(
                    structure_id=structure_id,
                    SapModel=SapModel,
                    geom_info=geom_info,
                    NumStory=NumStory,
                    StoryHeight=StoryHeight,
                    SpansX=SpansX,
                    LengthX=LengthX,
                    SpansY=SpansY,
                    LengthY=LengthY,
                    ForceX=params["ForceX"],
                    ForceY=params["ForceY"],
                )
                total_runs += 1
                run_end = time.time()
                print(
                    f">>> Run {structure_id} completed in "
                    f"{run_end - run_start:.2f} seconds\n"
                )

            except Exception as e:
                print("\n[ERROR] Run failed:")
                print(f"Params: {params}")
                print("Exception:", e)
                traceback.print_exc()
                # continue to the next run
                continue

    # ------------------------------
    # Close SAP2000 once at the end
    # ------------------------------
    mySapObject.ApplicationExit(False)

    total_elapsed = time.time() - start_time
    batch_end_dt = datetime.now()

    print("\n" + "-" * 80)
    print(f"Completed {total_runs} successful run(s).")
    print(
        f"Total elapsed time: {total_elapsed/60:.2f} minutes "
        f"({total_elapsed:.1f} seconds)"
    )
    print(f"Batch ended at: {batch_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Final graphs should be in: Data_SAP2000")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
