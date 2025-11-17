# batch_run.py
import itertools
import time
import traceback

from MP_TESTING import Test_SAP2000
from MP_TESTING import Del_Files
from MP_TESTING import graph_generation
from MP_TESTING import Update_Output_Features

# ------------------------------
# Fixed settings
# ------------------------------
Units = 4                 # kip-ft-F
template_type = 0         # OpenFrame
Restraint = True
Beam = "Default"
Column = "Default"
LType = 8                 # "Other"

# ------------------------------------
# Define your parameter sweep right here
# (Put one or more values in each list)
# ------------------------------------
PARAM_GRID = {
    "NumStory":   [2, 3, 4, 5, 6],
    "StoryHeight":[12, 16],            # feet
    "SpansX":     [3, 5, 6],
    "LengthX":    [20, 30],            # feet
    "SpansY":     [3, 4, 6],
    "LengthY":    [20, 40],            # feet
    "ForceX":     [60, 80, 100],        # Kip
    "ForceY":     [90, 120, 160],      # Kip
}

# If you prefer single-run defaults, set lists to one value each:
# PARAM_GRID = {
#     "NumStory":   [2],
#     "StoryHeight":[12],
#     "SpansX":     [4],
#     "LengthX":    [20],
#     "SpansY":     [4],
#     "LengthY":    [20],
#     "ForceX":     [90],
#     "ForceY":     [150],
# }

def run_one(structure_id: int,
            NumStory: int, StoryHeight: float,
            SpansX: int, LengthX: float,
            SpansY: int, LengthY: float,
            ForceX: float, ForceY: float):
    """
    Executes a single end-to-end pipeline run for one parameter combo.
    Assumes Update_Output_Features.update_output() reads the CSVs written
    by SAP2000 and writes the final graph to Modified_Graphs.
    """
    print("\n" + "="*80)
    print(f"RUN {structure_id} | "
          f"NumStory={NumStory}, StoryHeight={StoryHeight}, "
          f"SpansX={SpansX}, LengthX={LengthX}, SpansY={SpansY}, LengthY={LengthY}, "
          f"ForceX={ForceX}, ForceY={ForceY}")
    print("="*80)

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
        ForceY
    )

    # 2) Run SAP2000 analysis and export CSVs
    Test_SAP2000.SAPAnalysis(
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
    )

    # 3) Update node/edge outputs inside the graph from the generated CSVs
    Update_Output_Features.update_output(structure_id=structure_id)

    # 4) Clean up extra files for this run
    Del_Files.cleanup_SAP2000()
    Del_Files.cleanup_csv()

def main():
    keys = list(PARAM_GRID.keys())
    values_product = list(itertools.product(*(PARAM_GRID[k] for k in keys)))

    total_combos = len(values_product)
    print("\n" + "="*80)
    print(f"Starting batch run with {total_combos} total combinations.")
    print("="*80 + "\n")

    total_runs = 0
    start_time = time.time()

    for i, combo in enumerate(values_product, start=1):
        params = dict(zip(keys, combo))
        try:
            run_one(
                structure_id=i,
                NumStory=params["NumStory"],
                StoryHeight=params["StoryHeight"],
                SpansX=params["SpansX"],
                LengthX=params["LengthX"],
                SpansY=params["SpansY"],
                LengthY=params["LengthY"],
                ForceX=params["ForceX"],
                ForceY=params["ForceY"],
            )
            total_runs += 1
        except Exception as e:
            print("\n[ERROR] Run failed:")
            print(f"  Params: {params}")
            print("  Exception:", e)
            traceback.print_exc()
            continue

    elapsed = time.time() - start_time
    print("\n" + "-"*80)
    print(f"Completed {total_runs} run(s) in {elapsed:.1f} seconds.")
    print("Final graphs should be in: Modified_Graphs")
    print("-"*80 + "\n")


if __name__ == "__main__":
    main()
