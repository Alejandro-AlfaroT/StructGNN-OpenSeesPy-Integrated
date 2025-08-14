import comtypes.client
import os
import csv
import numpy as np
import time

# --- INITIALIZATION ---
print("Initializing SAP2000 API...")
try:
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')
    mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
    mySapObject.ApplicationStart()
    SapModel = mySapObject.SapModel
    print("SAP2000 initialized successfully.")
except (OSError, comtypes.COMError):
    print("[FATAL ERROR] Unable to start SAP2000. Please ensure it is installed correctly.")
    exit()

# Set units
kip_ft_F = 4
SapModel.InitializeNewModel(kip_ft_F)
print(f"Model initialized with units set to 'kip_ft_F'.")

# --- MODEL GENERATION ---
template_type = 2
NumStory = 2
StoryHeight = 12
SpansX = 4
LengthX = 20
SpansY = 4
LengthY = 20

print("\nCreating 3D Frame...")
ret = SapModel.File.New3DFrame(template_type, NumStory, StoryHeight, SpansX, LengthX, SpansY, LengthY)
if ret == 0:
    print("3D Frame model created successfully.")
else:
    print(f"[FATAL ERROR] Failed to create 3D Frame. Error code: {ret}. Aborting.")
    mySapObject.ApplicationExit(False)
    exit()

SapModel.View.RefreshView()
time.sleep(2)
print("[INFO] Model view refreshed and script paused.")

# --- LOAD PATTERN SETUP ---
LTYPE = 8
load_pattern_name = "TOP_LOAD"
SapModel.LoadPatterns.Add(load_pattern_name, LTYPE)
print(f"\nLoad pattern '{load_pattern_name}' created.")
# --- FIX: Assign forces based on user diagram ---
Force_on_X_edge = 90  # Purple arrows
Force_on_Y_edge = 150 # Yellow arrows

# --- DATA RETRIEVAL ---
print("\n[DEBUG] Retrieving joint names...")
NumberNames = 0
PointNames = []
result = SapModel.PointObj.GetNameList(NumberNames, PointNames)
NumberNames = result[0]
PointNames = result[1]

if NumberNames > 0 and PointNames:
    print(f"[SUCCESS] Retrieved {NumberNames} joint names.")
else:
    print(f"[FATAL ERROR] Could not retrieve joint names. Items found: {NumberNames}. Aborting.")
    mySapObject.ApplicationExit(False)
    exit()

print("[DEBUG] Fetching coordinates for all joints...")
max_z = -float('inf')
all_coords = {}
for name in PointNames:
    x, y, z = 0.0, 0.0, 0.0
    coord_result = SapModel.PointObj.GetCoordCartesian(name, x, y, z)
    if isinstance(coord_result, (list, tuple)) and len(coord_result) == 4 and coord_result[3] == 0:
        x_val, y_val, z_val = coord_result[0], coord_result[1], coord_result[2]
        all_coords[name] = {'x': x_val, 'y': y_val, 'z': z_val}
        if z_val > max_z:
            max_z = z_val
    else:
        print(f"  [WARNING] Could not get coordinates for joint {name}. API returned: {coord_result}")

if not all_coords:
    print("[FATAL ERROR] Succeeded in getting joint names, but failed to get coordinates for any of them. Aborting.")
    mySapObject.ApplicationExit(False)
    exit()

print(f"[DEBUG] Determined top floor Z-level is at Z = {max_z:.2f} ft.")

# Identify all joints on the top floor
top_joints = []
for name, coords in all_coords.items():
    if np.isclose(coords['z'], max_z):
        top_joints.append(name)

# --- FIX: Find correct edges based on user diagram (min-Y and min-X) ---
print("[DEBUG] Identifying correct edges on the top floor for load application...")
min_y_top = float('inf')
min_x_top = float('inf') # Changed from max_x_top
for name in top_joints:
    coords = all_coords[name]
    if coords['y'] < min_y_top:
        min_y_top = coords['y']
    if coords['x'] < min_x_top:
        min_x_top = coords['x']

print(f"[DEBUG] Identified {len(top_joints)} joints on the top floor.")
print(f"[DEBUG] Front edge (for Y-load, yellow) is at Y = {min_y_top:.2f}")
print(f"[DEBUG] Left edge (for X-load, purple) is at X = {min_x_top:.2f}")

# Apply loads to the correct edges
print("\n[DEBUG] Applying loads to identified top floor edge joints...")
if not top_joints:
    print("[ERROR] No top joints were found. Cannot apply loads.")
else:
    for joint_name in top_joints:
        coords = all_coords[joint_name]
        load_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Check if the joint is on the LEFT edge (min X)
        if np.isclose(coords['x'], min_x_top):
            load_vector[0] = Force_on_X_edge # Apply 90 in the +X direction
            
        # Check if the joint is on the FRONT edge (min Y)
        if np.isclose(coords['y'], min_y_top):
            load_vector[1] = Force_on_Y_edge # Apply 150 in the +Y direction
            
        # Apply the load if the vector is not all zeros
        if any(v != 0.0 for v in load_vector):
            ret = SapModel.PointObj.SetLoadForce(joint_name, load_pattern_name, load_vector)
            if ret != 0:
                print(f"  [WARNING] Failed to set load for joint {joint_name}")

print("Load application logic complete.")
# --- END OF FIXES ---

# --- ANALYSIS AND RESULTS ---
ModelPath = os.path.join(os.getcwd(), 'Automated_3DFrame.sdb')
SapModel.File.Save(ModelPath)
print(f"\nModel saved to {ModelPath}")

print("Running analysis...")
SapModel.Analyze.RunAnalysis()
print("Analysis complete.")

SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
SapModel.Results.Setup.SetCaseSelectedForOutput(load_pattern_name)

csv_filename = "joint_displacements.csv"
print(f"\nExporting joint displacements for load case '{load_pattern_name}' to {csv_filename}...")
result_final = SapModel.PointObj.GetNameList(0, [])
joint_names_list = []
if isinstance(result_final, tuple) and len(result_final) >= 2:
    joint_names_list = result_final[1]

with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Joint", "U1 (X)", "U2 (Y)", "U3 (Z)"]) 
    if joint_names_list:
        for joint in joint_names_list:
            NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3 = (0, [], [], [], [], [], [], [], [])
            [ret_code, NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, _, _, _] = \
                SapModel.Results.JointDispl(joint, 0, NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, [], [], [])
            for i in range(NumberResults):
                writer.writerow([Obj[i], U1[i], U2[i], U3[i]])
print("Displacement results exported successfully.")

# --- CLEANUP ---
# mySapObject.ApplicationExit(False)
# SapModel = None
# mySapObject = None
print("\nSAP2000 will remain open for inspection.")