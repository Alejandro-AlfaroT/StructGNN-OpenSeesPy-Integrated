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
# Format: New3DFrame(template_type, spansZ, LengthZ, spansX, LengthX, SpansY, LengthY)
ret = SapModel.File.New3DFrame(BeamSlab, 2, 12, 2, 20, 2, 20)

# Optionally save and close
ModelPath = os.path.join(os.getcwd(), '3DFrame.sdb')

# Apply a point load of 5 kip in global X direction
def apply_top_level_load(SapModel, load_pattern="TopLoad", horizontal_load=5.0):
    """
    Applies a horizontal point load to all joints at the top level (highest Z-coordinate).

    Parameters:
    - SapModel: The active SAP2000 model object.
    - load_pattern (str): Name of the load pattern to create/use.
    - horizontal_load (float): Load magnitude in the Z direction (negative = downward).
    """
    from ctypes import c_int, c_double, POINTER, pointer
    from comtypes import BSTR

    # Get all point names
    NumberNames = c_int()
    PointNames = POINTER(BSTR)()
    SapModel.PointObj.GetNameList(pointer(NumberNames), pointer(PointNames))
    names = [PointNames[i] for i in range(NumberNames.value)]

    # Find max Z and store coordinates
    top_z = -float('inf')
    point_coords = {}

    for name in names:
        x = c_double()
        y = c_double()
        z = c_double()
        SapModel.PointObj.GetCoordCartesian(name, pointer(x), pointer(y), pointer(z))
        point_coords[name] = (x.value, y.value, z.value)
        if z.value > top_z:
            top_z = z.value

    # Identify top-level joints
    tolerance = 1e-6
    top_level_points = [name for name, (x, y, z) in point_coords.items() if abs(z - top_z) < tolerance]

    # Create load pattern if not existing
    existing_patterns = [SapModel.LoadPatterns.GetName(i)[0] for i in range(SapModel.LoadPatterns.Count())]
    if load_pattern not in existing_patterns:
        LTYPE_OTHER = 8
        SapModel.LoadPatterns.Add(load_pattern, LTYPE_OTHER, 0, True)

    # Apply horizontal point load at each top-level joint
    for pt in top_level_points:
        SapModel.PointObj.SetLoadForce(pt, load_pattern, [0, 0, horizontal_load, 0, 0, 0])

    print(f"Applied {horizontal_load} kip horizontal load to {len(top_level_points)} top-level joints under pattern '{load_pattern}'.")


# Save and Run Analysis
SapModel.File.Save(ModelPath)
SapModel.Analyze.RunAnalysis()

# Extract Results

# ADD ^^^

# Exit SAP2000
mySapObject.ApplicationExit(False)
SapModel = None
mySapObject = None
