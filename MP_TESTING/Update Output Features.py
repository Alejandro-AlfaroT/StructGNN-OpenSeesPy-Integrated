import os
import torch
from torch_geometric.data import Data  # only needed for type hints

# 1) Get current script’s directory (StructGNN/MP TESTING)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2) Go up one level to StructGNN
structgnn_dir = os.path.dirname(script_dir)

# 3) Build path to the input file inside Data_Generated/structure_1
path_in = os.path.join(
    structgnn_dir,
    "Data_Generated",
    "structure_1",
    "structure_graph_NodeAsNode.pt"
)

# 4) Decide where to save modified file (StructGNN/MP TESTING)
path_out = os.path.join(script_dir, "graph_modified.pt")

# 5) Load
data: Data = torch.load(path_in)
assert hasattr(data, "y"), "This graph has no 'y' tensor."

num_nodes = data.num_nodes
print("num_nodes:", num_nodes, "| old y shape:", tuple(data.y.shape))

# 6) Declare each variable (easy to edit later)
UX = 1
UY = 1

MZ1 = 0
MZ2 = 0
MZ3 = 0
MZ4 = 0
MZ5 = 0
MZ6 = 0

MY1 = 0
MY2 = 0
MY3 = 0
MY4 = 0
MY5 = 0
MY6 = 0

VZ1 = 0
VZ2 = 0
VZ3 = 0
VZ4 = 0
VZ5 = 0
VZ6 = 0

VY1 = 0
VY2 = 0
VY3 = 0
VY4 = 0
VY5 = 0
VY6 = 0

# 7) Build new y tensor
new_y = [
    UX, UY,
    MZ1, MZ2, MZ3, MZ4, MZ5, MZ6,
    MY1, MY2, MY3, MY4, MY5, MY6,
    VZ1, VZ2, VZ3, VZ4, VZ5, VZ6,
    VY1, VY2, VY3, VY4, VY5, VY6
]

# make it shape [num_nodes, 26]
data.y = torch.tensor([new_y] * num_nodes, dtype=torch.float)

# 8) Save
torch.save(data, path_out)
print("new y shape:", tuple(data.y.shape), "→ saved to", path_out)
