import torch
from torch_geometric.data import Data  # only needed for type hints

path_in  = "graph.pt"
path_out = "graph_modified.pt"

# 1) Load
data: Data = torch.load(path_in)

# 2) Sanity checks
assert hasattr(data, "y"), "This graph has no 'y' tensor."
num_nodes = data.num_nodes
print("num_nodes:", num_nodes, "| old y shape:", tuple(data.y.shape))

# 3) Build your new node-level targets (shape: [num_nodes, new_dim])
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

new_y = [
    UX,
    UY,
    MZ1,
    MZ2,
    MZ3,
    MZ4,
    MZ5,
    MZ6,
    MY1,
    MY2,
    MY3,
    MY4,
    MY5,
    MY6,
    VZ1,
    VZ2,
    VZ3,
    VZ4,
    VZ5,
    VZ6,
    VY1,
    VY2,
    VY3,
    VY4,
    VY5,
    VY6,
]


# 4) Replace
data.y = new_y

# 5) Save
torch.save(data, path_out)
print("new y shape:", tuple(data.y.shape), "→ saved to", path_out)
