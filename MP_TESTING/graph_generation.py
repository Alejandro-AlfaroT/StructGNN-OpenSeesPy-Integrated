import os
import torch
import numpy as np
from torch_geometric.data import Data

def generate_structure(
    structure_id,
    num_floors,
    grid_x,         # ⬅ SPANS in X (nodes_x = grid_x + 1)
    grid_z,         # ⬅ SPANS in Z (nodes_z = grid_z + 1)
    beam_length_x,  # ⬅ X-direction beams (per span)
    beam_length_z,  # ⬅ Z-direction beams (per span)
    column_length,  # ⬅ vertical members
    ForceX=90.0,    # ⬅ applied on top level to min-x nodes
    ForceY=150.0,   # ⬅ applied on top level to min-z nodes (stored in force_z feature)
    output_dir="Data_Gen_Manual",
):
    """
    Generates a building structure with independent X/Z beam lengths.
    Node IDs increase with Y fastest (then X, then Z).

    Node feature layout (11):
      [nodes_x, nodes_y, nodes_z, x, y, z, support, joint, nodal_mass, force_x, force_z]
        - nodes_x = grid_x + 1  (node count along X)
        - nodes_y = num_floors + 1
        - nodes_z = grid_z + 1  (node count along Z)

    Edge features (3):
      [beam, column, length]
    """

    # --- Dimensions ---
    y_val = num_floors + 1              # nodes along Y (a.k.a. grid_y)
    grid_y = y_val
    max_y = y_val - 1
    size_x = grid_x + 1                 # nodes along X
    size_z = grid_z + 1                 # nodes along Z

    # --- First pass: collect coordinates (Z -> X -> Y with Y fastest) ---
    coords = []
    for z in range(size_z):
        for x in range(size_x):
            for y in range(y_val):
                coords.append((x, y, z))
    coords = np.array(coords, dtype=float)

    # Top level joints (max y), then min x and min z among them
    top_mask = coords[:, 1] == max_y
    top_coords = coords[top_mask]
    min_x = top_coords[:, 0].min()
    min_z = top_coords[:, 2].min()

    # --- Nodes ---
    nodes = []
    for (x, y, z) in coords:
        support = 1.0 if y == 0 else 0.0
        joint   = 1.0 if y != 0 else 0.0
        nodal_mass = 0.0   # always zero

        # Force assignment (keep features as force_x and force_z)
        on_top = (y == max_y)
        force_x = float(ForceX) if (on_top and x == min_x) else 0.0
        force_z = float(ForceY) if (on_top and z == min_z) else 0.0

        # Store NODE COUNTS (not spans) in the first three features
        nodes.append([
            float(size_x), float(grid_y), float(size_z),  # node counts along X/Y/Z
            float(x), float(y), float(z),                 # indices as coords
            support, joint, nodal_mass,                   # flags
            force_x, force_z                              # loads
        ])

    # --- Helper: Y-fastest linear index ---
    def get_node_index(xi, zi, yi):
        return int(zi * (size_x * y_val) + xi * y_val + yi)

    # --- Edges + attributes ---
    edges = []
    edge_attributes = []  # [beam, column, length]

    # Beams (horizontal) — only for floors above ground (y >= 1)
    for y in range(1, y_val):
        for z in range(size_z):
            for x in range(size_x):
                current = get_node_index(x, z, y)

                # X-direction neighbor
                if x < size_x - 1:
                    right = get_node_index(x + 1, z, y)
                    edges.append([current, right]); edge_attributes.append([1.0, 0.0, float(beam_length_x)])
                    edges.append([right, current]); edge_attributes.append([1.0, 0.0, float(beam_length_x)])

                # Z-direction neighbor
                if z < size_z - 1:
                    depth = get_node_index(x, z + 1, y)
                    edges.append([current, depth]); edge_attributes.append([1.0, 0.0, float(beam_length_z)])
                    edges.append([depth, current]); edge_attributes.append([1.0, 0.0, float(beam_length_z)])

    # Columns (vertical)
    for y in range(y_val - 1):
        for z in range(size_z):
            for x in range(size_x):
                current = get_node_index(x, z, y)
                above = get_node_index(x, z, y + 1)
                edges.append([current, above]); edge_attributes.append([0.0, 1.0, float(column_length)])
                edges.append([above, current]); edge_attributes.append([0.0, 1.0, float(column_length)])

    # --- Graph tensors ---
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)   # (E, 3)
    node_features = torch.tensor(nodes, dtype=torch.float)         # (N, 11)

    structure_graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,    # [beam, column, length]
        y=torch.zeros(node_features.shape[0], 38)
    )

    # --- Save ---
    struct_dir = os.path.join(output_dir, f"structure_{structure_id}")
    os.makedirs(struct_dir, exist_ok=True)
    save_path = os.path.join(struct_dir, "structure_graph.pt")
    torch.save(structure_graph, save_path)

    return structure_graph, save_path



if __name__ == "__main__":
    # Example with hardcoded defaults
    structure_id = 1
    num_floors = 2
    grid_x = 5
    grid_z = 6
    beam_length_x = 20.0
    beam_length_z = 10.0
    column_length = 12.0
    ForceX = 90.0
    ForceY = 150.0

    structure, path = generate_structure(
        structure_id, num_floors, grid_x, grid_z,
        beam_length_x, beam_length_z, column_length, ForceX, ForceY
    )
    print(f"Generated structure {structure_id}: "
          f"{structure.num_nodes} nodes, "
          f"{structure.num_edges} edges → {path}")
