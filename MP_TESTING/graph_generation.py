import os
import torch
import numpy as np
from torch_geometric.data import Data

def generate_structure(
    structure_id,
    num_floors,
    grid_x,
    grid_y,
    beam_length_x,
    beam_length_y,
    column_length,
    ForceX=90.0,
    ForceY=150.0,
    output_dir="Data_Gen_Manual",
):
    # --- Dimensions  ---
    size_x = grid_y + 1
    size_y = grid_x + 1
    size_z = num_floors + 1
    max_z  = size_z - 1

    # Directional lengths
    Lx   = float(beam_length_y)
    Ly   = float(beam_length_x)
    Lcol = float(column_length)

    coords = []
    for x in range(size_x):             # outer: +x
        for y in range(size_y):         # then: +y
            for z in range(size_z):     # fastest: +z (vertical)
                coords.append((x, y, z))
    coords = np.array(coords, dtype=float)

    # Top level (z = max_z): choose min-x and min-y among top nodes
    top_mask = coords[:, 2] == max_z
    top_coords = coords[top_mask]
    min_x = top_coords[:, 0].min()
    min_y = top_coords[:, 1].min()

    # --- Nodes ---
    nodes = []
    for (x, y, z) in coords:
        support    = 1.0 if z == 0 else 0.0
        joint      = 1.0 if z != 0 else 0.0
        nodal_mass = 0.0

        on_top  = (z == max_z)
        force_x = float(ForceX) if (on_top and x == min_x) else 0.0
        force_y = float(ForceY) if (on_top and y == min_y) else 0.0

        nodes.append([
            float(size_x), float(size_y), float(size_z),  # counts (x,y,z)
            float(x), float(y), float(z),                 # indices as coords
            support, joint, nodal_mass,
            force_x, force_y
        ])

    # --- Linear index consistent with the node order above ---
    def get_node_index(xi, yi, zi):
        # x-major, then y, with z fastest (column-wise; +y then +x)
        return int(xi * (size_y * size_z) + yi * size_z + zi)

    # ---------- EDGE ORDER ----------
    edges = []
    edge_attributes = []  # [beam, column, length]

    # 1) Columns: bottom→top, then +y, then +x
    for x in range(0, size_x):  # outer: +x
        for y in range(0, size_y):  # then: +y
            for z in range(0, size_z - 1):  # fastest: bottom→top
                a = get_node_index(x, y, z)
                b = get_node_index(x, y, z + 1)
                edges.append([a, b])
                edge_attributes.append([0.0, 1.0, Lcol])
                edges.append([b, a])
                edge_attributes.append([0.0, 1.0, Lcol])

    # 2) Beams along +x within floors (z >= 1):
    #    bottom→top (z fastest), then +x, then +y
    for y in range(0, size_y):  # outer: +y (last)
        for x in range(0, size_x - 1):  # then: +x
            for z in range(1, size_z):  # fastest: bottom→top (z = 1..top)
                a = get_node_index(x, y, z)
                b = get_node_index(x + 1, y, z)
                edges.append([a, b])
                edge_attributes.append([1.0, 0.0, Lx])
                edges.append([b, a])
                edge_attributes.append([1.0, 0.0, Lx])

    # 3) Beams along +y within floors (z >= 1):
    #    bottom→top (z fastest), then +y, then +x
    for x in range(0, size_x):  # outer: +x (last)
        for y in range(0, size_y - 1):  # then: +y
            for z in range(1, size_z):  # fastest: bottom→top (z = 1..top)
                a = get_node_index(x, y, z)
                b = get_node_index(x, y + 1, z)
                edges.append([a, b])
                edge_attributes.append([1.0, 0.0, Ly])
                edges.append([b, a])
                edge_attributes.append([1.0, 0.0, Ly])

    # --- Graph tensors ---
    edge_index    = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr     = torch.tensor(edge_attributes, dtype=torch.float)   # (E, 3)
    node_features = torch.tensor(nodes, dtype=torch.float)             # (N, 11)

    # Targets
    node_targets = torch.zeros(node_features.shape[0], 2, dtype=torch.float)
    edge_targets = torch.zeros(edge_attr.shape[0], 6, dtype=torch.float)

    structure_graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,    # [beam, column, length]
        y=node_targets,         # (N, 2)
        edge_y=edge_targets     # (E, 6)
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
    grid_x = 4
    grid_y = 4
    beam_length_x = 20.0
    beam_length_y = 10.0
    column_length = 12.0
    ForceX = 90.0
    ForceY = 150.0

    structure, path = generate_structure(
        structure_id, num_floors, grid_x, grid_y,
        beam_length_x, beam_length_y, column_length, ForceX, ForceY
    )
    print(f"Generated structure {structure_id}: "
          f"{structure.num_nodes} nodes, "
          f"{structure.num_edges} edges → {path}")
