import os
import torch
import numpy as np
from torch_geometric.data import Data

def generate_structure(
    structure_id,
    num_floors,
    grid_x,
    grid_z,
    beam_length_x,   # ⬅ X-direction beams
    beam_length_z,   # ⬅ Z-direction beams
    column_length,   # ⬅ vertical members
    output_dir="Data_Gen_Manual",
):
    """Generates a building structure with independent X/Z beam lengths.
       Node IDs now increase along Y first (then X, then Z)."""
    # Nodes (Y fastest)
    nodes = []
    y_val = num_floors + 1
    for z in range(grid_z):
        for x in range(grid_x):
            for y in range(y_val):
                mass_flag = 1.0 if (x == 0 and z == 0) or (x == grid_x - 1 and z == 0) else 0.0
                load = np.random.uniform(0.003, 0.015)
                nodes.append([
                    grid_x, y_val, grid_z,
                    float(x), float(y), float(z),
                    0.0 if y == 0 else 1.0,
                    mass_flag, load, 0.0, 0.0
                ])

    # Edges
    beams = []
    columns = []
    beam_attributes = []
    column_attributes = []

    # Y is fastest: index = z*(grid_x*y_val) + x*y_val + y
    def get_node_index(x, z, y):
        return z * (grid_x * y_val) + x * y_val + y

    # Horizontal beams (skip y=0 if you want a "table" ground)
    for y in range(1, y_val):
        for z in range(grid_z):
            for x in range(grid_x):
                current = get_node_index(x, z, y)

                # X-direction neighbor (right)
                if x < grid_x - 1:
                    right = get_node_index(x + 1, z, y)
                    beams.extend([[current, right], [right, current]])
                    beam_attributes.extend([[beam_length_x], [beam_length_x]])

                # Z-direction neighbor (deeper)
                if z < grid_z - 1:
                    depth = get_node_index(x, z + 1, y)
                    beams.extend([[current, depth], [depth, current]])
                    beam_attributes.extend([[beam_length_z], [beam_length_z]])

    # Columns (vertical)
    for y in range(y_val - 1):
        for z in range(grid_z):
            for x in range(grid_x):
                current = get_node_index(x, z, y)
                above = get_node_index(x, z, y + 1)
                columns.extend([[current, above], [above, current]])
                column_attributes.extend([[column_length], [column_length]])

    # Graph tensors
    edge_index = torch.tensor(beams + columns, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(beam_attributes + column_attributes, dtype=torch.float)
    node_features = torch.tensor(nodes, dtype=torch.float)

    structure_graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,    # each row is [length]
        y=torch.zeros(node_features.shape[0], 38)
    )

    # Save
    struct_dir = os.path.join(output_dir, f"structure_{structure_id}")
    os.makedirs(struct_dir, exist_ok=True)
    save_path = os.path.join(struct_dir, "structure_graph.pt")
    torch.save(structure_graph, save_path)

    return structure_graph, save_path


if __name__ == "__main__":
    # Hardcoded inputs
    structure_id = 1
    num_floors = 2
    grid_x = 5
    grid_z = 6
    beam_length_x = 20.0   # X beams
    beam_length_z = 10.0   # Z beams
    column_length = 12.0   # Columns

    structure, path = generate_structure(
        structure_id, num_floors, grid_x, grid_z,
        beam_length_x, beam_length_z, column_length
    )
    print(f"Generated structure {structure_id}: "
          f"{structure.num_nodes} nodes, "
          f"{structure.num_edges} edges → {path}")
