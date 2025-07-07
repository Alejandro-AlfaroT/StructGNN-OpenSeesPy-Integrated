import os
import torch
import numpy as np
from torch_geometric.data import Data
import tkinter as tk
from tkinter import simpledialog, messagebox

def get_user_input():
    """Creates a GUI to get structure parameters from the user."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    try:
        num_floors_str = simpledialog.askstring("Input", "Enter the number of floors:", parent=root)
        if num_floors_str is None: return None
        num_floors = int(num_floors_str)

        grid_x_str = simpledialog.askstring("Input", "Enter the number of blocks in the X direction:", parent=root)
        if grid_x_str is None: return None
        grid_x = int(grid_x_str)

        grid_z_str = simpledialog.askstring("Input", "Enter the number of blocks in the Z direction:", parent=root)
        if grid_z_str is None: return None
        grid_z = int(grid_z_str)

        beam_length_str = simpledialog.askstring("Input", "Enter the length of the beams:", parent=root)
        if beam_length_str is None: return None
        beam_length = float(beam_length_str)

        column_lengths = []
        for i in range(num_floors):
            length_str = simpledialog.askstring("Input", f"Enter the column length for floor {i+1}:", parent=root)
            if length_str is None: return None
            column_lengths.append(float(length_str))

    except (ValueError, TypeError):
        messagebox.showerror("Invalid Input", "Please enter valid numbers.")
        return None

    return num_floors, grid_x, grid_z, beam_length, column_lengths

def generate_structure(structure_id, num_floors, grid_x, grid_z, beam_length, column_lengths, output_dir="Data_Gen_Manual"):
    """Generates a building structure with specified parameters."""
    # Create nodes
    nodes = []
    y_val = num_floors + 1
    for y in range(y_val):
        for z in range(grid_z):
            for x in range(grid_x):
                mass_flag = 1.0 if (x == 0 and z == 0) or (x == grid_x - 1 and z == 0) else 0.0
                load = np.random.uniform(0.003, 0.015)

                nodes.append([
                    grid_x, y_val, grid_z,
                    float(x), float(y), float(z),
                    0.0 if y == 0 else 1.0,
                    mass_flag, load, 0.0, 0.0
                ])

    # Create edges and edge attributes
    beams = []
    columns = []
    beam_attributes = []
    column_attributes = []

    def get_node_index(x, z, y):
        return y * (grid_x * grid_z) + z * grid_x + x

    # Horizontal connections (beams)
    for y in range(1, y_val):  # Skip the ground floor (y=0) for the table effect
        for z in range(grid_z):
            for x in range(grid_x):
                current = get_node_index(x, z, y)
                if x < grid_x - 1:
                    right = get_node_index(x + 1, z, y)
                    beams.extend([[current, right], [right, current]])
                    beam_attributes.extend([[beam_length], [beam_length]])
                if z < grid_z - 1:
                    depth = get_node_index(x, z + 1, y)
                    beams.extend([[current, depth], [depth, current]])
                    beam_attributes.extend([[beam_length], [beam_length]])

    # Vertical connections (columns)
    for y in range(y_val - 1):
        for z in range(grid_z):
            for x in range(grid_x):
                current = get_node_index(x, z, y)
                above = get_node_index(x, z, y + 1)
                columns.extend([[current, above], [above, current]])
                column_len = column_lengths[y]
                column_attributes.extend([[column_len], [column_len]])

    # Combine edges and create edge features
    edge_index = torch.tensor(beams + columns, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(beam_attributes + column_attributes, dtype=torch.float)

    node_features = torch.tensor(nodes, dtype=torch.float)

    structure_graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.randn(node_features.shape[0], 38)
    )

    # Save structure
    struct_dir = os.path.join(output_dir, f"structure_{structure_id}")
    os.makedirs(struct_dir, exist_ok=True)
    save_path = os.path.join(struct_dir, "structure_graph.pt")
    torch.save(structure_graph, save_path)

    return structure_graph, save_path

if __name__ == "__main__":
    user_input = get_user_input()
    if user_input:
        num_floors, grid_x, grid_z, beam_length, column_lengths = user_input
        structure_id = 1 # Or get from command line if needed
        # The output directory is now set to "Data_Gen_Manual" by default
        structure, path = generate_structure(structure_id, num_floors, grid_x, grid_z, beam_length, column_lengths)
        print(f"Generated structure {structure_id}: "
              f"{structure.num_nodes} nodes, "
              f"{structure.num_edges} edges → {path}")