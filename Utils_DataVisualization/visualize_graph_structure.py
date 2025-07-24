import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import tkinter as tk
from tkinter import filedialog, messagebox
import re

def load_structure(data_path):
    """Load PyTorch Geometric data from a specific file path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data file found at: {data_path}")
    print(f"Loading structure from:\n{os.path.normpath(data_path)}")
    return torch.load(data_path, weights_only=False)

def visualize_graph(structure_graph, structure_id):
    """
    Generate an optimized 3D visualization of a structure graph with node labels.
    """
    positions = {
        i: (
            structure_graph.x[i, 3].item(),
            structure_graph.x[i, 4].item(),
            structure_graph.x[i, 5].item()
        )
        for i in range(structure_graph.num_nodes)
    }

    edge_list = structure_graph.edge_index.numpy().T

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    edge_segments = [(positions[u], positions[v]) for u, v in edge_list]
    edge_collection = Line3DCollection(edge_segments, colors="b", linewidths=0.8, zorder=1)
    ax.add_collection3d(edge_collection)

    node_coords = np.array(list(positions.values()))
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2],
               c='r', s=25, edgecolors='k', zorder=5)

    for node_id, pos in positions.items():
        ax.text(pos[0], pos[1], pos[2], str(node_id), fontsize=7, zorder=6, color='#333333')

    if node_coords.size > 0:
        min_vals = np.min(node_coords, axis=0)
        max_vals = np.max(node_coords, axis=0)
        center = (min_vals + max_vals) / 2
        span = np.max(max_vals - min_vals) * 0.6
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'3D Structural Graph (ID: {structure_id})', pad=20, fontsize=14)
    ax.view_init(elev=25, azim=-75)
    plt.tight_layout()
    plt.show()

def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a PyTorch Geometric .pt file",
        filetypes=[("PyTorch Geometric files", "*.pt"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected.")
        return

    try:
        match = re.search(r'structure_(\d+)', file_path)
        structure_id = match.group(1) if match else os.path.basename(file_path)
        structure_graph = load_structure(file_path)
        print(f"\nGraph Metadata: Nodes: {structure_graph.num_nodes}, Edges: {structure_graph.num_edges}")
        visualize_graph(structure_graph, structure_id)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(e)

if __name__ == "__main__":
    main()
