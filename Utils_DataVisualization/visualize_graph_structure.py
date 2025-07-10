import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection # Import for efficient line drawing
import tkinter as tk
from tkinter import ttk, messagebox
import re
from file_explorer_tk import CustomFileExplorer # Import the class directly

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
    # Extract node positions into a dictionary for easy lookup
    positions = {i: (
        structure_graph.x[i, 3].item(), 
        structure_graph.x[i, 4].item(), 
        structure_graph.x[i, 5].item()
    ) for i in range(structure_graph.num_nodes)}
    
    # Get the list of edges directly from the graph data
    edge_list = structure_graph.edge_index.numpy().T
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- 🚀 Performance Improvement ---
    # Use Line3DCollection to draw all edges in a single, fast operation
    # instead of a slow Python loop.
    edge_segments = [(positions[u], positions[v]) for u, v in edge_list]
    edge_collection = Line3DCollection(edge_segments, colors="b", linewidths=0.8, zorder=1)
    ax.add_collection3d(edge_collection)

    # Plot nodes with a single efficient scatter call
    node_coords = np.array(list(positions.values()))
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], c='r', s=25, edgecolors='k', zorder=5)

    # --- ✅ Fix: Restore Node Labels ---
    # Iterate through nodes to add a text label for each one.
    for node_id, pos in positions.items():
        ax.text(pos[0], pos[1], pos[2], str(node_id), fontsize=7, zorder=6, color='#333333')

    # Manually set axis limits to ensure the entire structure is visible and centered
    if node_coords.size > 0:
        min_vals = np.min(node_coords, axis=0)
        max_vals = np.max(node_coords, axis=0)
        center = (min_vals + max_vals) / 2
        span = np.max(max_vals - min_vals) * 0.6  # 0.6 provides a nice padding
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'3D Structural Graph (ID: {structure_id})', pad=20, fontsize=14)
    ax.view_init(elev=25, azim=-75) # Set a nice default viewing angle
    plt.tight_layout()
    plt.show()

class StructureVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Structure Visualizer")
        self.root.geometry("450x220")
        self.root.resizable(False, False)
        # Correctly find the project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(script_dir)
        if not os.path.isdir(os.path.join(self.project_root, 'data')): # Heuristic to check if root is correct
             self.project_root = script_dir

        self.set_modern_theme()
        self.file_path_var = tk.StringVar(value="No file selected...")
        self.file_browser_window = None
        self.create_widgets()
        
    def set_modern_theme(self):
        style = ttk.Style()
        if 'clam' in style.theme_names(): style.theme_use('clam')
        BG_COLOR, TEXT_COLOR, ACCENT_COLOR, FONT_FAMILY = '#eaf0f6', '#333333', '#0078d4', "Segoe UI"
        self.root.configure(background=BG_COLOR)
        style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, 10))
        style.configure('TFrame', background=BG_COLOR)
        style.configure('TLabel', background=BG_COLOR, font=(FONT_FAMILY, 10))
        style.configure('TButton', font=(FONT_FAMILY, 10, 'bold'), padding=6)
        style.configure('TLabelframe', background=BG_COLOR, font=(FONT_FAMILY, 11, 'bold'))
        style.configure('TLabelframe.Label', background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure('Accent.TButton', background=ACCENT_COLOR, foreground='white')
        style.map('Accent.TButton', background=[('active', '#005a9e')])

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=(20, 20))
        main_frame.pack(fill="both", expand=True)
        file_frame = ttk.LabelFrame(main_frame, text="Selected Structure File")
        file_frame.pack(fill="x", pady=(0, 15), ipady=10)
        path_label = ttk.Label(file_frame, textvariable=self.file_path_var, wraplength=380, anchor="center")
        path_label.pack(padx=10, pady=(5, 10), fill="x")
        browse_button = ttk.Button(file_frame, text="Browse...", command=self.open_file_browser)
        browse_button.pack(pady=(0, 5))
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        self.visualize_btn = ttk.Button(button_frame, text="Visualize", command=self.on_visualize, style="Accent.TButton", state="disabled")
        self.visualize_btn.pack(side="right")
        ttk.Button(button_frame, text="Cancel", command=self.root.destroy).pack(side="right", padx=(0, 10))

    def open_file_browser(self):
        if self.file_browser_window and self.file_browser_window.winfo_exists():
            self.file_browser_window.lift()
            return
        # Use the imported class
        self.file_browser_window = CustomFileExplorer(self.root, self.project_root, self.on_file_selected)

    def on_file_selected(self, filepath):
        if filepath:
            self.file_path_var.set(os.path.normpath(filepath))
            self.visualize_btn.config(state="normal")
            if self.file_browser_window:
                self.file_browser_window.destroy()
            self.file_browser_window = None

    def on_visualize(self):
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("File Error", "Please select a valid structure file first.")
            return
        match = re.search(r'structure_(\d+)', file_path)
        structure_id = match.group(1) if match else os.path.basename(file_path)
        try:
            structure_graph = load_structure(file_path)
            print(f"\nGraph Metadata: Nodes: {structure_graph.num_nodes}, Edges: {structure_graph.num_edges}")
            self.root.iconify() # Minimize the GUI window instead of destroying it
            visualize_graph(structure_graph, structure_id)
            self.root.deiconify() # Restore the GUI window after the plot is closed
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}\n\nCheck console for details.")
            print(e)

if __name__ == "__main__":
    root = tk.Tk()
    app = StructureVisualizerGUI(root)
    root.mainloop()