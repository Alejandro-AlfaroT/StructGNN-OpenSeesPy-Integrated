import os
import torch
import pandas as pd

def load_structure_from_path(data_path):
    """Load PyTorch Geometric data directly from a file path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data file found at the specified path: {data_path}")
    
    print(f"Loading structure from: {os.path.normpath(data_path)}")
    try:
        return torch.load(data_path, weights_only=False)
    except Exception as e:
        raise ValueError(f"Error loading PyTorch Geometric data from {data_path}: {e}")

def display_diagnostics_terminal(structure_graph, file_name):
    """Displays diagnostics of the structure graph in the terminal."""
    print("\n" + "="*50)
    print(f"Diagnostics for: {file_name}")
    print("="*50 + "\n")

    # 1. Graph Properties
    print("--- Graph Properties ---")
    print(f"Number of nodes: {structure_graph.num_nodes}")
    print(f"Number of edges: {structure_graph.num_edges}")
    print(f"Is directed: {structure_graph.is_directed()}")
    print(f"Has isolated nodes: {structure_graph.has_isolated_nodes()}")
    print(f"Has self-loops: {structure_graph.has_self_loops()}")
    print(f"Graph keys: {list(structure_graph.keys())}\n")

    # 2. Node Features
    print("--- Node Features ---")
    if hasattr(structure_graph, 'x') and structure_graph.x is not None:
        node_features_data = structure_graph.x.tolist()
        num_node_features = structure_graph.x.shape[1]

        node_feature_names = [
            "# Spans X", "# Spans Y", "# Spans Z",
            "X Coord", "Y Coord", "Z Coord",
            "Support", "Joint", "Nodal Mass",
            "Force X", "Force Z"
        ]
        
        if len(node_feature_names) == num_node_features:
            columns = node_feature_names
        else:
            columns = [f"Feature {i+1}" for i in range(num_node_features)]
        
        df_nodes = pd.DataFrame(node_features_data, columns=columns)
        df_nodes.index.name = "Node #"
        print(df_nodes.to_string())
    else:
        print("No node features (graph.x) found.")
    print("\n")

    # 3. Edge Features
    print("--- Edge Features ---")
    if hasattr(structure_graph, 'edge_attr') and structure_graph.edge_attr is not None:
        edge_features_data = structure_graph.edge_attr.tolist()
        num_edge_features = structure_graph.edge_attr.shape[1]

        edge_feature_names = ["Beam", "Column", "Length"]

        if len(edge_feature_names) == num_edge_features:
            columns = edge_feature_names
        else:
            columns = [f"Feature {i+1}" for i in range(num_edge_features)]

        df_edges_attr = pd.DataFrame(edge_features_data, columns=columns)
        df_edges_attr.index.name = "Edge #"
        print(df_edges_attr.to_string())
    else:
        print("No edge features (graph.edge_attr) found.")
    print("\n")

    # 4. Edge Index
    print("--- Edge Index (Source, Target) ---")
    if hasattr(structure_graph, 'edge_index') and structure_graph.edge_index is not None:
        edge_index_data = structure_graph.edge_index.t().tolist()
        df_edge_index = pd.DataFrame(edge_index_data, columns=["Source Node", "Target Node"])
        df_edge_index.index.name = "Edge #"
        print(df_edge_index.to_string())
    else:
        print("No edge index (graph.edge_index) found.")
    print("\n")

    # 5. Target Values (y)
    print("--- Target Values (y) ---")
    if hasattr(structure_graph, 'y') and structure_graph.y is not None:
        target_data = structure_graph.y.tolist()
        num_targets = structure_graph.y.shape[1] if structure_graph.y.dim() > 1 else 1

        target_names = ["Disp X", "Disp Y"]
        for i in range(6):
            target_names.append(f"Moment Y {i+1}")
        for i in range(6):
            target_names.append(f"Moment Z {i+1}")
        for i in range(6):
            target_names.append(f"Shear Y {i+1}")
        for i in range(6):
            target_names.append(f"Shear Z {i+1}")

        if len(target_names) == num_targets:
            columns = target_names
        else:
            columns = [f"Target {i+1}" for i in range(num_targets)]
        
        df_targets = pd.DataFrame(target_data, columns=columns)
        df_targets.index.name = "Node #"
        print(df_targets.to_string())
    else:
        print("No target values (graph.y) found.")
    print("\n")

def interactive_file_explorer(initial_path=None, file_extension=".pt"):
    """
    Provides an interactive terminal-based file explorer.
    Allows navigation through directories and selection of files.
    """
    if initial_path is None:
        initial_path = os.getcwd() # Start in current working directory
    
    current_path = os.path.abspath(initial_path)

    while True:
        print(f"\nCurrent Directory: {current_path}")
        
        items = sorted(os.listdir(current_path))
        
        # Separate directories and files
        dirs = [d for d in items if os.path.isdir(os.path.join(current_path, d))]
        files = [f for f in items if os.path.isfile(os.path.join(current_path, f))]

        numbered_items = []
        
        if os.path.abspath(current_path) != os.path.abspath(os.path.dirname(current_path)):
            # Only show '..' if not at the root of the drive/system
            numbered_items.append(".. (Go up one directory)")

        print("\n--- Directories ---")
        for i, d in enumerate(dirs):
            numbered_items.append(f"{d}/")
            print(f"{len(numbered_items) - 1}. {d}/")
        
        eligible_files = [f for f in files if f.endswith(file_extension)]
        other_files = [f for f in files if not f.endswith(file_extension)]

        print("\n--- .pt Files (Selectable) ---")
        if not eligible_files:
            print("  (No .pt files in this directory)")
        for i, f in enumerate(eligible_files):
            numbered_items.append(f)
            print(f"{len(numbered_items) - 1}. {f}")
        
        if other_files:
            print("\n--- Other Files ---")
            for f in other_files:
                print(f"  {f}") # Not numbered, just for visibility

        print("\n--- Options ---")
        print("Type 'q' to quit.")
        print("Type the number or name of a directory/file to navigate/select.")
        
        choice = input(f"Enter choice or directory/file name (e.g., '{numbered_items[1].split(' ')[0] if len(numbered_items) > 1 else 'q'}'): ").strip()

        if choice.lower() == 'q':
            return None
        
        target_item = None
        try:
            # Try to interpret as a number
            idx = int(choice)
            if 0 <= idx < len(numbered_items):
                target_item = numbered_items[idx]
        except ValueError:
            # Not a number, try to interpret as a name
            if choice == '..':
                target_item = '..'
            elif choice.endswith('/') and choice[:-1] in dirs:
                target_item = choice[:-1] # Remove trailing slash for actual dir name
            elif choice in dirs:
                target_item = choice
            elif choice in eligible_files:
                target_item = choice
            else:
                print(f"Invalid choice or name: '{choice}'. Please try again.")
                continue

        if target_item == '..':
            current_path = os.path.dirname(current_path)
            if not os.path.exists(current_path): # Handle going up from root
                current_path = os.path.abspath(os.sep) # Or back to the C:\ on Windows
            continue
        
        selected_path = os.path.join(current_path, target_item)

        if os.path.isdir(selected_path):
            current_path = selected_path
        elif os.path.isfile(selected_path) and selected_path.endswith(file_extension):
            return selected_path
        else:
            print(f"Invalid selection: '{target_item}'. Not a valid directory or a '{file_extension}' file. Please try again.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Adjust this if your project root is different

    print("Starting interactive file explorer. Navigate to your .pt file.")
    selected_file_path = interactive_file_explorer(initial_path=project_root, file_extension=".pt")

    if selected_file_path:
        try:
            structure_graph = load_structure_from_path(selected_file_path)
            display_diagnostics_terminal(structure_graph, os.path.basename(selected_file_path))
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print("File selection cancelled. Exiting.")