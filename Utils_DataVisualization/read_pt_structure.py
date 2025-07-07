import os
import torch


def load_structure(structure_id, use_generated=False, use_pseudo=False):
    """Load PyTorch Geometric data for specified structure
    
    Args:
        structure_id: Structure number to load
        use_generated: If True, load from Data_Generated folder, else from Data folder
        use_pseudo: If True, load pseudo version (only applicable when use_generated=True)
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root (StructGNN directory)
    project_root = os.path.dirname(script_dir)
    
    # Choose filename based on pseudo parameter
    if use_generated and use_pseudo:
        filename = "structure_graph_NodeAsNode_pseudo.pt"
        data_type = "Pseudo-generated"
    elif use_generated:
        filename = "structure_graph_NodeAsNode.pt"
        data_type = "Generated"
    else:
        filename = "structure_graph_NodeAsNode.pt"
        data_type = "Original"
    
    # Build path based on data type
    if use_generated:
        # Data_Generated structure: StructGNN/Data_Generated/structure_X/
        data_path = os.path.join(
            project_root,
            "Data_Generated",
            f"structure_{structure_id}",
            filename
        )
    else:
        # Original data structure: StructGNN/Data/Static_Linear_Analysis/structure_X/
        data_path = os.path.join(
            project_root,
            "Data",
            "Static_Linear_Analysis",
            f"structure_{structure_id}",
            filename
        )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data found for structure {structure_id} at: {data_path}")
    
    print(f"Loading {data_type} structure {structure_id} from:\n{os.path.normpath(data_path)}")
    return torch.load(data_path, weights_only=False)


def print_diagnostics(structure_graph):
    """Print comprehensive graph diagnostics"""
    print("\n" + "="*50)
    print(f"{'STRUCTURE DIAGNOSTICS':^50}")
    print("="*50)
    
    # Basic graph properties
    print(f"\n{'[GRAPH PROPERTIES]':^50}")
    print(f"Number of nodes: {structure_graph.num_nodes}")
    print(f"Number of edges: {structure_graph.num_edges}")
    print(f"Is directed: {structure_graph.is_directed()}")
    print(f"Contains isolated nodes: {structure_graph.has_isolated_nodes()}")
    print(f"Contains self-loops: {structure_graph.has_self_loops()}")
    
    # Node features
    print(f"\n{'[NODE FEATURES]':^50}")
    print(f"Feature tensor shape: {structure_graph.x.shape}")
    print("All node features:")
    for i in range(structure_graph.num_nodes):
        print(f"Node {i}: {structure_graph.x[i].tolist()}")

    # Edge features
    print(f"\n{'[EDGE FEATURES]':^50}")
    if hasattr(structure_graph, 'edge_attr') and structure_graph.edge_attr is not None:
        print(f"Edge attribute shape: {structure_graph.edge_attr.shape}")
        print("All edge features:")
        for i in range(structure_graph.edge_attr.size(0)):
            print(f"Edge {i}: {structure_graph.edge_attr[i].tolist()}")
    else:
        print("No edge attributes found")
    
    # Edge information
    print(f"\n{'[EDGE INFORMATION]':^50}")
    print(f"Edge index shape: {structure_graph.edge_index.shape}")
    print("All edges:")
    for i in range(structure_graph.edge_index.shape[1]):
        src, tgt = structure_graph.edge_index[:, i].tolist()
        print(f"Edge {i}: {src} → {tgt}")
    
    # Additional diagnostics
    print(f"\n{'[ADDITIONAL PROPERTIES]':^50}")
    print(f"Graph keys: {list(structure_graph.keys())}")
    if 'y' in structure_graph:
        print(f"Target values shape: {structure_graph.y.shape}")
        print(f"Target values: {structure_graph.y.tolist()}")

        print("\nNo target outputs found in this structure graph.")
        return

    print(f"\n{'[TARGET OUTPUTS]':^50}")
    y = structure_graph.y
    print(f"Target shape: {y.shape}")

    try:
        print("\nDisplacement:")
        print(f"  disp_x: {y[:, 0]}")
        print(f"  disp_z: {y[:, 1]}")

        print("\nMomentY (cols 2–7):")
        print(y[:, 2:8])

        print("\nMomentZ (cols 8–13):")
        print(y[:, 8:14])

        print("\nShearY (cols 14–19):")
        print(y[:, 14:20])

        print("\nShearZ (cols 20–25):")
        print(y[:, 20:26])

        if y.shape[1] > 26:
            print("\nAxialForce (cols 26–31):")
            print(y[:, 26:32])
        if y.shape[1] > 32:
            print("\nTorsion (cols 32–37):")
            print(y[:, 32:38])
        if y.shape[1] > 38:
            print("\n[EXTRA COLUMN] (col 38):")
            print(y[:, 38])
    except Exception as e:
        print(f"Error extracting target components: {e}")


if __name__ == "__main__":
    # Ask user to choose data source
    while True:
        try:
            data_choice = input("Choose data source:\n1. Original data (Data folder)\n2. Generated data (Data_Generated folder)\n3. Pseudo-generated data (Data_Generated folder)\nEnter choice (1, 2, or 3): ")
            if data_choice in ['1', '2', '3']:
                if data_choice == '1':
                    use_generated = False
                    use_pseudo = False
                elif data_choice == '2':
                    use_generated = True
                    use_pseudo = False
                else:  # data_choice == '3'
                    use_generated = True
                    use_pseudo = True
                break
            print("Error: Please enter 1, 2, or 3")
        except ValueError:
            print("Error: Please enter 1, 2, or 3")
    
    # Ask for structure ID
    while True:
        try:
            structure_id = int(input("Enter structure number (1-2000): "))
            if 1 <= structure_id <= 2000:
                break
            print("Error: Number must be between 1-2000")
        except ValueError:
            print("Error: Please enter a valid integer")
    
    try:
        structure_graph = load_structure(structure_id, use_generated, use_pseudo)
        print_diagnostics(structure_graph)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check if the structure exists in the selected data folder.")
