import torch
import os
import sys
import pandas as pd

print("Working directory:", os.getcwd())

# Add the parent directory (StructGNN) to sys.path so Python can find 'GNN'
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from GNN.models import Structure_GraphNetwork_pseudo  # Adjust if you're using a different model

# Step 1: Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Reconstruct the model architecture (must match training!)
model = Structure_GraphNetwork_pseudo(
    layer_num=3,
    input_dim=11,
    hidden_dim=512,
    edge_attr_dim=3,
    aggr='mean',
    node_out_dispX_dim=1,
    node_out_dispZ_dim=1,
    node_out_momentY_dim=6,
    node_out_momentZ_dim=6,
    node_out_shearY_dim=6,
    node_out_shearZ_dim=6,
    device=device
)

# Step 3: Load the state dict (weights)
checkpoint_folder = '2025_07_01__01_42_38'
model_path = os.path.normpath(os.path.join(script_dir, '..', 'Results', 'Static_Linear_Analysis', checkpoint_folder, 'model.pt'))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# Step 4: Load graph data
data_relative_path = os.path.join('Data', 'Static_Linear_Analysis', 'structure_2', 'structure_graph_NodeAsNode_pseudo.pt')
data_path = os.path.normpath(os.path.join(script_dir, '..', data_relative_path))
data = torch.load(data_path, weights_only=False)

# Step 5: Run inference
model.eval()
with torch.no_grad():
    data = data.to(device)
    output = model(data.x, data.edge_index, data.edge_attr, layer_num=3)

# Step 6: Print labeled output
torch.set_printoptions(sci_mode=False)

# Define output labels
labels = (
    ['disp_x', 'disp_z'] +
    [f'moment_y_{i}' for i in range(6)] +
    [f'moment_z_{i}' for i in range(6)] +
    [f'shear_y_{i}' for i in range(6)] +
    [f'shear_z_{i}' for i in range(6)]
)

# Convert to DataFrame and display
if not isinstance(output, dict):
    df = pd.DataFrame(output.cpu().numpy(), columns=labels)

    # Configure pandas to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\nLabeled Predictions for All Nodes:")
    print(df)
    df.to_csv("predicted_node_outputs.csv", index_label='node_index')
else:
    print("Model returned a dictionary of outputs. Please adapt labeling logic accordingly.")

# Optional: Save raw tensor output
# torch.save(output, 'output_predictions.pt')
