import torch
import os
import sys
import pandas as pd
from Utils import normalization

print("Working directory:", os.getcwd())

# Add the parent directory (StructGNN) to sys.path so Python can find 'GNN'
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from GNN.models import Structure_GraphNetwork  # Or Structure_GraphNetwork if used

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Reconstruct the model architecture (must match training)
model = Structure_GraphNetwork(
    layer_num=9,
    input_dim=11,
    hidden_dim=256,
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

#Load trained weights
checkpoint_folder = '2025_04_17__02_26_56'
model_path = os.path.normpath(os.path.join(script_dir, '..', 'Results', 'Static_Linear_Analysis', checkpoint_folder, 'model.pt'))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

#Load graph data for one structure
data_relative_path = os.path.join('Data', 'Static_Linear_Analysis', 'structure_1', 'structure_graph_NodeAsNode.pt')
data_path = os.path.normpath(os.path.join(script_dir, '..', data_relative_path))
data = torch.load(data_path, weights_only=False)

#Normalize data
norm_dict = {'grid_num': [0, 7.], 'coord': [0., 6.], 'mass': [0, 0.0237], 'force': [0, 200.], 'length': [0, 8.],
             'disp': [0, 238.1000], 'momentY': [0, 167100.], 'momentZ': [0, 206300.], 'shearY': [0, 93.1100], 'shearZ': [0, 93.1300], 'axialForce': [0, 1420.], 'torsion': [0, 1.8080]}
normalization.normalize_linear(data, norm_dict)

#Run inference and extract node outputs
model.eval()
with torch.no_grad():
    data = data.to(device)
    output = model(data.x, data.edge_index, data.edge_attr)

normalization.denormalize_y_linear(output, norm_dict)

#Format and print per-node output features
torch.set_printoptions(sci_mode=False)

labels = (
    ['disp_x', 'disp_z'] +
    [f'moment_y_{i}' for i in range(6)] +
    [f'moment_z_{i}' for i in range(6)] +
    [f'shear_y_{i}' for i in range(6)] +
    [f'shear_z_{i}' for i in range(6)]
)

# Create and print the output table
df = pd.DataFrame(output.cpu().numpy(), columns=labels)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\nLabeled Predictions for All Nodes:")
print(df)

# Save to CSV
df.to_csv("predicted_node_outputs.csv", index_label='node_index')

