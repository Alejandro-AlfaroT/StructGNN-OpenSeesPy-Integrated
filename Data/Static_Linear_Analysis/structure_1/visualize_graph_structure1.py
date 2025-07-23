import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots

# Construct path to the .pt file relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.normpath(os.path.join(script_dir, 'structure_graph_NodeAsNode.pt'))

print(f"Loading graph data from: {file_path}")

# Attempt to load the file with weights_only=False to bypass the error
structure_graph = torch.load(file_path, weights_only=False)

# Assume `structure_graph` is your PyG Data object
# structure_graph.x is (num_nodes, 11) → [span, story, ID, x, y, z, DoF, mass_flag, load, angle, dist]
x, y, z = structure_graph.x[:, 3], structure_graph.x[:, 4], structure_graph.x[:, 5]

# Convert to numpy-style dict for positions
positions = {i: (x[i].item(), y[i].item(), z[i].item()) for i in range(structure_graph.num_nodes)}

# Create undirected graph for visualization
G = nx.Graph()
edge_index = structure_graph.edge_index.numpy()

# Add edges
for i in range(edge_index.shape[1]):
    src = edge_index[0, i]
    tgt = edge_index[1, i]
    G.add_edge(src, tgt)

# 3D Plot (x-y or x-z projection)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot edges
for edge in G.edges():
    x_coords = [positions[edge[0]][0], positions[edge[1]][0]]
    y_coords = [positions[edge[0]][1], positions[edge[1]][1]]
    z_coords = [positions[edge[0]][2], positions[edge[1]][2]]
    ax.plot(x_coords, y_coords, z_coords, c='b')

# Plot nodes
xs = [positions[i][0] for i in G.nodes]
ys = [positions[i][1] for i in G.nodes]
zs = [positions[i][2] for i in G.nodes]
ax.scatter(xs, ys, zs, c='r', s=10)

# Label each node
for node_id, (x_, y_, z_) in positions.items():
    ax.text(x_, y_, z_ + 0.1, str(node_id), fontsize=8, color='black')  # 0.1 offset to lift label above node

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Structural Graph')

plt.tight_layout()
plt.show()
