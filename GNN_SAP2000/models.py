import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from .layers import *

class Structure_GraphNetwork(torch.nn.Module):
    """
    Node head: 2 outputs per node -> [dispX, dispY]
    Edge head: 6 outputs per edge -> [axial_shear, shear2, shear3, moment2, moment3, torsion]
    Returns: (node_out [N,2], edge_out [E,6])
    """
    def __init__(
        self,
        layer_num,
        input_dim,
        hidden_dim,
        edge_attr_dim,
        aggr,
        gnn_act=True,
        gnn_dropout=True,
        dropout_p=0.0,
        node_out_dim=2,     # [dispX, dispY]
        edge_out_dim=6,     # [axial_shear, shear2, shear3, moment2, moment3, torsion]
        include_edge_attr_in_edge_head=True,
        device="cuda",
        **kwargs
    ):
        super().__init__()
        self.layer_num = layer_num
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device
        self.include_edge_attr_in_edge_head = include_edge_attr_in_edge_head

        # --- Encoder & GNN ---
        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        self.conv_layer = GraphNetwork_layer(
            hidden_dim, hidden_dim, aggr=aggr, edge_attr_dim=edge_attr_dim
        )

        # --- Decoders ---
        # Node: single compact head -> 2 targets
        self.node_decoder = MLP(hidden_dim, [64], node_out_dim, act=True, dropout=False)

        # Edge: concatenation of [x_src, x_dst, (edge_attr?)] -> 6 targets
        edge_in_dim = hidden_dim * 2 + (edge_attr_dim if include_edge_attr_in_edge_head else 0)
        self.edge_decoder = MLP(edge_in_dim, [128, 64], edge_out_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr):
        # Encode nodes
        x = self.encoder(x)

        # Message passing
        for _ in range(self.layer_num):
            x = self.conv_layer(x, edge_index, edge_attr)
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        # --- Node outputs: [N,2] -> [dispX, dispY] ---
        node_out = self.node_decoder(x)

        # --- Edge outputs: [E,6] in the specified order ---
        src, dst = edge_index
        x_src = x[src]                   # [E, H]
        x_dst = x[dst]                   # [E, H]

        if self.include_edge_attr_in_edge_head and edge_attr is not None:
            edge_feats = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        else:
            edge_feats = torch.cat([x_src, x_dst], dim=-1)

        edge_out = self.edge_decoder(edge_feats)  # [E,6]

        return node_out, edge_out


class Structure_GraphNetwork_pseudo(torch.nn.Module):
    """
    Same heads/returns as Structure_GraphNetwork, but mirrors your original 'pseudo' setup.
    Note: original pseudo GraphNetwork_layer didn't pass edge_attr_dim; we keep that behavior.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        aggr,
        edge_attr_dim,   # kept for signature symmetry
        gnn_act=True,
        gnn_dropout=True,
        dropout_p=0.0,
        node_out_dim=2,
        edge_out_dim=6,
        include_edge_attr_in_edge_head=True,
        device="cuda",
        **kwargs
    ):
        super().__init__()
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device
        self.include_edge_attr_in_edge_head = include_edge_attr_in_edge_head
        self.edge_attr_dim = edge_attr_dim

        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        # original pseudo class didn't forward edge_attr_dim to the layer
        self.conv_layer = GraphNetwork_layer(hidden_dim, hidden_dim, aggr=aggr)

        self.node_decoder = MLP(hidden_dim, [64], node_out_dim, act=True, dropout=False)

        edge_in_dim = hidden_dim * 2 + (edge_attr_dim if include_edge_attr_in_edge_head else 0)
        self.edge_decoder = MLP(edge_in_dim, [128, 64], edge_out_dim, act=True, dropout=False)

    def forward(self, x, edge_index, edge_attr, layer_num):
        x = self.encoder(x)

        for _ in range(layer_num):
            x = self.conv_layer(x, edge_index, edge_attr)
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        node_out = self.node_decoder(x)

        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        if self.include_edge_attr_in_edge_head and edge_attr is not None:
            edge_feats = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        else:
            edge_feats = torch.cat([x_src, x_dst], dim=-1)

        edge_out = self.edge_decoder(edge_feats)

        return node_out, edge_out
