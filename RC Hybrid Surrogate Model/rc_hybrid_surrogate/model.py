"""Causal GNN-conditioned LSTM for bidirectional roof response prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 8,
        edge_dim: int = 14,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.convolutions = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convolutions.append(
                GINEConv(
                    _mlp(hidden_dim, hidden_dim * 2, hidden_dim),
                    edge_dim=hidden_dim,
                    train_eps=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph) -> torch.Tensor:
        x = self.node_encoder(graph.x)
        edge_attr = self.edge_encoder(graph.edge_attr)
        for convolution, norm in zip(self.convolutions, self.norms):
            update = convolution(x, graph.edge_index, edge_attr)
            x = norm(x + self.dropout(torch.nn.functional.silu(update)))
        mean = global_mean_pool(x, graph.batch)
        maximum = global_max_pool(x, graph.batch)
        return torch.cat((mean, maximum), dim=-1)


class HybridGNNLSTM(nn.Module):
    """Encode a structural graph once and condition a causal response sequence."""

    def __init__(
        self,
        node_dim: int = 8,
        edge_dim: int = 14,
        global_dim: int = 43,
        record_dim: int = 12,
        graph_hidden_dim: int = 128,
        graph_layers: int = 4,
        condition_dim: int = 128,
        lstm_hidden_dim: int = 192,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 2,
    ):
        super().__init__()
        self.graph_encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=graph_hidden_dim,
            layers=graph_layers,
            dropout=dropout,
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, graph_hidden_dim),
            nn.LayerNorm(graph_hidden_dim),
            nn.SiLU(),
        )
        self.record_encoder = nn.Sequential(
            nn.Linear(record_dim, graph_hidden_dim),
            nn.LayerNorm(graph_hidden_dim),
            nn.SiLU(),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(graph_hidden_dim * 4, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.SiLU(),
        )
        self.lstm = nn.LSTM(
            input_size=condition_dim + 2,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim // 2, output_dim),
        )

    def encode_condition(self, batch: dict) -> torch.Tensor:
        graph_embedding = self.graph_encoder(batch["graph"])
        global_embedding = self.global_encoder(batch["global_features"])
        record_embedding = self.record_encoder(batch["record_features"])
        return self.condition_encoder(
            torch.cat((graph_embedding, global_embedding, record_embedding), dim=-1)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        condition = self.encode_condition(batch)
        sequence_length = batch["ground_motion"].shape[1]
        repeated_condition = condition.unsqueeze(1).expand(-1, sequence_length, -1)
        sequence_input = torch.cat((batch["ground_motion"], repeated_condition), dim=-1)
        packed = pack_padded_sequence(
            sequence_input,
            batch["lengths"].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence_length,
        )
        return self.decoder(output)
