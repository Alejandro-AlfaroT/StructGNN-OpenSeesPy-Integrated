import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from argparse import ArgumentParser

from GNN_SAP2000.models import *
from GNN_SAP2000 import losses
from Utils_SAP2000 import plot
from Utils_SAP2000 import datasets
from Utils_SAP2000 import accuracy
from Utils_SAP2000 import normalization

import time
from datetime import datetime
import os
import json


space = plot.print_space()

# Args
parser = ArgumentParser()
# ----------------------- General / dataset -----------------------
parser.add_argument('--data_num', dest='data_num', default=2160, type=int)
parser.add_argument('--train_ratio', dest='train_ratio', default=0.85, type=float)
parser.add_argument('--normalization', dest='normalization', default=True, type=bool)

# ----------------------- GNN -----------------------
parser.add_argument('--model', dest='model', default='Structure_GraphNetwork', type=str,
                    help='model class name. E.g., Structure_GraphNetwork')
parser.add_argument('--aggr_type', dest='aggr', default='mean', type=str)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=256, type=int)
parser.add_argument('--layer_num', dest='layer_num', default=9, type=int)
parser.add_argument('--dropout_p', dest='dropout_p', default=0.0, type=float)
parser.add_argument('--gnn_dropout', dest='gnn_dropout', default=True, type=bool,
                    help='whether dropout, default 0.5')
parser.add_argument('--gnn_act', dest='gnn_act', default=True, type=bool,
                    help='whether activation, default RELU')

# ----------------------- Training -----------------------
parser.add_argument('--target', dest='target', default='all', type=str,
                    help='which output target you are going to train (node slice), e.g., all')
parser.add_argument('--epoch_num', dest='epoch_num', default=3500, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)
parser.add_argument('--lr', dest='lr', default=5e-5, type=float)
parser.add_argument('--loss_function', dest='loss_function', default='L1', type=str,
                    help='L1 or L2 (combined node+edge).')
parser.add_argument('--accuracy_threshold', dest='accuracy_threshold', default=1e-4, type=float,
                    help='The normalized output value smaller than the threshold will be ignored.')

# Bookkeeping
parser.add_argument('--training_time', dest='training_time', default=0, type=float)

args = parser.parse_args()
print(args, end=space)

torch.manual_seed(0)
date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
print("Start time: " + datetime.now().strftime('%b %d, %H:%M:%S'), end=space)

# ----------------------- Load in data -----------------------
dataset = datasets.get_dataset(structure_num=args.data_num)  # always reads Data_SAP2000
print("dataset root: Data_SAP2000")
print(f"requested: {args.data_num}, loaded: {len(dataset)}", end=space)

if len(dataset) == 0:
    raise RuntimeError("No graphs were loaded from Data_SAP2000. "
                       "Check that files like Data_SAP2000/structure_1.pt exist.")

# ----------------------- Normalization -----------------------
if args.normalization:
    dataset, norm_dict = normalization.normalize_dataset(dataset, analysis='linear')
    print("Normalization: True")
    print("Normlization dictionary: ")
    print(norm_dict, end=space)
else:
    print("Normalization: False", end=space)

# ----------------------- Inspect first graph -----------------------
data = dataset[0]
print("x shape:", data.x.shape)
print(data.x[:5])
print("edge_index:", data.edge_index.shape)
print("edge_attr shape:", data.edge_attr.shape if hasattr(data, "edge_attr") else None)
print("Node target (y) shape:", getattr(data, "y", None).shape if hasattr(data, "y") else None)
print("Edge target (edge_y) shape:", getattr(data, "edge_y", None).shape if hasattr(data, "edge_y") else None)

# ----------------------- Device -----------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", end=space)

# ----------------------- Split data -----------------------
torch.manual_seed(0)
train_dataset, valid_dataset, _ = datasets.split_dataset(
    dataset,
    train_ratio=args.train_ratio,
    valid_ratio=1 - args.train_ratio
)

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(valid_dataset)}", end=space)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

# ----------------------- Model construction -----------------------
input_dim = dataset[0].x.shape[1]
edge_attr_dim = dataset[0].edge_attr.shape[1] if hasattr(dataset[0], "edge_attr") else 0

node_out_dim = dataset[0].y.shape[1]
edge_out_dim = getattr(dataset[0], "edge_y", None).shape[1] if hasattr(dataset[0], "edge_y") else 0

print(f"Input dim: {input_dim}")
print(f"Edge attr dim: {edge_attr_dim}")
print(f"Node out dim: {node_out_dim}")
print(f"Edge out dim: {edge_out_dim}")

# sanity check: expects 2 node outputs and 6 edge outputs in your current setup
if node_out_dim != 2:
    print(f"[WARN] Expected node_y dim == 2, got {node_out_dim}")
if hasattr(dataset[0], "edge_y") and edge_out_dim != 6:
    print(f"[WARN] Expected edge_y dim == 6, got {edge_out_dim}")

model_constructor_args = {
    'input_dim': input_dim,
    'hidden_dim': args.hidden_dim,
    'dropout_p': args.dropout_p,
    'layer_num': args.layer_num,
    'edge_attr_dim': edge_attr_dim,
    'aggr': args.aggr,
    'gnn_act': args.gnn_act,
    'gnn_dropout': args.gnn_dropout,
    'device': device,
    # IMPORTANT: names must match models.py
    'node_out_dim': node_out_dim,
    'edge_out_dim': edge_out_dim,
    # keep edge attributes by default for edge head
    'include_edge_attr_in_edge_head': True,
}
model = locals()[args.model](**model_constructor_args).to(device)

# ----------------------- Optimizer & Loss -----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Use the new combined losses
# Training: keep combined loss as "sum" so gradients scale with total error.
# Evaluation (node-only / edge-only): use "mean" so logged losses are
# average error per active DOF, similar to the pseudo version.
if args.loss_function.lower() in ('l1', 'l1_loss', 'nodeedge_l1'):
    # For backprop
    criterion_combined = losses.NodeEdgeL1Loss(
        node_weight=1.0, edge_weight=1.0, reduction="sum"
    )
    # For logging (training/validation metrics)
    criterion_node_only = losses.NodeEdgeL1Loss(
        node_weight=1.0, edge_weight=0.0, reduction="mean"
    )
    criterion_edge_only = losses.NodeEdgeL1Loss(
        node_weight=0.0, edge_weight=1.0, reduction="mean"
    )
elif args.loss_function.lower() in ('l2', 'l2_loss', 'nodeedge_l2'):
    # For backprop
    criterion_combined = losses.NodeEdgeL2Loss(
        node_weight=1.0, edge_weight=1.0, reduction="sum"
    )
    # For logging (training/validation metrics)
    criterion_node_only = losses.NodeEdgeL2Loss(
        node_weight=1.0, edge_weight=0.0, reduction="mean"
    )
    criterion_edge_only = losses.NodeEdgeL2Loss(
        node_weight=0.0, edge_weight=1.0, reduction="mean"
    )
else:
    raise ValueError(f"Unknown loss_function: {args.loss_function} (use 'L1' or 'L2')")

accuracy_record = np.zeros((3, args.epoch_num))
loss_record = np.zeros((3, args.epoch_num))

# NEW: separate tracking for node-only and edge-only metrics
accuracy_record_node = np.zeros((3, args.epoch_num))
accuracy_record_edge = np.zeros((3, args.epoch_num))
loss_record_node = np.zeros((3, args.epoch_num))
loss_record_edge = np.zeros((3, args.epoch_num))

best_accuracy = 0

print(model, end=space)

# ----------------------- Save Model -----------------------
task = 'Data_SAP2000'  # fixed task name for results path
save_model_dir = os.path.join('Results', task)
os.makedirs(save_model_dir, exist_ok=True)
save_model_dir = os.path.join(save_model_dir, date_str)
os.makedirs(save_model_dir, exist_ok=True)

def forward_model(model, data):
    """Return (node_out, edge_out) no matter how the model is implemented."""
    out = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    if isinstance(out, (tuple, list)):
        node_out = out[0]
        edge_out = out[1] if len(out) > 1 else None
    else:
        node_out, edge_out = out, None
    return node_out, edge_out

# ----------------------- Train and Validate -----------------------
start_time = time.time()
for epoch in range(args.epoch_num):
    # ---- Train ----
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        node_out, edge_out = forward_model(model, data)

        # Single combined loss (handles node + edge)
        loss = criterion_combined(
            node_out, data.y,
            edge_out=edge_out, edge_y=getattr(data, "edge_y", None),
            node_thresh=args.accuracy_threshold,
            edge_thresh=args.accuracy_threshold
        )

        loss.backward()
        optimizer.step()

    # ---- Evaluate ----
    model.eval()
    with torch.no_grad():
        split_metrics = {}
        for i, (split_name, loader) in enumerate([("train", train_loader), ("valid", valid_loader)]):
            loss_val_node = 0.0
            loss_val_edge = 0.0
            total_correct_node = 0
            total_elems_node = 0
            total_correct_edge = 0
            total_elems_edge = 0
            num_batches = 0

            for data in loader:
                num_batches += 1
                data = data.to(device)
                node_out, edge_out = forward_model(model, data)

                # Node metrics
                # (Use node-only criterion so the masking/weighting matches training)
                node_loss = criterion_node_only(
                    node_out, data.y,
                    edge_out=None, edge_y=None,
                    node_thresh=args.accuracy_threshold
                )
                loss_val_node += node_loss.item()

                # Accuracy uses your existing utility (on the requested slice)
                y_start, y_finish = datasets.get_target_index(args.target)
                node_pred_slice = node_out[:, y_start:y_finish]
                node_true_slice = data.y[:, y_start:y_finish]

                correct_n, elems_n = accuracy.node_accuracy(
                    node_pred_slice, node_true_slice, args.accuracy_threshold
                )
                total_correct_node += correct_n
                total_elems_node += elems_n

                # Edge metrics
                if hasattr(data, "edge_y") and (edge_out is not None):
                    edge_loss = criterion_edge_only(
                        node_out, data.y,
                        edge_out=edge_out, edge_y=data.edge_y,
                        edge_thresh=args.accuracy_threshold
                    )
                    loss_val_edge += edge_loss.item()

                    # If you have a dedicated edge_accuracy, use it; otherwise fallback.
                    if hasattr(accuracy, "edge_accuracy"):
                        correct_e, elems_e = accuracy.edge_accuracy(
                            edge_out, data.edge_y, args.accuracy_threshold
                        )
                    else:
                        correct_e, elems_e = accuracy.node_accuracy(
                            edge_out, data.edge_y, args.accuracy_threshold
                        )

                    total_correct_edge += correct_e
                    total_elems_edge += elems_e

            # Aggregate
            acc_node  = (total_correct_node / total_elems_node) if total_elems_node else 0.0
            loss_node = (loss_val_node / max(1, num_batches))  # avg over batches
            acc_edge  = (total_correct_edge / total_elems_edge) if total_elems_edge else 0.0
            loss_edge = (loss_val_edge / max(1, num_batches))  # avg over batches

            if total_elems_edge:
                combined_acc  = 0.5 * (acc_node + acc_edge)
                combined_loss = 0.5 * (loss_node + loss_edge)
            else:
                combined_acc  = acc_node
                combined_loss = loss_node

            # Record metrics: 0 -> train, 1 -> valid
            accuracy_record[i][epoch]       = combined_acc
            loss_record[i][epoch]           = combined_loss
            accuracy_record_node[i][epoch]  = acc_node
            accuracy_record_edge[i][epoch]  = acc_edge
            loss_record_node[i][epoch]      = loss_node
            loss_record_edge[i][epoch]      = loss_edge

            split_metrics[split_name] = dict(
                acc_node=acc_node, loss_node=loss_node,
                acc_edge=acc_edge, loss_edge=loss_edge,
                combined_acc=combined_acc, combined_loss=combined_loss
            )

    # Log
    tr = split_metrics["train"]
    va = split_metrics["valid"]
    print(
        f" Epoch: {epoch:03d} | "
        f"Train Acc (node/edge/comb): {tr['acc_node']:.4f}/{tr['acc_edge']:.4f}/{tr['combined_acc']:.4f} | "
        f"Valid Acc (node/edge/comb): {va['acc_node']:.4f}/{va['acc_edge']:.4f}/{va['combined_acc']:.4f} | "
        f"Train Loss (node/edge/comb): {tr['loss_node']:.4f}/{tr['loss_edge']:.4f}/{tr['combined_loss']:.4f} | "
        f"Valid Loss (node/edge/comb): {va['loss_node']:.4f}/{va['loss_edge']:.4f}/{va['combined_loss']:.4f}"
    )

    # Save best
    valid_combined_acc = split_metrics["valid"]["combined_acc"]
    if valid_combined_acc > best_accuracy:
        best_accuracy = valid_combined_acc
        torch.save(model.state_dict(), os.path.join(save_model_dir, 'model.pt'))
        print(f'Trained model saved, valid acc (combined): {best_accuracy:.4f}')

finish_time = time.time()
args.training_time = (finish_time - start_time)/60
print(space)
print(f"Time spent: {(finish_time - start_time)/60:.2f} min")
print("Finish time: " + datetime.now().strftime('%b %d, %H:%M:%S'), end=space)

with open(os.path.join(save_model_dir, 'training_args.json'), 'w') as f:
    json.dump(vars(args), f)

# Plot the results (combined metrics)
title = ', '.join([args.model, 'Data_SAP2000\n', date_str, args.target])

# Combined (node + edge) curves
plot.plot_learningCurve(accuracy_record, save_model_dir, title=title, target=args.target)
plot.plot_lossCurve(loss_record, save_model_dir, title=title, target=args.target)

# EXTRA: node vs edge accuracy and loss curves
plot.plot_node_edge_accuracy_curves(
    accuracy_record_node,
    accuracy_record_edge,
    save_model_dir,
    title=title,
    target=args.target,
)

plot.plot_node_edge_loss_curves(
    loss_record_node,
    loss_record_edge,
    save_model_dir,
    title=title,
    target=args.target,
)
