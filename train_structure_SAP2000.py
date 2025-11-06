import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from argparse import ArgumentParser

from GNN.models import *
from GNN import losses
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
# general args
# Dataset
parser.add_argument('--dataset_name', dest='dataset_name', default='Static_Linear_Analysis', type=str)
parser.add_argument('--whatAsNode', dest='whatAsNode', default='NodeAsNode', type=str)
parser.add_argument('--data_num', dest='data_num', default=2000, type=int)
parser.add_argument('--train_ratio', dest='train_ratio', default=0.9, type=float)
parser.add_argument('--normalization', dest='normalization', default=True, type=bool)

# GNN
parser.add_argument('--model', dest='model', default='Structure_GraphNetwork', type=str, help='model class name. E.g., GCN, Tony_GNN, ...')
parser.add_argument('--aggr_type', dest='aggr', default='mean', type=str)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=256, type=int)
parser.add_argument('--layer_num', dest='layer_num', default=9, type=int)
parser.add_argument('--dropout_p', dest='dropout_p', default=0.0, type=float)
parser.add_argument('--gnn_dropout', dest='gnn_dropout', default=True, type=bool, help='whether dropout, default 0.5')
parser.add_argument('--gnn_act', dest='gnn_act', default=True, type=bool, help='whether activation, default RELU')

# Training
parser.add_argument('--target', dest='target', default='all', type=str, help='which output target you are going to train, like displacement, moment, shear......')
parser.add_argument('--epoch_num', dest='epoch_num', default=10, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)
parser.add_argument('--lr', dest='lr', default=5e-5, type=float)
parser.add_argument('--loss_function', dest='loss_function', default='L1_Loss', type=str)
parser.add_argument('--accuracy_threshold', dest='accuracy_threshold', default=1e-4, type=float, help='The normalized output value smaller than the threshold will be ignored.')

# Training
parser.add_argument('--training_time', dest='training_time', default=0, type=float)

args = parser.parse_args()
print(args, end=space)

torch.manual_seed(0)
date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
print("Start time: " + datetime.now().strftime('%b %d, %H:%M:%S'), end=space)

# Load in data
dataset = datasets.get_dataset(dataset_name=args.dataset_name,
                               whatAsNode=args.whatAsNode,
                               structure_num=args.data_num)
print(f"dataset: {args.dataset_name}")

# Normalization
if args.normalization:
    dataset, norm_dict = normalization.normalize_dataset(dataset, analysis='linear')
    print("Normalization: True")
    print("Normlization dictionary: ")
    print(norm_dict, end=space)

# Get the first graph object.
data = dataset[0]
print("x shape:", data.x.shape)
print(data.x[:5])
print("edge_index:", data.edge_index.shape)
print("edge_attr shape:", data.edge_attr.shape if hasattr(data, "edge_attr") else None)
print("Node target (y) shape:", getattr(data, "y", None).shape if hasattr(data, "y") else None)
print("Edge target (edge_y) shape:", getattr(data, "edge_y", None).shape if hasattr(data, "edge_y") else None)

# Device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", end=space)

# Split data into train, test set
torch.manual_seed(0)
train_dataset, valid_dataset, _ = datasets.split_dataset(dataset,
                                                         train_ratio=args.train_ratio,
                                                         valid_ratio=1-args.train_ratio)

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(valid_dataset)}", end=space)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

# Training targets (node-level slice)
y_start, y_finish = datasets.get_target_index(args.target)

# Model setup
input_dim = data.x.shape[1]
edge_attr_dim = data.edge_attr.shape[1] if hasattr(data, "edge_attr") else 0  # dynamic
node_output_dim = data.y.shape[1] if hasattr(data, "y") else 0               # should be 2 now
edge_output_dim = getattr(data, "edge_y", None).shape[1] if hasattr(data, "edge_y") else 0  # should be 6 if provided

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
    # If your model accepts these, great; if not, they'll be ignored if you remove them.
    'node_output_dim': node_output_dim,
    'edge_output_dim': edge_output_dim,
}
model = locals()[args.model](**model_constructor_args).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# criterion
if args.loss_function == 'L1_Loss':
    criterion = losses.L1_Loss()
elif args.loss_function == 'L2_Loss':
    criterion = losses.L2_Loss()
else:
    raise ValueError(f"Unknown loss_function: {args.loss_function}")

accuracy_record = np.zeros((3, args.epoch_num))
loss_record = np.zeros((3, args.epoch_num))
best_accuracy = 0

print(model, end=space)

# Save Model
task = args.dataset_name
save_model_dir = 'Results/' + task + '/'
if os.path.exists(save_model_dir) is False:
    os.mkdir(save_model_dir)
save_model_dir += date_str + '/'
os.mkdir(save_model_dir)

def forward_model(model, data):
    """Return (node_out, edge_out) no matter how the model is implemented."""
    out = model(data.x, data.edge_index, data.edge_attr)
    if isinstance(out, (tuple, list)):
        if len(out) >= 2:
            node_out, edge_out = out[0], out[1]
        else:
            node_out, edge_out = out[0], None
    else:
        node_out, edge_out = out, None
    return node_out, edge_out

# Train and Test
start_time = time.time()
for epoch in range(args.epoch_num):
    # -------------------- Train --------------------
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        node_out, edge_out = forward_model(model, data)

        # Node loss
        node_loss = criterion(
            node_out[:, y_start:y_finish],
            data.y[:, y_start:y_finish],
            args.accuracy_threshold
        )

        # Edge loss (if available)
        if hasattr(data, "edge_y") and edge_out is not None:
            edge_loss = criterion(edge_out, data.edge_y, args.accuracy_threshold)
            loss = node_loss + edge_loss
        else:
            loss = node_loss

        loss.backward()
        optimizer.step()

    # -------------------- Evaluate --------------------
    model.eval()
    with torch.no_grad():
        # To print per-split detailed metrics, capture both splits
        split_metrics = {}
        for i, (split_name, loader) in enumerate([("train", train_loader), ("valid", valid_loader)]):
            loss_val_node = 0.0
            loss_val_edge = 0.0
            total_correct_node = 0
            total_elems_node = 0
            total_correct_edge = 0
            total_elems_edge = 0

            for data in loader:
                data = data.to(device)
                node_out, edge_out = forward_model(model, data)

                # Node metrics
                node_loss = criterion(
                    node_out[:, y_start:y_finish],
                    data.y[:, y_start:y_finish],
                    args.accuracy_threshold
                )
                loss_val_node += node_loss.item()

                correct_n, elems_n = accuracy.node_accuracy(
                    node_out[:, y_start:y_finish],
                    data.y[:, y_start:y_finish],
                    args.accuracy_threshold
                )
                total_correct_node += correct_n
                total_elems_node += elems_n

                # Edge metrics
                if hasattr(data, "edge_y") and edge_out is not None:
                    edge_loss = criterion(edge_out, data.edge_y, args.accuracy_threshold)
                    loss_val_edge += edge_loss.item()

                    if hasattr(accuracy, "edge_accuracy"):
                        correct_e, elems_e = accuracy.edge_accuracy(edge_out, data.edge_y, args.accuracy_threshold)
                    else:
                        # Reuse node_accuracy utility for edge tensors (same [N, D] shape)
                        correct_e, elems_e = accuracy.node_accuracy(edge_out, data.edge_y, args.accuracy_threshold)

                    total_correct_edge += correct_e
                    total_elems_edge += elems_e

            # Aggregate
            acc_node  = (total_correct_node / total_elems_node) if total_elems_node else 0.0
            loss_node = (loss_val_node / total_elems_node) if total_elems_node else 0.0
            acc_edge  = (total_correct_edge / total_elems_edge) if total_elems_edge else 0.0
            loss_edge = (loss_val_edge / total_elems_edge) if total_elems_edge else 0.0

            # Combined metrics (average node & edge when edges exist)
            if total_elems_edge:
                combined_acc  = 0.5 * (acc_node + acc_edge)
                combined_loss = 0.5 * (loss_node + loss_edge)
            else:
                combined_acc  = acc_node
                combined_loss = loss_node

            # Save for plotting (keep prior shape/meaning: [0]=train, [1]=valid)
            accuracy_record[i][epoch] = combined_acc
            loss_record[i][epoch] = combined_loss

            split_metrics[split_name] = dict(
                acc_node=acc_node, loss_node=loss_node,
                acc_edge=acc_edge, loss_edge=loss_edge,
                combined_acc=combined_acc, combined_loss=combined_loss
            )

    # Pretty print epoch summary
    tr = split_metrics["train"]
    va = split_metrics["valid"]
    print(
        f" Epoch: {epoch:03d} | "
        f"Train Acc (node/edge/comb): {tr['acc_node']:.4f}/{tr['acc_edge']:.4f}/{tr['combined_acc']:.4f} | "
        f"Valid Acc (node/edge/comb): {va['acc_node']:.4f}/{va['acc_edge']:.4f}/{va['combined_acc']:.4f} | "
        f"Train Loss (node/edge/comb): {tr['loss_node']:.4f}/{tr['loss_edge']:.4f}/{tr['combined_loss']:.4f} | "
        f"Valid Loss (node/edge/comb): {va['loss_node']:.4f}/{va['loss_edge']:.4f}/{va['combined_loss']:.4f}"
    )

    # Save best model by combined valid accuracy
    valid_combined_acc = split_metrics["valid"]["combined_acc"]
    if valid_combined_acc > best_accuracy:
        best_accuracy = valid_combined_acc
        torch.save(model.state_dict(), save_model_dir + 'model.pt')
        print(f'Trained model saved, valid acc (combined): {best_accuracy:.4f}')

finish_time = time.time()
args.training_time = (finish_time - start_time)/60
print(space)
print(f"Time spent: {(finish_time - start_time)/60:.2f} min")
print("Finish time: " + datetime.now().strftime('%b %d, %H:%M:%S'), end=space)

with open(save_model_dir + 'training_args.json', 'w') as f:
    json.dump(vars(args), f)

# Plot the results (combined metrics)
plot.plot_learningCurve(accuracy_record, save_model_dir, title=', '.join([args.model, task+'\n', date_str, args.target]), target=args.target)
plot.plot_lossCurve(loss_record, save_model_dir, title=', '.join([args.model, task+'\n', date_str, args.target]), target=args.target)
