import torch
from torch.utils.data import random_split
import os

# -------------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------------
def get_dataset(structure_num=300):
    """
    Loads a list of PyTorch Geometric graph data objects.
    Each graph is expected at:
        Data_SAP2000/structure_<index>.pt
    """
    root = 'Data_SAP2000'

    data_list = []
    for index in range(1, structure_num + 1):
        structure_graph_path = os.path.join(root, f'structure_{index}.pt')

        if os.path.exists(structure_graph_path):
            graph = torch.load(structure_graph_path, weights_only=False)
            data_list.append(graph)
        else:
            print("No file:", structure_graph_path)

    return data_list


# -------------------------------------------------------------------------
# Split dataset
# -------------------------------------------------------------------------
def split_dataset(dataset, train_ratio=0.9, valid_ratio=0.1, test_ratio=None):
    """
    Splits the dataset into train, validation, and optional test sets.
    """
    length = len(dataset)
    if test_ratio is None:
        train_len = int(length * train_ratio)
        valid_len = length - train_len
        train_dataset, valid_dataset = random_split(
            dataset,
            [train_len, valid_len],
            generator=torch.Generator().manual_seed(731)
        )
        return train_dataset, valid_dataset, None
    else:
        train_len = int(length * train_ratio)
        valid_len = int(length * valid_ratio)
        test_len = length - train_len - valid_len
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset,
            [train_len, valid_len, test_len],
            generator=torch.Generator().manual_seed(731)
        )
        return train_dataset, valid_dataset, test_dataset


# -------------------------------------------------------------------------
# Target index mapping
# -------------------------------------------------------------------------
def get_target_index(target):
    """
    Returns the start and end indices for selecting node or edge target features.

    Node outputs:
        0: disp_x
        1: disp_y

    Edge outputs (for reference):
        0–2: edge_shear
        3–5: edge_moment
    """
    # Node-level targets (2 total)
    if target == 'disp_x':
        y_start, y_finish = 0, 1
    elif target == 'disp_y':
        y_start, y_finish = 1, 2
    elif target in ['disp', 'all', 'node_all']:
        y_start, y_finish = 0, 2

    # Edge-level targets (6 total)
    elif target == 'edge_shear':
        y_start, y_finish = 0, 3
    elif target == 'edge_moment':
        y_start, y_finish = 3, 6
    elif target in ['edge_all', 'edge']:
        y_start, y_finish = 0, 6

    else:
        raise ValueError(f"No such output target: '{target}'")

    return y_start, y_finish
