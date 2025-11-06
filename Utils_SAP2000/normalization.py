import torch
from torch_geometric.loader import DataLoader


# ---------------------------------------------
# NODE INPUT NORMALIZATION
# ---------------------------------------------
def getMinMax_x(dataset, norm_dict):
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    x, y, edge_attr = data.x, data.y, data.edge_attr

    # Node input features
    min_grid_num = 0
    max_grid_num = torch.max(torch.abs(x[:, :3]))
    norm_dict['grid_num'] = [min_grid_num, max_grid_num]

    min_coord = torch.min(torch.abs(x[:, 3:6]))
    max_coord = torch.max(torch.abs(x[:, 3:6]))
    norm_dict['coord'] = [min_coord, max_coord]

    min_mass = 0
    max_mass = torch.max(torch.abs(x[:, 8]))
    norm_dict['mass'] = [min_mass, max_mass]

    min_force = 0
    max_force = torch.max(torch.abs(x[:, 9:]))
    norm_dict['force'] = [min_force, max_force]

    # Edge attributes (length)
    min_length = 0
    max_length = torch.max(torch.abs(edge_attr[:, 2]))
    norm_dict['length'] = [min_length, max_length]

    del x, y, edge_attr
    return norm_dict


# ---------------------------------------------
# NODE & EDGE OUTPUT NORMALIZATION
# ---------------------------------------------
def getMinMax_y_linear(dataset, norm_dict):
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    y = data.y

    # Node outputs (2 displacement features)
    min_disp = 0
    max_disp = torch.max(torch.abs(y[:, 0:2]))
    norm_dict['disp'] = [min_disp, max_disp]

    del y

    # Edge outputs (6 total, split into two groups)
    if hasattr(data, 'edge_y'):
        edge_y = data.edge_y

        # First 3 = shear
        min_edge_shear = 0
        max_edge_shear = torch.max(torch.abs(edge_y[:, 0:3]))
        norm_dict['edge_shear'] = [min_edge_shear, max_edge_shear]

        # Last 3 = moment
        min_edge_moment = 0
        max_edge_moment = torch.max(torch.abs(edge_y[:, 3:6]))
        norm_dict['edge_moment'] = [min_edge_moment, max_edge_moment]

        del edge_y

    return norm_dict


# ---------------------------------------------
# APPLY NORMALIZATION
# ---------------------------------------------
def normalize_linear(data, norm_dict):
    # Node input features
    data.x[:, :3] = (data.x[:, :3] - norm_dict['grid_num'][0]) / (norm_dict['grid_num'][1] - norm_dict['grid_num'][0])
    data.x[:, 3:6] = (data.x[:, 3:6] - norm_dict['coord'][0]) / (norm_dict['coord'][1] - norm_dict['coord'][0])
    data.x[:, 8]   = (data.x[:, 8] - norm_dict['mass'][0]) / (norm_dict['mass'][1] - norm_dict['mass'][0])
    data.x[:, 9:]  = (data.x[:, 9:] - norm_dict['force'][0]) / (norm_dict['force'][1] - norm_dict['force'][0])

    # Edge input (length)
    data.edge_attr[:, 2] = (data.edge_attr[:, 2] - norm_dict['length'][0]) / (norm_dict['length'][1] - norm_dict['length'][0])

    # Node outputs (2 displacements)
    data.y[:, 0:2] = (data.y[:, 0:2] - norm_dict['disp'][0]) / (norm_dict['disp'][1] - norm_dict['disp'][0])

    # Edge outputs (6 total)
    if hasattr(data, 'edge_y'):
        # First 3 = shear
        data.edge_y[:, 0:3] = (data.edge_y[:, 0:3] - norm_dict['edge_shear'][0]) / (norm_dict['edge_shear'][1] - norm_dict['edge_shear'][0])
        # Last 3 = moment
        data.edge_y[:, 3:6] = (data.edge_y[:, 3:6] - norm_dict['edge_moment'][0]) / (norm_dict['edge_moment'][1] - norm_dict['edge_moment'][0])

    return data


# ---------------------------------------------
# DENORMALIZATION HELPERS
# ---------------------------------------------
def denormalize_disp(disp, norm_dict):
    disp = disp * (norm_dict['disp'][1] - norm_dict['disp'][0]) + norm_dict['disp'][0]
    return disp


def denormalize_edge_y(edge_y, norm_dict):
    # First 3 = shear
    edge_y[:, 0:3] = edge_y[:, 0:3] * (norm_dict['edge_shear'][1] - norm_dict['edge_shear'][0]) + norm_dict['edge_shear'][0]
    # Last 3 = moment
    edge_y[:, 3:6] = edge_y[:, 3:6] * (norm_dict['edge_moment'][1] - norm_dict['edge_moment'][0]) + norm_dict['edge_moment'][0]
    return edge_y


def denormalize_y_linear(y, norm_dict):
    y[:, 0:2] = y[:, 0:2] * (norm_dict['disp'][1] - norm_dict['disp'][0]) + norm_dict['disp'][0]
    return y


# ---------------------------------------------
# MAIN NORMALIZATION ENTRY POINTS
# ---------------------------------------------
def normalize_dataset(dataset, analysis='linear'):
    norm_dict = dict()
    norm_dict = getMinMax_x(dataset, norm_dict)
    norm_dict = getMinMax_y_linear(dataset, norm_dict)

    for data in dataset:
        normalize_linear(data, norm_dict)

    return dataset, norm_dict


def normalize_dataset_byNormDict(dataset, norm_dict, analysis='linear'):
    for data in dataset:
        normalize_linear(data, norm_dict)
    return dataset
