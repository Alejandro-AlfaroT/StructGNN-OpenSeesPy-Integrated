import torch
from torch_geometric.loader import DataLoader

# Small constant to avoid division-by-zero
_EPS = 1e-8

def _to_dev_dtype(v, ref_tensor):
    """Convert scalar/tensor v to the device/dtype of ref_tensor."""
    return torch.as_tensor(v, device=ref_tensor.device, dtype=ref_tensor.dtype)

def _safe_range(lo, hi, ref_tensor):
    """
    Returns (lo_t, den_t) on the proper device/dtype, with denominator clamped >= _EPS.
    lo, hi can be Python floats or tensors.
    """
    lo_t = _to_dev_dtype(lo, ref_tensor)
    hi_t = _to_dev_dtype(hi, ref_tensor)
    den_t = (hi_t - lo_t).abs().clamp_min(_EPS)
    return lo_t, den_t


# ---------------------------------------------
# NODE INPUT NORMALIZATION (0 .. max|·|)
# ---------------------------------------------
def getMinMax_x(dataset, norm_dict):
    # Load a single concatenated batch across the whole dataset
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    x = data.x
    edge_attr = data.edge_attr

    # grid index-like features (first 3 columns)
    min_grid_num = 0.0
    max_grid_num = torch.max(torch.abs(x[:, :3]))
    norm_dict['grid_num'] = [min_grid_num, max_grid_num]

    # coordinates (cols 3:6) — keep your original absolute scaling intent, min=0
    min_coord = 0.0
    max_coord = torch.max(torch.abs(x[:, 3:6]))
    norm_dict['coord'] = [min_coord, max_coord]

    # mass (col 8)
    min_mass = 0.0
    max_mass = torch.max(torch.abs(x[:, 8]))
    norm_dict['mass'] = [min_mass, max_mass]

    # forces (cols 9:)
    min_force = 0.0
    max_force = torch.max(torch.abs(x[:, 9:])) if x.shape[1] > 9 else torch.tensor(1.0, device=x.device, dtype=x.dtype)
    norm_dict['force'] = [min_force, max_force]

    # Edge attributes (length at col index 2)
    if edge_attr is not None and edge_attr.shape[1] > 2:
        min_length = 0.0
        max_length = torch.max(torch.abs(edge_attr[:, 2]))
    else:
        # Fallback if edge_attr missing or too short
        min_length = 0.0
        max_length = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    norm_dict['length'] = [min_length, max_length]

    del x, edge_attr
    return norm_dict


# ---------------------------------------------
# NODE & EDGE OUTPUT NORMALIZATION (0 .. max|·|)
# ---------------------------------------------
def getMinMax_y_linear(dataset, norm_dict):
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))

    # Node outputs (first 2 displacement features)
    y = data.y
    min_disp = 0.0
    max_disp = torch.max(torch.abs(y[:, 0:2]))
    norm_dict['disp'] = [min_disp, max_disp]
    del y

    # Edge outputs (6 total, split into two groups: 0:3 shear, 3:6 moment)
    if hasattr(data, 'edge_y') and data.edge_y is not None:
        edge_y = data.edge_y

        min_edge_shear = 0.0
        max_edge_shear = torch.max(torch.abs(edge_y[:, 0:3]))
        norm_dict['edge_shear'] = [min_edge_shear, max_edge_shear]

        min_edge_moment = 0.0
        max_edge_moment = torch.max(torch.abs(edge_y[:, 3:6]))
        norm_dict['edge_moment'] = [min_edge_moment, max_edge_moment]

        del edge_y

    return norm_dict


# ---------------------------------------------
# APPLY NORMALIZATION (safe denominators)
# ---------------------------------------------
def normalize_linear(data, norm_dict):
    # Node input features
    lo, den = _safe_range(*norm_dict['grid_num'], ref_tensor=data.x)
    data.x[:, :3] = (data.x[:, :3] - lo) / den

    lo, den = _safe_range(*norm_dict['coord'], ref_tensor=data.x)
    data.x[:, 3:6] = (data.x[:, 3:6] - lo) / den

    lo, den = _safe_range(*norm_dict['mass'], ref_tensor=data.x)
    data.x[:, 8] = (data.x[:, 8] - lo) / den

    if data.x.shape[1] > 9:
        lo, den = _safe_range(*norm_dict['force'], ref_tensor=data.x)
        data.x[:, 9:] = (data.x[:, 9:] - lo) / den

    # Edge input (length)
    if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.shape[1] > 2:
        lo, den = _safe_range(*norm_dict['length'], ref_tensor=data.edge_attr)
        data.edge_attr[:, 2] = (data.edge_attr[:, 2] - lo) / den

    # Node outputs (2 displacements)
    if hasattr(data, 'y') and data.y is not None and data.y.shape[1] >= 2:
        lo, den = _safe_range(*norm_dict['disp'], ref_tensor=data.y)
        data.y[:, 0:2] = (data.y[:, 0:2] - lo) / den

    # Edge outputs (6 total)
    if hasattr(data, 'edge_y') and data.edge_y is not None and data.edge_y.shape[1] >= 6:
        lo, den = _safe_range(*norm_dict['edge_shear'], ref_tensor=data.edge_y)
        data.edge_y[:, 0:3] = (data.edge_y[:, 0:3] - lo) / den

        lo, den = _safe_range(*norm_dict['edge_moment'], ref_tensor=data.edge_y)
        data.edge_y[:, 3:6] = (data.edge_y[:, 3:6] - lo) / den

    return data


# ---------------------------------------------
# DENORMALIZATION HELPERS (mirror the above)
# ---------------------------------------------
def denormalize_disp(disp, norm_dict):
    lo, den = norm_dict['disp']
    # den here is hi - lo; reproduce original formula: x = norm * (hi - lo) + lo
    return disp * (_to_dev_dtype(den, disp) - _to_dev_dtype(0.0, disp)) + _to_dev_dtype(lo, disp)

def denormalize_edge_y(edge_y, norm_dict):
    # First 3 = shear
    lo, hi = norm_dict['edge_shear']
    edge_y[:, 0:3] = edge_y[:, 0:3] * (_to_dev_dtype(hi, edge_y) - _to_dev_dtype(lo, edge_y)) + _to_dev_dtype(lo, edge_y)
    # Last 3 = moment
    lo, hi = norm_dict['edge_moment']
    edge_y[:, 3:6] = edge_y[:, 3:6] * (_to_dev_dtype(hi, edge_y) - _to_dev_dtype(lo, edge_y)) + _to_dev_dtype(lo, edge_y)
    return edge_y

def denormalize_y_linear(y, norm_dict):
    lo, hi = norm_dict['disp']
    y[:, 0:2] = y[:, 0:2] * (_to_dev_dtype(hi, y) - _to_dev_dtype(lo, y)) + _to_dev_dtype(lo, y)
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
