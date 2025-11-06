import os
import csv
import torch
from torch_geometric.data import Data

def _next_version_path(save_dir, stem="structure", ext=".pt"):
    """Return a unique path in save_dir by incrementing a plain integer suffix if needed."""
    base = os.path.join(save_dir, f"{stem}{ext}")
    if not os.path.exists(base):
        return base
    i = 1
    while True:
        cand = os.path.join(save_dir, f"{stem}_{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1


def update_output(structure_id: int = 1,
                  in_root: str = "Data_Gen_Manual",
                  out_root: str = "Data_SAP2000"):
    # ---------- Paths ----------
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../StructGNN/MP_TESTING
    structgnn_dir = os.path.dirname(script_dir)              # .../StructGNN

    # Read input graph from StructGNN/Data_Gen_Manual/structure_{id}/...
    path_in = os.path.join(structgnn_dir, in_root, f"structure_{structure_id}", "structure_graph.pt")

    # ✅ CSVs live ONE LEVEL UP from this script (the folder outside)
    parent_dir = os.path.dirname(script_dir)
    path_csv_nodes = os.path.join(parent_dir, "joint_displacements.csv")
    path_csv_edges = os.path.join(parent_dir, "frame_joint_forces.csv")

    # Optional: fail fast if files aren’t where we think
    if not os.path.isfile(path_csv_nodes):
        raise FileNotFoundError(f"Couldn't find node CSV at: {path_csv_nodes}")
    if not os.path.isfile(path_csv_edges):
        raise FileNotFoundError(f"Couldn't find edge CSV at: {path_csv_edges}")

    # ✅ Save modified graphs into StructGNN/Data_SAP2000/
    save_dir = os.path.join(structgnn_dir, out_root)
    os.makedirs(save_dir, exist_ok=True)
    path_out = _next_version_path(save_dir)  # <--- auto-increment plain integer filename

    # ----- Load graph -----
    data: Data = torch.load(path_in, weights_only=False)
    assert hasattr(data, "y"), "This graph has no 'y' tensor."
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    print(f"[structure_{structure_id}] num_nodes:", num_nodes, "| old y shape:", tuple(data.y.shape))
    print(f"[structure_{structure_id}] num_edges:", num_edges, "| has edge_y:", hasattr(data, "edge_y"))

    if data.y.dim() != 2 or data.y.size(1) < 2:
        raise ValueError(f"Expected data.y with at least 2 columns, got shape {tuple(data.y.shape)}")

    # ---------- Update NODE outputs from joint_displacements.csv ----------
    with open(path_csv_nodes, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            raw_headers = next(reader)
        except StopIteration:
            raise ValueError("Node CSV appears empty.")

        headers = [h.strip() for h in raw_headers]
        try:
            idx_joint = headers.index("Joint Name")
            idx_ux    = headers.index("Displacement X")
            idx_uy    = headers.index("Displacement Y")
        except ValueError as e:
            raise ValueError(
                f"Node CSV headers found: {headers}\n"
                "Expected: 'Joint Name', 'Displacement X', 'Displacement Y'"
            ) from e

        parsed_nodes = []
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            row = [c.strip() for c in row]
            ux = float(row[idx_ux])
            uy = float(row[idx_uy])
            parsed_nodes.append((ux, uy))

    if len(parsed_nodes) > num_nodes:
        raise ValueError(f"Node CSV has {len(parsed_nodes)} joints, but graph has only {num_nodes} nodes.")

    new_y = data.y.clone()
    new_y[:, 0] = 0.0  # UX
    new_y[:, 1] = 0.0  # UY
    for i, (ux, uy) in enumerate(parsed_nodes):
        new_y[i, 0] = ux
        new_y[i, 1] = uy
    data.y = new_y

    # ---------- Update EDGE outputs from frame_joint_forces.csv (last 6 numeric columns) ----------
    with open(path_csv_edges, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Edge CSV appears empty.")

    def _looks_like_header(cells):
        for c in cells:
            c = (c or "").strip()
            if c == "":
                continue
            try:
                float(c)
            except ValueError:
                return True
        return False

    start_idx = 1 if _looks_like_header(rows[0]) else 0
    data_rows = [r for r in rows[start_idx:] if r and any(c.strip() for c in r)]

    parsed_edges = []
    for r in data_rows:
        nums = []
        for c in r:
            c = (c or "").strip()
            if c == "":
                continue
            try:
                nums.append(float(c))
            except ValueError:
                continue
        if len(nums) < 6:
            raise ValueError("Edge CSV row has fewer than 6 numeric values; cannot take last 6.")
        parsed_edges.append(nums[-6:])

    if len(parsed_edges) < num_edges:
        raise ValueError(
            f"Edge CSV has only {len(parsed_edges)} usable rows, but graph has {num_edges} edges. "
            "Add more rows or verify edge ordering."
        )

    parsed_edges = parsed_edges[:num_edges]
    data.edge_y = torch.tensor(parsed_edges, dtype=torch.float32)
    print(f"[structure_{structure_id}] new edge_y shape:", tuple(data.edge_y.shape))

    # ---------- Save ----------
    torch.save(data, path_out)
    print(f"[structure_{structure_id}] new y shape:", tuple(data.y.shape), f"→ saved to {path_out}")

if __name__ == "__main__":
    update_output()
