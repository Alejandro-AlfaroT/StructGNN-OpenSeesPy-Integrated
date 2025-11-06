import os
import csv
import torch
from torch_geometric.data import Data

def update_output():

    # ---------- Paths ----------
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../StructGNN/MP_TESTING
    structgnn_dir = os.path.dirname(script_dir)              # .../StructGNN

    # Read input graph from StructGNN/Data_Gen_Manual/...
    path_in  = os.path.join(structgnn_dir, "Data_Gen_Manual", "structure_1", "structure_graph.pt")

    # ✅ CSVs live ONE LEVEL UP from this script (the folder outside)
    parent_dir = os.path.dirname(script_dir)  # same as structgnn_dir here, but explicit for clarity
    path_csv_nodes = os.path.join(parent_dir, "joint_displacements.csv")
    path_csv_edges = os.path.join(parent_dir, "frame_joint_forces.csv")

    # Optional: fail fast if files aren’t where we think
    if not os.path.isfile(path_csv_nodes):
        raise FileNotFoundError(f"Couldn't find node CSV at: {path_csv_nodes}")
    if not os.path.isfile(path_csv_edges):
        raise FileNotFoundError(f"Couldn't find edge CSV at: {path_csv_edges}")

    # Save modified graph into a new folder inside StructGNN
    save_dir = os.path.join(structgnn_dir, "Modified_Graphs")
    os.makedirs(save_dir, exist_ok=True)
    path_out = os.path.join(save_dir, "graph_modified.pt")

    # Load graph
    data: Data = torch.load(path_in, weights_only=False)
    assert hasattr(data, "y"), "This graph has no 'y' tensor."
    num_nodes = data.num_nodes
    num_edges = data.num_edges  # counts directed edges in PyG
    print("num_nodes:", num_nodes, "| old y shape:", tuple(data.y.shape))
    print("num_edges:", num_edges, "| has edge_y:", hasattr(data, "edge_y"))

    if data.y.dim() != 2 or data.y.size(1) < 2:
        raise ValueError(f"Expected data.y with at least 2 columns, got shape {tuple(data.y.shape)}")

    # ---------- Update NODE outputs from joint_displacements.csv (existing logic) ----------
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
        new_y[i, 0] = ux  # UX
        new_y[i, 1] = uy  # UY
    data.y = new_y

    # ---------- NEW: Update EDGE outputs from frame_joint_forces.csv ----------
    # Rule: take the LAST 6 numeric columns in each non-empty row as the 6 edge outputs.
    with open(path_csv_edges, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        # Try to read headers; if the file has no headers this still works (we just treat them as data)
        rows = list(reader)

    if not rows:
        raise ValueError("Edge CSV appears empty.")

    # If the first row contains any non-numeric cell (besides blanks), assume it's a header row and skip it.
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
        # Coerce cells to floats where possible
        nums = []
        for c in r:
            c = (c or "").strip()
            if c == "":
                continue
            try:
                nums.append(float(c))
            except ValueError:
                # ignore non-numeric cells
                continue
        if len(nums) < 6:
            # Not enough numeric columns, skip or raise
            raise ValueError("Edge CSV row has fewer than 6 numeric values; cannot take last 6.")
        last6 = nums[-6:]
        parsed_edges.append(last6)

    if len(parsed_edges) < num_edges:
        raise ValueError(
            f"Edge CSV has only {len(parsed_edges)} usable rows, but graph has {num_edges} edges. "
            "Add more rows or verify edge ordering."
        )

    # Trim or use first num_edges rows to match the graph
    parsed_edges = parsed_edges[:num_edges]

    edge_y_new = torch.tensor(parsed_edges, dtype=torch.float32)

    # Create or overwrite data.edge_y with shape [num_edges, 6]
    data.edge_y = edge_y_new

    print("new edge_y shape:", tuple(data.edge_y.shape))

    # ---------- Save ----------
    torch.save(data, path_out)
    print("new y shape:", tuple(data.y.shape), "→ saved to", path_out)

    # Quick sanity checks
    for i in range(min(5, len(parsed_nodes))):
        print(f"node {i}: UX={data.y[i,0].item()}, UY={data.y[i,1].item()}")

    for e in range(min(5, num_edges)):
        vals = ", ".join(f"{v:.4g}" for v in data.edge_y[e].tolist())
        print(f"edge {e}: last6=[{vals}]")

if __name__ == "__main__":
    update_output()
