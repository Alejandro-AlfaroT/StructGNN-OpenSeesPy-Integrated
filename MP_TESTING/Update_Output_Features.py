import os
import csv
import torch
from torch_geometric.data import Data  # type hints only

def update_output():

    # ---------- Paths ----------
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../StructGNN/MP_TESTING
    structgnn_dir = os.path.dirname(script_dir)              # .../StructGNN

    # Read input graph from StructGNN/Data_Gen_Manual/...
    path_in  = os.path.join(structgnn_dir, "Data_Gen_Manual", "structure_1", "structure_graph.pt")

    # CSV stays where your script is (MP_TESTING) unless you move it
    path_csv = os.path.join(script_dir, "joint_displacements.csv")

    # Save modified graph into a new folder inside StructGNN
    save_dir = os.path.join(structgnn_dir, "Modified_Graphs")
    os.makedirs(save_dir, exist_ok=True)
    path_out = os.path.join(save_dir, "graph_modified.pt")

    # ---------- Load graph (trusted file) ----------
    data: Data = torch.load(path_in, weights_only=False)
    assert hasattr(data, "y"), "This graph has no 'y' tensor."
    num_nodes = data.num_nodes
    print("num_nodes:", num_nodes, "| old y shape:", tuple(data.y.shape))

    if data.y.dim() != 2 or data.y.size(1) < 2:
        raise ValueError(f"Expected data.y with at least 2 columns, got shape {tuple(data.y.shape)}")

    # ---------- Read CSV (trim headers/cells) ----------
    with open(path_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            raw_headers = next(reader)
        except StopIteration:
            raise ValueError("CSV appears empty.")

        headers = [h.strip() for h in raw_headers]
        try:
            idx_joint = headers.index("Joint Name")
            idx_ux    = headers.index("Displacement X")
            idx_uy    = headers.index("Displacement Y")
        except ValueError as e:
            raise ValueError(f"CSV headers found: {headers}\n"
                             "Expected: 'Joint Name', 'Displacement X', 'Displacement Y'") from e

        parsed = []
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            row = [c.strip() for c in row]
            ux = float(row[idx_ux])
            uy = float(row[idx_uy])
            parsed.append((ux, uy))

    # ---------- Validate counts ----------
    if len(parsed) > num_nodes:
        raise ValueError(f"CSV has {len(parsed)} joints, but graph has only {num_nodes} nodes.")

    # ---------- Update only UX/ UY; keep other outputs unchanged ----------
    new_y = data.y.clone()
    new_y[:, 0] = 0.0  # UX
    new_y[:, 1] = 0.0  # UY
    for i, (ux, uy) in enumerate(parsed):
        new_y[i, 0] = ux  # UX
        new_y[i, 1] = uy  # UY
    data.y = new_y

    # ---------- Save ----------
    torch.save(data, path_out)
    print("new y shape:", tuple(data.y.shape), "→ saved to", path_out)

    # Quick sanity check (first 5)
    for i in range(min(5, len(parsed))):
        print(f"node {i}: UX={data.y[i,0].item()}, UY={data.y[i,1].item()}")

if __name__ == "__main__":
    update_output()
