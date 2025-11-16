import os
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils.convert import to_networkx
import networkx as nx


# -------------------------------------------------------------------------
# General utilities
# -------------------------------------------------------------------------
def print_space():
    """Visual separator for console output."""
    return "\n" * 3 + "=" * 100 + "\n" * 3


def moving_average(record, half_length=10):
    """
    Safe adaptive moving average.
    Automatically reduces the smoothing window for short training runs.
    If the window would cover nearly the whole series, no smoothing is applied.
    """
    record = np.asarray(record, dtype=np.float32)
    n = len(record)
    if n == 0:
        return record

    # Cap the window to at most ~n/5 on each side
    half = min(max(1, half_length), max(1, (n - 1) // 5))
    if (2 * half + 1) >= n:
        return record

    avg = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        avg[i] = record[start:end].mean()
    return avg


# -------------------------------------------------------------------------
# Learning and loss curves (combined; optional node/edge overlays)
# -------------------------------------------------------------------------
def _maybe_plot_curve(series, label):
    """Helper: plot a curve if it’s non-empty."""
    if series is None:
        return
    arr = np.asarray(series, dtype=np.float32)
    if arr.size == 0:
        return
    epochs = np.arange(1, len(arr) + 1)
    plt.plot(epochs, moving_average(arr), label=label)


def plot_learningCurve(
    accuracy_record,
    save_model_dir,
    title=None,
    target="node_edge",
    # Optional separate curves (each expected shape: (3, epochs) for [train, valid, test])
    accuracy_record_node=None,
    accuracy_record_edge=None,
):
    """
    Plots training/validation accuracy curves.

    accuracy_record: np.array with shape (3, epochs) for [train, valid, test]
                     representing the **combined (node+edge)** accuracy.
    accuracy_record_node / accuracy_record_edge: optional arrays with the same shape
                     if you also log node-only and edge-only accuracies.
    """
    train_acc, valid_acc, test_acc = accuracy_record

    plt.figure(figsize=(8, 6))

    # Title w/ best validation combined accuracy
    best_val = float(np.max(valid_acc)) if len(valid_acc) else 0.0
    full_title = (title or "Learning Curve") + f"\nBest Validation (Combined): {best_val * 100:.1f}%"

    # Combined curves
    _maybe_plot_curve(train_acc, "Train (Combined)")
    _maybe_plot_curve(valid_acc, "Validation (Combined)")
    if len(test_acc) and np.any(test_acc):
        _maybe_plot_curve(test_acc, "Test (Combined)")

    # Optional node-only / edge-only overlays
    if accuracy_record_node is not None:
        n_tr, n_va, n_te = accuracy_record_node
        _maybe_plot_curve(n_tr, "Train (Node)")
        _maybe_plot_curve(n_va, "Validation (Node)")
        if len(n_te) and np.any(n_te):
            _maybe_plot_curve(n_te, "Test (Node)")

    if accuracy_record_edge is not None:
        e_tr, e_va, e_te = accuracy_record_edge
        _maybe_plot_curve(e_tr, "Train (Edge)")
        _maybe_plot_curve(e_va, "Validation (Edge)")
        if len(e_te) and np.any(e_te):
            _maybe_plot_curve(e_te, "Test (Edge)")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (Fraction) — Combined (Node + Edge)")
    plt.ylim([-0.05, 1.05])
    plt.title(full_title)
    plt.tight_layout()
    out_path = os.path.join(save_model_dir, f"LearningCurve_{target}.png")
    plt.savefig(out_path)
    plt.close()


def plot_lossCurve(
    loss_record,
    save_model_dir,
    title=None,
    target="node_edge",
    # Optional separate curves (each expected shape: (3, epochs) for [train, valid, test])
    loss_record_node=None,
    loss_record_edge=None,
):
    """
    Plots training/validation loss curves.

    loss_record: np.array with shape (3, epochs) for [train, valid, test]
                 representing the **combined (node+edge)** loss.
    loss_record_node / loss_record_edge: optional arrays with the same shape
                 if you also log node-only and edge-only losses.
    """
    train_loss, valid_loss, test_loss = loss_record

    plt.figure(figsize=(8, 6))

    # Combined curves
    _maybe_plot_curve(train_loss, "Train (Combined)")
    _maybe_plot_curve(valid_loss, "Validation (Combined)")
    if len(test_loss) and np.any(test_loss):
        _maybe_plot_curve(test_loss, "Test (Combined)")

    # Optional node-only / edge-only overlays
    if loss_record_node is not None:
        n_tr, n_va, n_te = loss_record_node
        _maybe_plot_curve(n_tr, "Train (Node)")
        _maybe_plot_curve(n_va, "Validation (Node)")
        if len(n_te) and np.any(n_te):
            _maybe_plot_curve(n_te, "Test (Node)")

    if loss_record_edge is not None:
        e_tr, e_va, e_te = loss_record_edge
        _maybe_plot_curve(e_tr, "Train (Edge)")
        _maybe_plot_curve(e_va, "Validation (Edge)")
        if len(e_te) and np.any(e_te):
            _maybe_plot_curve(e_te, "Test (Edge)")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("Loss — Combined (Node + Edge)")
    plt.title(title or "Loss Curve (Combined)")
    plt.tight_layout()
    out_path = os.path.join(save_model_dir, f"LossCurve_{target}.png")
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------------------
# Accuracy distribution plots
# -------------------------------------------------------------------------
def plot_accuracy_distribution(y_pred, y_real, save_model_dir, target=None, max_value=None, threshold=0.0):
    """
    Plots error and distribution histograms for node or edge predictions.
    Works for both node (2D) and edge (6D) outputs.
    """
    y_pred = np.array(y_pred)
    y_real = np.array(y_real)

    mask = np.abs(y_real) > threshold * (max_value or np.max(np.abs(y_real)))
    y_pred = y_pred[mask]
    y_real = y_real[mask]

    error = np.divide(np.abs(y_pred - y_real), np.abs(y_real) + 1e-9)

    # Boxplot of relative errors
    plt.figure(figsize=(8, 4))
    plt.boxplot(error[np.isfinite(error)], vert=False, showfliers=True)
    plt.xlabel("Relative Absolute Error")
    plt.title(f"Error Distribution - {target or 'output'}")
    plt.tight_layout()
    out_path = os.path.join(save_model_dir, f"Error_Boxplot_{target}.png")
    plt.savefig(out_path)
    plt.close()

    # Histogram of absolute error distribution
    plt.figure(figsize=(8, 4))
    plt.hist(error[np.isfinite(error)], bins=25, alpha=0.7)
    plt.xlabel("Relative Absolute Error")
    plt.ylabel("Count")
    plt.title(f"Error Histogram - {target or 'output'}")
    plt.tight_layout()
    out_path = os.path.join(save_model_dir, f"Error_Hist_{target}.png")
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------------------
# Graph visualization
# -------------------------------------------------------------------------
def visualize_graph(data, save_model_dir, name="graph_visual"):
    """Visualizes a torch_geometric graph using networkx."""
    vis = to_networkx(data)
    plt.figure(figsize=(8, 8))
    nx.draw(vis, node_size=120, linewidths=1.5, with_labels=False)
    plt.title("Graph Structure")
    plt.tight_layout()
    out_path = os.path.join(save_model_dir, f"{name}.png")
    plt.savefig(out_path)
    plt.close()
