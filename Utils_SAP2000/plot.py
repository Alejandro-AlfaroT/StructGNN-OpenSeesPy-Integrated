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
    """Computes a simple moving average for smoothing training curves."""
    record = np.array(record)
    average_record = np.zeros(len(record))
    for index in range(len(record)):
        start = max(0, index - half_length)
        end = min(len(record), index + half_length)
        average_record[index] = record[start:end].mean()
    return average_record


# -------------------------------------------------------------------------
# Learning and loss curves
# -------------------------------------------------------------------------
def plot_learningCurve(accuracy_record, save_model_dir, title=None, target="node_edge"):
    """
    Plots training and validation accuracy curves.

    accuracy_record: np.array with shape (3, epochs) for [train, valid, test].
    """
    train_acc, valid_acc, test_acc = accuracy_record
    epochs = list(range(1, len(train_acc) + 1))
    plt.figure(figsize=(8, 6))

    title = (title or "") + f"\nValidation accuracy: {valid_acc.max() * 100:.1f}%"

    plt.plot(epochs, moving_average(train_acc), label="Train")
    plt.plot(epochs, moving_average(valid_acc), label="Validation")
    if test_acc[-1] != 0:
        plt.plot(epochs, moving_average(test_acc), label="Test")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("Combined Accuracy (Node + Edge)")
    plt.ylim([-0.05, 1.05])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_model_dir + f"LearningCurve_{target}.png")
    plt.close()


def plot_lossCurve(loss_record, save_model_dir, title=None, target="node_edge"):
    """
    Plots training and validation loss curves.
    """
    train_loss, valid_loss, test_loss = loss_record
    epochs = list(range(1, len(train_loss) + 1))
    plt.figure(figsize=(8, 6))

    plt.plot(epochs, moving_average(train_loss), label="Train")
    plt.plot(epochs, moving_average(valid_loss), label="Validation")
    if test_loss[-1] != 0:
        plt.plot(epochs, moving_average(test_loss), label="Test")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title or "Loss Curve")
    plt.tight_layout()
    plt.savefig(save_model_dir + f"LossCurve_{target}.png")
    plt.close()


# -------------------------------------------------------------------------
# Accuracy distribution plots
# -------------------------------------------------------------------------
def plot_accuracy_distribution(y_pred, y_real, save_model_dir, target=None, max_value=None, threshold=0.0):
    """
    Plots error and distribution histograms for node or edge predictions.
    Works for both node (2D) and edge (6D) outputs.

    Parameters
    ----------
    y_pred : np.ndarray or tensor
        Model predictions (flattened or per-feature).
    y_real : np.ndarray or tensor
        Ground truth values, same shape as y_pred.
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
    plt.savefig(save_model_dir + f"Error_Boxplot_{target}.png")
    plt.close()

    # Histogram of absolute error distribution
    plt.figure(figsize=(8, 4))
    plt.hist(error[np.isfinite(error)], bins=25, color="steelblue", alpha=0.7)
    plt.xlabel("Relative Absolute Error")
    plt.ylabel("Count")
    plt.title(f"Error Histogram - {target or 'output'}")
    plt.tight_layout()
    plt.savefig(save_model_dir + f"Error_Hist_{target}.png")
    plt.close()


# -------------------------------------------------------------------------
# Graph visualization
# -------------------------------------------------------------------------
def visualize_graph(data, save_model_dir, name="graph_visual"):
    """
    Visualizes a torch_geometric graph using networkx.
    """
    vis = to_networkx(data)
    plt.figure(figsize=(8, 8))
    nx.draw(vis, node_size=120, linewidths=1.5, with_labels=False, node_color="lightblue")
    plt.title("Graph Structure")
    plt.tight_layout()
    plt.savefig(save_model_dir + f"{name}.png")
    plt.close()
