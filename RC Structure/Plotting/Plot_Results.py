import csv
from pathlib import Path

import matplotlib


def _load_pyplot(prefer_interactive=False):
    backends = []

    if prefer_interactive:
        backends.extend(["QtAgg", "TkAgg"])

    backends.append("Agg")

    last_error = None
    for backend in backends:
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt

            test_fig = plt.figure()
            plt.close(test_fig)
            return plt, backend
        except Exception as exc:
            last_error = exc

    raise RuntimeError("Could not initialize a Matplotlib backend.") from last_error


def plot_pushover_curve(pushover_results, show=False):
    plt, backend = _load_pyplot(prefer_interactive=show)
    roof_disp = pushover_results["roof_disp"]
    base_shear = pushover_results["base_shear"]

    fig, ax = plt.subplots()
    ax.plot(roof_disp, base_shear, "-o", markersize=2)
    ax.set_xlabel("Roof displacement X (in)")
    ax.set_ylabel("Base shear X (kip)")
    ax.set_title("Pushover Curve")
    ax.grid(True)
    fig.tight_layout()

    if show:
        if "agg" in backend.lower():
            print("Interactive plotting backend unavailable; plot was saved instead.")
        else:
            try:
                plt.show()
            except Exception as exc:
                print(f"Interactive plot display failed: {exc}")

    return fig, ax


def plot_roof_drift_curve(pushover_results, show=False):
    plt, backend = _load_pyplot(prefer_interactive=show)
    roof_drift_percent = [100.0 * drift for drift in pushover_results["roof_drift"]]
    base_shear = pushover_results["base_shear"]

    fig, ax = plt.subplots()
    ax.plot(roof_drift_percent, base_shear, "-o", markersize=2)
    ax.set_xlabel("Roof drift ratio (%)")
    ax.set_ylabel("Base shear X (kip)")
    ax.set_title("Pushover Curve by Roof Drift")
    ax.grid(True)
    fig.tight_layout()

    if show:
        if "agg" in backend.lower():
            print("Interactive plotting backend unavailable; plot was saved instead.")
        else:
            try:
                plt.show()
            except Exception as exc:
                print(f"Interactive plot display failed: {exc}")

    return fig, ax


def save_pushover_curve(pushover_results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "roof_disp_x_in", "base_shear_x_kip"])

        for step, (disp, shear) in enumerate(
            zip(pushover_results["roof_disp"], pushover_results["base_shear"])
        ):
            writer.writerow([step, disp, shear])

    return output_path
