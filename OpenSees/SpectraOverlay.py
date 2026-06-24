import numpy as np
import matplotlib.pyplot as plt
import re

# -----------------------
# USER INPUTS
# -----------------------
file_x = r"C:\Users\andro\Downloads\RSN6_IMPVALL.I_I-ELC180.AT2"
file_y = r"C:\Users\andro\Downloads\RSN6_IMPVALL.I_I-ELC270.AT2"

record_scale = 1.4277
g = 32.2  # ft/s^2
scale = record_scale * g

T_values = np.linspace(0.05, 10, 200)  # periods in seconds


# -----------------------
# READ AT2 FILE
# -----------------------
def read_at2(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    dt = None
    data_start = 0

    for i, line in enumerate(lines):
        if "DT" in line.upper():
            match = re.search(r"DT\s*=\s*([0-9.Ee+-]+)", line.upper())
            if match:
                dt = float(match.group(1))
            data_start = i + 1
            break

    values = []
    for line in lines[data_start:]:
        for item in line.split():
            try:
                values.append(float(item))
            except ValueError:
                pass

    if dt is None:
        raise ValueError(f"Could not find DT in {filename}")

    ag = np.array(values) * scale  # convert from g to ft/s^2
    return ag, dt


# -----------------------
# NEWMARK RESPONSE SPECTRUM
# -----------------------
def response_spectrum(ag, dt, T_values, zeta):
    beta = 1 / 4
    gamma = 1 / 2

    Sd = []
    Sa = []

    for T in T_values:
        omega = 2 * np.pi / T
        m = 1.0
        k = omega**2 * m
        c = 2 * zeta * omega * m

        u = 0.0
        v = 0.0
        a = -ag[0]

        u_hist = []

        a0 = 1 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1 / (beta * dt)
        a3 = 1 / (2 * beta) - 1
        a4 = gamma / beta - 1
        a5 = dt * (gamma / (2 * beta) - 1)

        k_eff = k + a0 * m + a1 * c

        for i in range(1, len(ag)):
            p_eff = (
                -m * ag[i]
                + m * (a0 * u + a2 * v + a3 * a)
                + c * (a1 * u + a4 * v + a5 * a)
            )

            u_new = p_eff / k_eff
            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1 - gamma) * a + gamma * a_new)

            u, v, a = u_new, v_new, a_new
            u_hist.append(u)

        sd = max(abs(np.array(u_hist)))
        sa = omega**2 * sd

        Sd.append(sd)
        Sa.append(sa)

    return np.array(Sd), np.array(Sa)

# -----------------------
# RUN OVERLAY SPECTRA
# -----------------------
ag_x, dt_x = read_at2(file_x)
ag_y, dt_y = read_at2(file_y)

if abs(dt_x - dt_y) > 1e-9:
    raise ValueError("X and Y time steps do not match.")

damping_cases = {
    "5% Damping": 0.05,
    "20% Damping": 0.20
}

T_fixed = 0.4      # fixed-base period
T_isolated = 2.68   # isolated period

results = {}

for label, zeta in damping_cases.items():
    Sd_x, Sa_x = response_spectrum(ag_x, dt_x, T_values, zeta)
    Sd_y, Sa_y = response_spectrum(ag_y, dt_y, T_values, zeta)

    Sa_srss = np.sqrt(Sa_x**2 + Sa_y**2)
    Sd_srss = np.sqrt(Sd_x**2 + Sd_y**2)

    results[label] = {
        "Sa": Sa_srss,
        "Sd": Sd_srss
    }

# -----------------------
# SAVE CSV
# -----------------------
out = np.column_stack([
    T_values,
    results["5% Damping"]["Sa"],
    results["20% Damping"]["Sa"],
    results["5% Damping"]["Sd"] * 12,
    results["20% Damping"]["Sd"] * 12
])

np.savetxt(
    "ElCentro_Response_Spectra_5pct_20pct_SRSS.csv",
    out,
    delimiter=",",
    header="T_sec,Sa_5pct_ft_s2,Sa_20pct_ft_s2,Sd_5pct_in,Sd_20pct_in",
    comments=""
)

# -----------------------
# PLOT Sa OVERLAY
# -----------------------
plt.figure()
plt.plot(T_values, results["5% Damping"]["Sa"], label="5% Damping")
plt.plot(T_values, results["20% Damping"]["Sa"], label="20% Damping")

plt.axvline(T_fixed, linestyle="--", color="blue", label=f"Fixed Base T = {T_fixed:.2f}s")
plt.axvline(T_isolated, linestyle="--", color="red", label=f"Isolated T = {T_isolated:.2f}s")

plt.xlabel("Period, T (sec)")
plt.ylabel("Spectral Acceleration, Sa (ft/s²)")
plt.title("El Centro Acceleration Response Spectrum - 5% vs 20% Damping")
plt.grid(True)
plt.legend()
plt.savefig("ElCentro_Sa_5pct_20pct_Overlay.png", dpi=300)

# -----------------------
# PLOT Sd OVERLAY
# -----------------------
plt.figure()
plt.plot(T_values, results["5% Damping"]["Sd"] * 12, label="5% Damping")
plt.plot(T_values, results["20% Damping"]["Sd"] * 12, label="20% Damping")

plt.axvline(T_fixed, linestyle="--", color="blue", label=f"Fixed Base T = {T_fixed:.2f}s")
plt.axvline(T_isolated, linestyle="--", color="red", label=f"Isolated T = {T_isolated:.2f}s")

plt.xlabel("Period, T (sec)")
plt.ylabel("Spectral Displacement, Sd (in)")
plt.title("El Centro Displacement Response Spectrum - 5% vs 20% Damping")
plt.grid(True)
plt.legend()
plt.savefig("ElCentro_Sd_5pct_20pct_Overlay.png", dpi=300)

plt.show()