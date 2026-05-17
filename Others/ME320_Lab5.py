import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ================= Load data =================
df = pd.read_csv("/Users/andrewtao/Desktop/ME320_Independent_lab/Data.csv")

elapsed_sec = df["Elapsed_sec"].to_numpy()
ch0 = df["CH0_Temp(C)"].to_numpy()
ch2 = df["CH2_Temp(C)"].to_numpy()
ch1 = df["CH1_Temp(C)"].to_numpy()
ch3 = df["CH3_Temp(C)"].to_numpy()

# ================= Trim startup =================
start = 1170


t = elapsed_sec[start:]
brass = ch0[start:]
aluminum = ch2[start:]
brass_surface = ch1[start:]
aluminum_surface = ch3[start:]



# Make time start from zero after trimming
t = t

# ================= Relaxation time function =================
def compute_relaxation_time(time, temp, window_length=101, polyorder=3):
    """
    Inputs:
        time: elapsed time array, seconds
        temp: temperature array
    Outputs:
        tau: relaxation time array, same size as temp
        tau_mean: mean relaxation time
    """

    # Estimate final equilibrium temperature
    T_inf = np.mean(temp[-10:])

    # Smooth temperature to reduce derivative noise
    if window_length >= len(temp):
        window_length = len(temp) - 1

    if window_length % 2 == 0:
        window_length += 1

    temp_smooth = savgol_filter(temp, window_length, polyorder)

    # Numerical derivative dT/dt
    dTdt = np.gradient(temp_smooth, time)

    # Relaxation time: tau = -(T - T_inf)/(dT/dt)
    tau = -(temp_smooth - T_inf) / dTdt

    # Remove bad values caused by near-zero derivative
    tau[np.abs(dTdt) < 1e-6] = np.nan
    tau[tau < 0] = np.nan

    back_trim = 1500
    cut_time = t[len(t) - back_trim]
    print(f"Cutoff time: {cut_time:.2f} s")

    tau_mean = np.nanmean(tau[:-back_trim])

    return tau, tau_mean


# Apply to both channels
tau_brass, tau_brass_mean = compute_relaxation_time(t, brass)
tau_aluminum, tau_aluminum_mean = compute_relaxation_time(t, aluminum)

print(f"Mean relaxation time for brass: {tau_brass_mean:.2f} s")
print(f"Mean relaxation time for aluminum: {tau_aluminum_mean:.2f} s")

# ================= Plot style =================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
# ================= Plot style: clean academic =================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 1.2,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

# ================= Custom figure size =================
fig_width = 6.0
height_width_ratio = 0.7
fig_height = fig_width * height_width_ratio
# ================= Prepare log-temperature vs time data =================
mask = (t > 0) & (brass > 0) & (aluminum > 0)

t_plot = t[mask]
brass_plot = brass[mask]
aluminum_plot = aluminum[mask]

log_brass = np.log(brass_plot)
log_al = np.log(aluminum_plot)

# Linear fit: log(T) = k*t + b
k_b, b_b = np.polyfit(t_plot, log_brass, 1)
k_a, b_a = np.polyfit(t_plot, log_al, 1)

print(f"Brass linear fit slope: {k_b:.6e} 1/s")
print(f"Aluminum linear fit slope: {k_a:.6e} 1/s")

# Fit lines
brass_fit_log = k_b * t_plot + b_b
al_fit_log = k_a * t_plot + b_a

# ================= Plot =================
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Log-temperature data
ax.scatter(t_plot, log_brass, s=14, alpha=0.6, color="#1f77b4", label="Brass log-temperature data")
ax.scatter(t_plot, log_al, s=14, alpha=0.6, color="#ff7f0e", label="Aluminum log-temperature data")

# Linear fitting lines
ax.plot(t_plot, brass_fit_log, color="black", linewidth=2.2, linestyle="--",
        label="Brass linear fit")
ax.plot(t_plot, al_fit_log, color="black", linewidth=2.2, linestyle="-.",
        label="Aluminum linear fit")

# Labels
ax.set_xlabel(r"Time, $t$ (s)")
ax.set_ylabel(r"$\ln(T)$")
ax.set_title("Log-Temperature Decay with Linear Fits")

# Grid
ax.grid(True, which="major", linestyle="-", linewidth=1, alpha=1)
ax.grid(True, which="minor", linestyle=":", linewidth=1, alpha=1)

ax.tick_params(direction="in", which="both", top=True, right=True)

# Equation boxes
ax.text(
    0.31, 0.85,
    fr"Brass: $\ln(T) = {k_b:.3e}t + {b_b:.2f}$",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
)

ax.text(
    0.31, 0.75,
    fr"Aluminum: $\ln(T) = {k_a:.3e}t + {b_a:.2f}$",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
)

ax.legend(frameon=False, loc="lower left")

fig.tight_layout()
plt.show()

