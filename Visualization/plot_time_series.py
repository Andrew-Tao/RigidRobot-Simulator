import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

file_path = "/Users/andrewtao/Desktop/Testing_local_temp_log_20260403_100150.csv"
df = pd.read_csv(file_path)
print(df.columns)

columns_as_arrays = {}
for col in df.columns:
    columns_as_arrays[col] = df[col].to_numpy()

# ================================
# YOUR DATA
# ================================
time = columns_as_arrays['Elapsed_sec'] / 60
var1 = columns_as_arrays['CH0_Temp(C)']
var2 = columns_as_arrays['CH1_Temp(C)']
var3 = columns_as_arrays['CH2_Temp(C)']
diff = var1 - var2

Q = diff * 47.5


# ================================
# CUSTOM COLORS (EDIT HERE)
# ================================
color_var1 = "#1f77b4"   # blue
color_var2 = "#d62728"   # red
color_var3 = "#2ca02c"   # green

# ================================
# GLOBAL STYLE (Nature-like)
# ================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1,
    "ytick.major.width": 1,
})

# ================================
# FIGURE
# ================================
fig, ax = plt.subplots(figsize=(8.5, 3.5), dpi=300)

# ================================
# PLOT
# ================================
ax.plot(time, diff * 47.5, color=color_var1, linewidth=1.8, label="$\dot{Q}$")
#ax.plot(time, var2, color=color_var2, linewidth=1.6, label="Outer Temperature")
#ax.plot(time, var3, color=color_var3, linewidth=1.6, label="PCM Temperature")

# ================================
# LABELS
# ================================
ax.set_xlabel("Time (min)")
ax.set_ylabel("Heat Transfer Rate  $\dot{Q}$ (W)")
ax.set_title("Heat-transfer rate of water - Time", pad=8)

# ================================
# FULL BOUNDARY BOX (4 SIDES)
# ================================
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

# ================================
# TICKS
# ================================
ax.tick_params(axis='both', which='major', length=5, width=1)


# Optional: ticks on all sides (very journal-style)
ax.tick_params(top=False, right=False)

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.set_xlim(left=0)
# ================================
# NO GRID (still Nature clean)
# ================================
ax.grid(False)

# ================================
# LEGEND
# ================================
ax.legend(frameon=False, loc="upper left", fontsize=10)

# ================================
# LAYOUT
# ================================
plt.tight_layout()

# Save high-quality figure
plt.savefig("nature_style_boxed.png", dpi=300, bbox_inches='tight')

plt.show()

def integrate_Q(time, Q):

    return np.trapezoid(Q, time)
Q_total = integrate_Q(time[2:]*60, Q[2:])
print(Q_total)