"""
Model effect of increased excitability on selectivity
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from src.simulation import relu, simulate
from src.constants import pink, green, I_example
from src.metrics import compute_osi

plt.style.use("styles/ingie.mplstyle")
figdir = "figures/"
datadir = "data/"

# Load data. 'Aver.' and 'Aver.1 columsn are control and GluA2 mice, respectively
df = pd.read_excel(
    datadir + "20230411 Ext-Data 16.xlsx",
    skiprows=5,
    usecols=[18, 19, 20, 21, 22],
    nrows=16,
    index_col=0,
)


def control_fi(x):
    # Fit the increase in excitability. Use interpolation to compare the two datasets
    # - actually just imputing 0s for even smaller (negative) currents.
    return interp1d(df.index, df["Aver"], fill_value="extrapolate")(x)


def glua2_fi(x):
    return interp1d(df.index, df["Aver.1"], fill_value="extrapolate")(x)


def objective(shift):
    # Squared difference between PV and shifted GluA2 curve
    x = np.arange(-100, 710)
    control = control_fi(x)
    glua2 = glua2_fi(x - shift)
    return np.mean((control - glua2) ** 2)


# Fit the shift
boundaries = [-200, 200]
best_shift = minimize_scalar(objective, boundaries).x
print(f"Fitted shift: {best_shift:0.2f}")

# plot
fig, ax = plt.subplots(1, 4, figsize=(6.2, 1.3), sharey=False)

# Empirical
ax[0].plot(df.index, df["Aver"], color=pink, alpha=0.9, label="PV")
ax[0].plot(df.index, df["Aver.1"], color=green, alpha=0.9, label="GluA2")
start_ix = 2  # Don't show first fitted zeros
ax[0].plot(
    df.index[start_ix:] - best_shift,
    df["Aver"].iloc[start_ix:],
    color="gray",
    linestyle=":",
    alpha=0.9,
)


# Indicate shift with arrow
x0 = 10
ax[0].annotate(
    "",
    xy=(df.index[8] - x0, df["Aver.1"].iloc[8]),
    xycoords="data",
    xytext=(df.index[11], df["Aver.1"].iloc[8]),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=0.75, connectionstyle="arc3", color="gray"),
)
ax[0].text(df.index[9] + 25, df["Aver.1"].iloc[8] + 10, "$E$", color="gray")


# Show the model
x = np.arange(-1, 1, 0.01)
for i, (I0, label, color, linestyle) in enumerate(
    zip([0, 0.4], ["Control", "GluA2"], [pink, green], ["-", "-"])
):
    ax[1].plot(x, relu(x + I0), color=color, linestyle=linestyle)


# Axes
ax[0].set_xlabel("Current Injected (pA)")
ax[0].set_ylabel("Spike Frequency (Hz)")
ax[0].set_yticks(np.arange(0, 250, 50))
ax[0].set_xticks(np.arange(0, 800, 200))
# Model
ax[1].set_xlabel("Current Injected (norm)")
ax[1].set_ylabel("Rate (norm)")
ax[1].set_yticks([0, 1])
ax[1].set_xticks(np.arange(-1, 2.0, 1))

# Simulate the model with and without increased excitability
for i, (I0, color, label) in enumerate(
    zip([0, I_example], [pink, green], ["Control", "GluA2"])
):
    stimuli, pre_rates, rates = simulate(I0)

    # ax[2].plot(stimuli, rates[-1] / rates[-1].max(), color=color)
    ax[2].plot(stimuli, rates[-1], color=color)

ticks = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax[2].set_xticks(ticks, np.array(ticks * 180 / np.pi, dtype=int))
ax[2].set_xlabel(r"Stim. direction ($\Delta^\circ$)")
ax[2].set_yticks([0, 5, 10, 15])
ax[2].set_ylabel("Rate (norm)")

# Simulate for a range of excitability/input increases
inputs = np.arange(0, 11, 1)
epsp_scales = [1, 0.62]  # baseline and empirical
osis = np.zeros((len(inputs), 2))  # save orientation selectivit indices
for j, scale in enumerate(epsp_scales):
    for i, I0 in enumerate(inputs):
        stimuli, pre_rates, rates = simulate(I0, epsp_scale=scale)
        osis[i, j] = compute_osi(stimuli, rates[-1])  # Record in steady state
        if scale == 1 and I0 in [I_example, 0]:
            print(f"OSI I0={I0:0.2f}: {osis[i,j]:0.2f}")


# Make the figure
for j, color in zip([0, 1], ["gray", "black"]):
    plt.plot(inputs, osis[:, j], color=color)
plt.scatter(0, osis[0, 0], color=pink)
assert I_example == 4  # assume it's 4, for indexing osis
plt.scatter(I_example, osis[4, 0], color=green)
plt.text(2, 0.4, "EPSP scale = 1", color="gray")
plt.text(2, 0.08, "EPSP scale = .62", color="black")
plt.xlabel(r"Excitability $E$")
plt.ylabel("Selectivity (OSI)")
plt.ylim([0, 0.51])
plt.xticks([0, 5, 10])

fig.tight_layout()
plt.savefig(figdir + "fig25abcd_excitability.png", dpi=300)
