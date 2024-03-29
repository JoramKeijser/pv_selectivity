"""
Simulate mixture of conductances
"""

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import seaborn as sns
import pandas as pd
from src.constants import pink, green, blue, min_rate, max_rate
from src.constants import V_reversal, V_cutoff
from src.constants import figdir, stylesheet
from src.metrics import compute_osi
from src.simulation import scaling_fn, simulate_empirical_conductance

sns.set_context("poster")
sns.set_palette("colorblind")
plt.style.use(stylesheet)

# Load the data
df = pd.read_excel(
    "data/Average IV data for Joram.xlsx",
    skiprows=20,
    index_col=0,
    usecols=[0, 3, 4],
    names=["Membrane potential (mV)", "GluA2 KO", "GluA2 WT"],
)
df = df[df.index < V_cutoff]
voltage = df.index
I_KO = df["GluA2 KO"]
I_WT = df["GluA2 WT"]
# Normalization constants
M = np.max(I_KO / (V_reversal - voltage))
m = np.min(I_KO / (V_reversal - voltage))
M_WT = np.max(I_WT / (V_reversal - voltage))


# Make the figure
fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.3), sharey=False)
# The data
ax[0].text(-60, 0.0, "CI-AMPARs", color="grey")
ax[0].text(-60, 0.13, "CP-AMPARs", color=blue)
ax[0].plot(voltage, I_KO, color="grey")
ax[0].plot(voltage, I_WT, color=blue)
ax[0].set_xlabel("Membrane potential (mV)")
ax[0].set_ylabel("Current (norm.)")
ax[0].set_ylim([-0.05, 1.05])
ax[0].set_xticks([-60, -40, -20])
ax[0].set_yticks([0, 0.5, 1])


# Estimated conductance
ax[1].text(-60, 0.0, "CI-AMPARs", color="grey")
ax[1].text(-60, 0.13, "CP-AMPARs", color=blue)
ax[1].plot(voltage, (I_KO / (V_reversal - voltage) - m) / (M - m), color="grey")
ax[1].plot(voltage, I_WT / (V_reversal - voltage) / M_WT, color=blue)
ax[1].set_xlabel("Membrane potential (mV)")
ax[1].set_ylabel("Conductance (norm.)")
ax[1].set_ylim([-0.05, 1.05])
ax[1].set_xticks([-60, -40, -20])
ax[1].set_yticks([0, 0.5, 1])

# Scaling functions
labels = ["0CP + 1CI (GluA2)", "0.8CP + 0.2CI (eGFP)"]
colors = [green, pink]
ax[2].text(0.33, 0.13, labels[0], color=colors[0])
ax[2].text(0.33, 0, labels[1], color=colors[1])
plot_rates = np.arange(min_rate, max_rate)

ax[2].plot(
    plot_rates,
    scaling_fn(plot_rates, 0, df, I_WT, I_KO, M_WT, M, m),
    label=labels[0],
    color=colors[0],
)
ax[2].plot(
    plot_rates,
    scaling_fn(plot_rates, 0.8, df, I_WT, I_KO, M_WT, M, m),
    label=labels[1],
    color=colors[1],
)
ax[2].set_xlabel("Rate (1/s)")
ax[2].set_ylabel("Weight scaling")
ax[2].set_ylim([-0.05, 1.05])
ax[2].set_xlim([0, 10])

fig.tight_layout()
plt.savefig(figdir + "fig26abc_rectification.png", dpi=300)

# Simulate
fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.3), sharey=False)

for alpha, color in zip([0, 0.8], [green, pink]):
    stimuli, pre_rates, rates, _ = simulate_empirical_conductance(
        df, I_WT, I_KO, M_WT, M, m, alpha
    )
    ax[0].plot(stimuli * 180 / np.pi, rates[-1], color=color)
    ax[1].plot(stimuli * 180 / np.pi, rates[-1] / rates[-1].max(), color=color)

ax[0].set_ylim([0, 7])
ax[1].set_ylim([0, 1.1])


alphas = np.arange(0, 1.01, 0.01)
osis = np.zeros((len(alphas),))
for i, alpha in enumerate(alphas):
    stimuli, pre_rates, rates, _ = simulate_empirical_conductance(
        df, I_WT, I_KO, M_WT, M, m, alpha
    )
    osis[i] = compute_osi(stimuli, rates[-1])
    if alpha == 0:
        ax[2].scatter(alpha, osis[i], color=green)
        print(f"OSI GluA2: {osis[i]:0.2f}")
    elif alpha == 0.8:
        ax[2].scatter(alpha, osis[i], color=pink)
        print(f"OSI eGFP: {osis[i]:0.2f}")
ax[2].plot(alphas, osis, color="grey")


ax[2].set_xlabel(r"CP-AMPAR coefficient $\alpha$")
ax[-1].set_ylim([0.3, 0.6])
ax[-1].set_yticks([0.3, 0.6])
ax[0].set_xlabel("Rate (1/s)")

ticks = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
for i in [0, 1]:
    ax[i].set_xlabel(r"Stim. direction ($\Delta^\circ$)")
    ax[i].set_xticks([-180, -90, 0, 90, 180])
ax[2].set_ylabel("Selectivity (OSI)")
ax[0].set_ylabel("Rate (1/s)")
ax[1].set_ylabel("Rate (norm.)")

fig.tight_layout()
plt.savefig(figdir + "fig26def_rectification.png", dpi=300)
