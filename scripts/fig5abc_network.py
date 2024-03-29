"""
Rates for network icon
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.simulation import network
from src.constants import blue, pink
from src.constants import kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim
from src.constants import figdir, stylesheet

sns.set_context("poster")
sns.set_palette("colorblind")
plt.style.use(stylesheet)


stimuli, pre_tuning, pre_rates, pre_rates_multi, weights, post_rates = network(
    kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim, verbose=False
)

# Tuning of PV and Pyr cells (panel a: inset to circuit diagram)
height, width = 0.4, 1.8  # inch
fig, ax = plt.subplots(figsize=(width, height))
for i in np.arange(0, 64, 8):
    if i == 32:
        alpha_shade = 1  # highlight one curve
    else:
        alpha_shade = 0.33
    plt.plot(stimuli, pre_rates_multi[i], color=blue, alpha=alpha_shade)
sns.despine(left=True, bottom=True)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(figdir + "fig5a_pyr_tuning", dpi=300)

fig, ax = plt.subplots(figsize=(width, height))
plt.plot(stimuli, post_rates, color=pink)
sns.despine(left=True, bottom=True)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(figdir + "fig5a_pv_tuning", dpi=300)

# Seperate figs
stimuli, pre_tuning, pre_rates, pre_rates_multi, weights, post_rates = network(
    kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim
)
# Post rates are indexed by stimuli, but responses are smooth so we only need to
# visualize responses centered around Pyr preferences
post_rates = post_rates[::2]

height, width = 1.3, 3.2  # inch
fig, ax = plt.subplots(1, 2, sharey=False, figsize=(width, height))
yticks = [0, 0.5, 1]

# Presynaptic rates
for i in range(2):
    ax[i].set_ylim([-0.1, 1.1])
    ticks = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[i].set_xticks(ticks, np.array(ticks * 180 / np.pi, dtype=int))
    ax[i].set_yticks(yticks)
sns.despine()

# Connectivity
ax[0].set_ylabel("Weight (norm)")
ax[0].plot(pre_tuning, weights, color="gray")
ax[0].text(-np.pi, 0.8, r"Pyr$\rightarrow$PV", color="gray")
ax[0].set_xlabel(r"Stim. direction ($\Delta^\circ$)")

# Rates
ax[1].set_ylabel("Rate (norm)")
ax[1].plot(pre_tuning, pre_rates, color=blue, label="Pyr")
ax[1].text(-np.pi, 0.65, "Pyr", color=blue)
ax[1].text(-np.pi, 0.8, "PV", color=pink)
ax[1].plot(pre_tuning, post_rates, color=pink, label="PV")
ax[1].set_xlabel(r"Stim. direction ($\Delta^\circ$)")
plt.tight_layout()
plt.savefig(figdir + "fig5bc_network.png", dpi=300)

# Save data: Pyr-PV weights, and rates
df_weights = pd.DataFrame()
df_weights["direction"] = pre_tuning
df_weights["weights"] = weights
df_rates = pd.DataFrame()
df_rates["direction"] = pre_tuning
df_rates["pyr_rates"] = pre_rates
df_rates["pv_rates"] = post_rates
with pd.ExcelWriter("results/figure5.xlsx") as writer:
    df_weights.to_excel(writer, sheet_name="b")
    df_rates.to_excel(writer, sheet_name="c")
