"""
Rates for network icon
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.simulation import network
from src.constants import blue, pink
from src.constants import kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim
from src.constants import figdir, stylesheet
from src.utils import write_excel

# Added code to plot with Arial font in Windows
import matplotlib
from matplotlib import rc
matplotlib.rcParams['pdf.fonttype']=42
rc('font',**{'family':'serif','serif':['Arial']})
#matplotlib.rcParams['font.family']='san-serif'
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

sns.set_context("poster")
sns.set_palette("colorblind")
plt.style.use(stylesheet)

# collecting/saving data:
save_path = "results/Source Data Fig. 5.xlsx"
data_frames = {}
for panel in ['b', 'c']: # Panel a is the circuit diagram
    data_frames[panel] = pd.DataFrame()

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
# Export to PDF too
plt.savefig(figdir + "fig5a_pyr_tuning.pdf", dpi=300)

fig, ax = plt.subplots(figsize=(width, height))
plt.plot(stimuli, post_rates, color=pink)
sns.despine(left=True, bottom=True)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(figdir + "fig5a_pv_tuning", dpi=300)
# Export to PDF too
plt.savefig(figdir + "fig5a_pv_tuning.pdf", dpi=300)

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
ax[0].text(-np.pi, 0.8, r"Pyr→PV", color="gray")
ax[0].set_xlabel(r"Stim. direction (Δ°)")
ax[0].tick_params(axis='both', which='major', labelsize=5)

# Rates
ax[1].set_ylabel("Rate (norm)")
ax[1].plot(pre_tuning, pre_rates, color=blue, label="Pyr")
ax[1].text(-np.pi, 0.65, "Pyr", color=blue)
ax[1].text(-np.pi, 0.8, "PV", color=pink)
ax[1].plot(pre_tuning, post_rates, color=pink, label="PV")
ax[1].set_xlabel(r"Stim. direction (Δ°)")
ax[1].tick_params(axis='both', which='major', labelsize=5)
plt.tight_layout()
plt.savefig(figdir + "fig5bc_network.png", dpi=300)
# Export to PDF too
plt.savefig(figdir + "fig5bc_network.pdf", dpi=300)


# Save data: Pyr-PV weights, and rates
data_frames['b']["direction"] = pre_tuning
data_frames['b']["weights"] = weights
data_frames['c']["direction"] = pre_tuning
data_frames['c']["pyr_rates"] = pre_rates
data_frames['c']["pv_rates"] = post_rates

write_excel(save_path, data_frames)

