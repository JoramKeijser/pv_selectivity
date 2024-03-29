"""
Increase excitability & homeostatic weight scaling
to preserve mean rate
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.optimize import minimize_scalar
from src.simulation import network, steady_state, relu
from src.constants import pink, green
from src.constants import figdir, stylesheet
from src.metrics import compute_osi
from src.constants import kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim

sns.set_context("poster")
sns.set_palette("colorblind")
plt.style.use(stylesheet)

fig, ax = plt.subplots(1, 4, figsize=(6.2, 1.3), sharey=False)

stimuli, pre_tuning, pre_rates, pre_rates_multi, weights, post_rates = network(
    kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim, verbose=False
)


# Baseline mean rate -
base_mean = steady_state(I0=0, epsp_scale=1).mean()


def objective(scale, I0):
    # Squared difference b/w mean rate and excitability + EPSP scale
    return (steady_state(I0=I0, epsp_scale=scale).mean() - base_mean) ** 2


osis = []
scalings = []
input_vals = np.arange(0, 7, 1)
cmap = mpl.colors.LinearSegmentedColormap.from_list("", [green, "gray", pink])
colors = cmap(np.linspace(1, 0, len(input_vals)))
min_colors = cmap(np.linspace(0, 1, len(input_vals)))
x = np.arange(-3, 10, 0.1)
for I0, color, mcolor in zip(input_vals, colors, min_colors):
    ax[0].plot(x, relu(x - I0), color=mcolor, alpha=0.9)
    res = minimize_scalar(objective, [1e-3, 1], args=I0)
    rate = steady_state(I0, res.x)
    ax[0].hlines(5 + 3 * res.x, -2, 1, color=color)
    ax[1].plot(stimuli, rate, color=color)
    ax[2].plot(stimuli, rate / rate.max(), color=color)
    osis.append(compute_osi(stimuli, rate))
    scalings.append(res.x)
    ax[3].scatter(I0, osis[-1], color=color)
    print(f"I0 = {I0}, scale = {res.x:0.2f}, OSI = {osis[-1]:0.2f}")

ax[0].text(-3, 8.5, "W scale", fontsize=6, color="gray")
ax[3].plot(input_vals, osis, color="gray")
ax[3].set_xlabel("Shift")
ax[3].set_ylim([-0.1, 0.8])
ax[0].set_ylabel("Rate (1/s)")
ax[0].set_xlabel("Input current (a.u.)")

for i in [1, 2]:
    ax[i].set_xlabel(r"Stim. direction ($\Delta^\circ$)")
    ax[i].set_xticks([-np.pi, 0, np.pi], [-180, 0, 180])

ax[1].set_ylabel("Rate (1/s)")
ax[2].set_ylabel("Rate (norm)")
ax[3].set_xlabel("Excitability")
ax[3].set_ylabel("Selectivity (OSI)")
sns.despine()
fig.tight_layout()
fig.savefig(figdir + "fig25efgh_excitability.png", dpi=300)
