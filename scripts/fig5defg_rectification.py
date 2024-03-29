"""
Inward rectification by CP-AMPARs
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr  # colormap
import seaborn as sns
import pandas as pd
from src.constants import pink, green
from src.metrics import compute_osi
from src.simulation import simulate_with_conductance, weight_scale
from src.constants import amplitude, slope, midpoint, reversal
from src.constants import scaling_egfp, scaling_glua2
from src.constants import figdir, stylesheet
from src.utils import write_excel

sns.set_context("poster")
sns.set_palette("colorblind")
plt.style.use(stylesheet)

# collecting/saving data:
save_path = "results/Source Data Fig. 5.xlsx"
data_frames = {}
for panel in ['d', 'e', 'f', 'g']:
    data_frames[panel] = pd.DataFrame()

# Amplification function
rate = np.arange(0, 10, 0.1)  # PV rate/voltage
fig, ax = plt.subplots(1, 4, figsize=(6.2, 1.3), sharey=False)
ax[0].plot(rate, weight_scale(rate, midpoint, slope, scaling_egfp), color=pink)
ax[0].hlines(1, rate.min(), rate.max(), color=green, linestyle="-.")
ax[0].set_ylim([-0.1, amplitude * 1.1])
ax[0].set_xlabel("Rate (1/s)")
ax[0].set_ylabel("Weight scale")
ax[0].set_xticks([0, 5, 10])
ax[0].set_yticks([0, 1, 2])
# Annotate amp and midpoint
ax[0].annotate(
    "",
    xy=(1.75, amplitude),
    xycoords="data",
    xytext=(3.75, amplitude),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=0.5, connectionstyle="arc3", color="gray"),
)
ax[0].text(midpoint - 0.4, amplitude - 0.1, "A", color="gray", rotation=0)
ax[0].annotate(
    "",
    xy=(midpoint, 0),
    xycoords="data",
    xytext=(midpoint, 0.5),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=0.5, connectionstyle="arc3", color="gray"),
)
ax[0].text(midpoint - 0.4, 0.5, "M", color="gray", rotation=0)
# dataframes for writing data
data_frames['d']["rate"] = rate
data_frames['d']['scale'] = weight_scale(rate, midpoint, slope, scaling_egfp)


# Compare tuning
for i, (block, color, label, scaling) in enumerate(
    zip(
        [True, False],
        [pink, green],
        ["Control", "No Amp."],
        [scaling_egfp, scaling_glua2],
    )
):

    stimuli, pre_rates, rates, scale = simulate_with_conductance(
        block,
        reversal=reversal,
        threshold=midpoint,
        slope=slope,
        max_scale=scaling,
    )

    # Quantify:
    osi = compute_osi(stimuli, rates[-1])
    if block:
        print(f"eGFP: {osi:0.2f}; rate: {rates[-1].mean():0.2f}")
    else:
        print(f"GluA2 OSI: {osi:0.2f}; rate: {rates[-1].mean():0.2f}")

    ax[1].plot(stimuli, rates[-1], color=color)
    ax[2].plot(stimuli, rates[-1] / rates[-1].max(), color=color)
    # for saving data
    data_frames['e']['stimuli'] = stimuli
    data_frames['e'][f'rates {label}'] = rates[-1]
    data_frames['f']['stimuli'] = stimuli
    data_frames['f'][f'rates (norm) {label}'] = rates[-1] / rates[-1].max()

for i in [1, 2]:
    ticks = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[i].set_xticks(ticks, np.array(ticks * 180 / np.pi, dtype=int))
    ax[i].set_xlabel(r"Stim. direction ($\Delta^\circ$)")

ax[1].set_ylim([0, 11])
ax[2].set_yticks([0, 0.5, 1])
ax[2].set_ylim([0, 1.05])
ax[1].set_ylabel("Rate (1/s)")
ax[2].set_ylabel("Rate (norm)")


# Robustness
scales = np.linspace(1, 4, 20, endpoint=True)
thresholds = np.linspace(1, 5, 20, endpoint=True)
osis = np.zeros((len(scales), len(thresholds)))
for i, scale in enumerate(scales):
    for j, threshold in enumerate(thresholds):
        stimuli, pre_rates, rates, _ = simulate_with_conductance(
            True,
            reversal=reversal,
            threshold=threshold,
            slope=slope,
            max_scale=scale,
        )
        osis[i, j] = compute_osi(stimuli, rates[-1])

# Baseline
stimuli, pre_rates, rates, _ = simulate_with_conductance(
    True, threshold=midpoint, reversal=reversal, slope=slope, max_scale=scale
)
base_osi = compute_osi(stimuli, rates[-1])

# No amp
stimuli, pre_rates, rates, _ = simulate_with_conductance(
    False, threshold=midpoint, reversal=reversal, slope=slope, max_scale=amplitude
)
no_osi = compute_osi(stimuli, rates[-1])

# Fig: OSI without rectification - osi with
pl = plt.pcolor(no_osi - osis, cmap="cmr.guppy_r")
# save data
data_frames['g'] = pd.DataFrame(no_osi - osis, index = thresholds, columns = scales)
plt.scatter(
    np.argmin(np.abs(scales - amplitude)),
    np.argmin(np.abs(thresholds - midpoint)),
    color=pink,
    s=5,
)
plt.ylabel(r"Midpoint M (1/s)")
plt.xlabel(r"Amplitude A")
plt.xticks([1, len(scales)], [1, scales[-1]])
plt.yticks([0, len(thresholds)], [0, thresholds[-1]])
cbar = plt.colorbar(pl)
if np.max(no_osi - osis) > 0.25:
    cbar.set_ticks([0, 0.1, 0.2, 0.3])
cbar.set_label(r"$\Delta$ OSI", fontsize=7, rotation=90)

plt.tight_layout()
plt.savefig(figdir + "fig5defg_rectification.png", dpi=300)

# Save data
write_excel(save_path, data_frames)



