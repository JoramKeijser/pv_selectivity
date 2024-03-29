# Parameter values
figdir = "figures/" # where to save figs
datadir = "data/" # where to load data from
resdir = "results/"
stylesheet = "styles/ingie.mplstyle" # fig formatting

# Colors
blue = "#809fff"
pink = "#ff95ff"
green = "#8aff8a"

# Tuning
kappa_pre = 2
alpha_pre = 0.5
kappa_w = 3
alpha_w = 0

# Network size & stimulus number
n_pre = 64 # presynaptic pyramidal cells
n_stim = 128 # arbitrary number of equally spaced directions
tau = 10  # membrane stim const

# Rectification
kappa_pre_rectification = 3.6
amplitude = 1.6 # increased EPSPs
slope = 0.5
midpoint = 4
# For conductance-based synapses
reversal = 30 # arbitrary rate scaling

# Empirical scaling: reversal potentials
scaling_egfp = 1.6
scaling_glua2 = 1.0
min_scaling = 0.

# Plasticity
Tplasticity = 20000
lr = 0.02
control_th = 8 # Example
increased_th = 10
min_th = 6.5 # For sweep
max_th = 11
# for LTP-dominated rule with 3 zeros
theta_lo = 6 # 2nd zero LTD -> TLP
theta_hi = 20 # 3rd zero: LTP -> LTD

# Excitability
I_example = 4

# Mixture of receptors
V_reversal = 0  # mV
V_cutoff = -10 # any point below reversal to avoid singularity
min_rate = 0
max_rate = 10
