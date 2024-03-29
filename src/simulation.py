"""
Helper functions for network simulations
"""

import numpy as np
from scipy.interpolate import interp1d
from src.metrics import compute_osi


## Basic network ##
def relu(x):
    """
    Pointwise nonlinearity
    """
    out = x.copy()
    out[out < 0] = 0
    return out


# Circular tuning inputs and weights (could use a mixture for orientation tuning)
def von_mises(theta, mu, kappa):
    """
    Pdf of Von Mises distribution, normalized

    Arguments:
        theta (float): stimulus orientation, in radians
        mu (float): preferred orientation, in radians
        kappa (float): selectivity

    Returns:
        float: normalized pdf evaluated at theta
    """
    out = np.exp(kappa * np.cos(theta - mu))
    return out / out.max()


def network(kappa_pre, alpha, kappa_w, alpha_w, n_pre, n_stim, verbose=True):
    """
    Create ingredients for network simulation

    Arguments:
        kappa (float): width of pyr & synaptic tuning
        alpha (float): strengt of orientation tuning
        n_pre (int): number of Pyr neurons
        n_stim (int): number of stimuli

    Returns:
        stimuli (n_stim, ): equally spaced [-pi, pi)
        pre_tuning (n_pre, ): preferred stimulus of pyr units
        pre_rates (n_stim, ): response of pyr unit, centered
        pre_rates_multi: (n_pre, n_stim): response of all units to all stimuli
        weights (n_pre, ): synaptic strength Pyr->PV
        post_rates (n_stim, ): response of PV unit

    """
    # Pyr rates: normalized s.t. unpreferred stimuli elicit 0 response
    pre_tuning = np.linspace(-np.pi, np.pi, n_pre)
    pre_rates = von_mises(0, pre_tuning, kappa=kappa_pre)
    pre_rates += alpha * von_mises(0 - np.pi, pre_tuning, kappa=kappa_pre)
    pre_rates = (pre_rates - pre_rates.min()) / (pre_rates.max() - pre_rates.min())

    # Connectivity, also normalized
    weights = von_mises(0, pre_tuning, kappa_w)  # weights from pre to post (n_pre, )
    weights += alpha_w * von_mises(0 - np.pi, pre_tuning, kappa_w)
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    # Postsynaptic response, for multiple stimuli
    stimuli = np.linspace(-np.pi, np.pi, n_stim, endpoint=False)
    pre_tuning = np.linspace(-np.pi, np.pi, n_pre)
    pre_rates_multi = von_mises(stimuli[None], pre_tuning[:, None], kappa=kappa_pre)
    pre_rates_multi += alpha * von_mises(
        stimuli[None] - np.pi, pre_tuning[:, None], kappa=kappa_pre
    )
    # pre_rates: (n_pre, n_stim)
    post_rates = weights.dot(pre_rates_multi)  # (n_stim)
    # Normalize s.t. max response = 1.
    post_rates /= post_rates.max()

    # Compute selectivity
    if verbose:
        print(f"OSI PV: {compute_osi(stimuli, post_rates):0.2f}")
        print(f"OSI Pyr: {compute_osi(stimuli, pre_rates_multi[int(n_pre/2)]):0.2f}")

    return stimuli, pre_tuning, pre_rates, pre_rates_multi, weights, post_rates


def simulate(I0=0, T=100, epsp_scale=1, dt=1):
    """
    Simulate network dynamics

    Arguments:
        I0 (float): baseline input
        T (int): simulation time
        epsp_scale (float): up/down scale of all weights
        dt (float): integration time const

    Returns:
        stimuli (n_stim, ): stimuli (directions)
        pre_rates (n_stim, ): pyramidal rates
        r_rec (n_stim, ): PV rates
    """
    from src.constants import kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim
    from src.constants import tau

    stimuli, _, pre_rates, pre_rates_multi, weights, _ = network(
        kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim, verbose=False
    )
    weights *= epsp_scale  # Down/upscale all weights
    r = np.zeros(
        (
            len(
                stimuli,
            )
        )
    )  # response to each stimulus
    r_rec = []

    for _ in range(T):
        inputs = weights.dot(pre_rates_multi) + I0  # Sum over stimuli
        drdt = -r + inputs
        r = r + dt / tau * drdt
        r[r < 0] = 0  # rectify
        r_rec.append(r.copy())

    return stimuli, pre_rates, np.array(r_rec)


## Inward-rectification ##
def weight_scale(rate, threshold=2, slope=1, max_scale=2):
    """
    Amplify weights for low PV rates

    Arguments:
        rate (array): PV rate
        threshold (float): midway
        slope (float): steepness of decrease
        max_scale (float): maximum scale for rate -> 0

    Returns:
        float: scale, between [1, max_scale]
    """
    scale = np.tanh(-slope * (rate - threshold))
    scale = 0.5 * (scale + 1)  # [0, 1]
    scale = (max_scale - 1) * scale + 1
    return scale


def simulate_with_conductance(
    block=False, reversal=10, threshold=15, slope=1, max_scale=2, I0=0, T=100, dt=1
):
    """
    Voltage-dependent blockade/hyperpolarization with conductance-based synapses
    """
    from src.constants import (
        alpha_pre,
        kappa_w,
        alpha_w,
        n_pre,
        n_stim,
        kappa_pre_rectification,
        tau,
    )

    stimuli, _, pre_rates, pre_rates_multi, weights, _ = network(
        kappa_pre_rectification,
        alpha_pre,
        kappa_w,
        alpha_w,
        n_pre,
        n_stim,
        verbose=False,
    )
    r = np.ones(
        (
            len(
                stimuli,
            )
        )
    )  # (n_pre, )
    r_rec = []
    scale_rec = []

    for _ in range(T):
        if block:
            scale = weight_scale(r, threshold, slope, max_scale)[None]
        else:
            scale = np.ones(
                (
                    len(
                        stimuli,
                    )
                )
            )
        effective_weights = (
            weights[:, None] * scale * (reversal - r) / reversal
        )  # (n_pre, n_stim)
        inputs = np.sum(effective_weights * pre_rates_multi, axis=0)  # Sum over pyrs
        drdt = -r + inputs + I0  # integrate
        r = r + dt / tau * drdt
        r[r < 0] = 0
        r_rec.append(r.copy())
        scale_rec.append(scale.copy())

    return stimuli, pre_rates, np.array(r_rec), np.squeeze(np.array(scale_rec))


# Plasticity
def bcm_update(pre, post, threshold):
    """
    BCM plasticity rule - normalized
    """
    update = pre * post * (post - threshold)
    return update / update.max()


def simulate_with_plasticity(
    learning_rate=0.02, threshold="mean", mean_scaling=True, T0=100, T=20000, dt=1
):
    """
    Simulate with synaptic plasticity

    Arguments:
        learning_rate (float): step size
        threshold (str or float): "mean", (mean) "squared", or absolute rate
        mean_scaling (bool): rescale weights to preserve mean rate before plasticity
        T0 (int): onset of plasticity (default: 100)
        T (int): total number of time steps (default: 2000)
        dt (float): integration time step

    Returns:

    """
    from src.constants import kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim, tau

    stimuli, _, pre_rates, pre_rates_multi, weights, _ = network(
        kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim, verbose=False
    )
    r = np.zeros(
        (
            len(
                stimuli,
            )
        )
    )  # (n_pre, )
    r_rec = []
    w_rec = []

    for t in range(T):
        inputs = weights.dot(pre_rates_multi)  # Sum over stimuli
        drdt = -r + inputs
        r = r + dt / tau * drdt
        r[r < 0] = 0  # rectify
        r_rec.append(r.copy())

        if t == T0:  # record baseline
            r0 = r.mean()
        if t >= T0:
            if isinstance(threshold, float) or isinstance(threshold, int):
                theta = threshold
            elif threshold == "mean":
                theta = r.mean()  # avg stimuli
            elif threshold == "squared":
                theta = np.mean(r**2)
            dw = bcm_update(pre_rates_multi, r, theta).mean(1)  # mean across stimuli
            # Average update across all stimuli
            weights = weights + learning_rate * dw
            # Scale
            if mean_scaling:
                weights *= r0 / r.mean()
            weights[weights < 0] = 0  # stay excitatory
        w_rec.append(weights.copy())

    return stimuli, pre_rates, np.array(r_rec), np.array(w_rec)


def steady_state(I0=0, epsp_scale=1):
    from src.constants import kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim

    # Closed-form solution - no need to simulate
    _, _, _, pre_rates_multi, weights, _ = network(
        kappa_pre, alpha_pre, kappa_w, alpha_w, n_pre, n_stim, verbose=False
    )
    weights *= epsp_scale

    rate = weights.dot(pre_rates_multi) + I0
    return rate


## Empirical inward rectification ###
# Methods. TODO: move to src
def scaling_fn_voltage(voltage, alpha, I_WT, I_KO, M_WT, M, m):
    """
    Model conuductance as convex combination
    of GluA2 wild type (WT) and knockout (KO)
    """
    from src.constants import V_reversal

    # normalize:
    WT_contrib = I_WT / (V_reversal - voltage) / M_WT
    K0_contrib = (I_KO / (V_reversal - voltage) - m) / (M - m)
    return alpha * K0_contrib + (1 - alpha) * WT_contrib


def scaling_fn(rate, alpha, df, I_WT, I_KO, M_WT, M, m):
    """
    scaling function now as function of rate
    linearly interpolate between datapoints
    """
    voltage = df.index
    from src.constants import min_rate, max_rate

    rates = np.linspace(min_rate, max_rate, len(voltage))
    return interp1d(
        rates,
        scaling_fn_voltage(voltage, alpha, I_WT, I_KO, M_WT, M, m),
        fill_value="extrapolate",
    )(rate)


def simulate_empirical_conductance(
    df, I_WT, I_KO, M_WT, M, m, alpha=0, reversal=15, T=100, dt=1
):
    """
    Voltage-dependent blockade/hyperpolarization with conductance-based synapses
    """
    from src.constants import (
        alpha_pre,
        kappa_w,
        alpha_w,
        n_pre,
        n_stim,
        kappa_pre_rectification,
        tau,
    )

    stimuli, _, pre_rates, pre_rates_multi, weights, _ = network(
        kappa_pre_rectification,
        alpha_pre,
        kappa_w,
        alpha_w,
        n_pre,
        n_stim,
        verbose=False,
    )
    r = np.ones(
        (
            len(
                stimuli,
            )
        )
    )  # (n_pre, )
    r_rec = []
    scale_rec = []

    for _ in range(T):
        scale = scaling_fn(r, alpha, df, I_WT, I_KO, M_WT, M, m)[None]
        effective_weights = (
            weights[:, None] * scale * (reversal - r) / reversal
        )  # (n_pre, n_stim)
        inputs = np.sum(effective_weights * pre_rates_multi, axis=0)  # Sum over pyrs
        drdt = -r + inputs  # integrate
        r = r + dt / tau * drdt
        r[r < 0] = 0
        r_rec.append(r.copy())
        scale_rec.append(scale.copy())

    return stimuli, pre_rates, np.array(r_rec), np.squeeze(np.array(scale_rec))
