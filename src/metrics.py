"""
Metrics for quantifying stimulus selectivity
"""
import numpy as np


def compute_osi(stimuli, R):
    """
    Orientation Selectivity Index (OSI)

    Arguments:
        stimuli (n_stim, ): angles in rad.
        R (n_stim, ): response/rates

    Returns:
        float
    """
    pref = np.where(stimuli == 0)[0][0]
    oppo = np.where(stimuli == -np.pi)[0][0]
    ortho = np.where(stimuli == -np.pi/2)[0][0]
    ortho_plus = np.where(stimuli == np.pi/2)[0][0]
    osi = (R[pref] + R[oppo] - R[ortho] - R[ortho_plus]) / (R[pref] + R[oppo])
    return osi
