import numpy as np
from typing import Dict
import ot  # Python Optimal Transport


def analyze_ts(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 10,
    metric: str = "wasserstein",
) -> Dict:
    """
    Global Sensitivity Analysis for time series outputs using Optimal Transport.

    Parameters
    ----------
    problem : dict
        SALib-style problem dictionary with keys:
        - 'num_vars'
        - 'names'
    X : ndarray (N, D)
        Sample matrix
    Y : ndarray (N, T)
        Model output time series for each sample
    n_bins : int
        Number of stratified bins per input dimension
    metric : str
        Distance metric; currently supports 'wasserstein' (L2)

    Returns
    -------
    Dict[str, ndarray]
        Dictionary with sensitivity index per input (normalised)
    """

    assert (
        metric == "wasserstein"
    ), "Only 'wasserstein' metric is currently implemented."

    N, D = X.shape
    _, T = Y.shape
    num_vars = problem["num_vars"]
    assert (
        num_vars == D
    ), "Number of variables in problem dictionary must match X shape."
    assert T > 1, "Y must have more than one time step."
    assert n_bins > 0, "Number of bins must be greater than zero."

    names = problem["names"]

    # Normalize Y globally to [0,1] for numerical stability
    Y = (Y - Y.min()) / (Y.max() - Y.min() + 1e-12)

    # Create reference empirical distribution over full time series outputs
    Y_ref = Y.copy()
    mu = np.ones(N) / N

    # Compute pairwise ground cost matrix for reference distribution
    M = ot.dist(Y_ref, Y_ref, metric="euclidean") ** 2  # (N x N)

    indices = np.zeros(num_vars)

    for i in range(num_vars):
        xi = X[:, i]
        bins = np.quantile(xi, np.linspace(0, 1, n_bins + 1))
        bin_indices = np.digitize(xi, bins) - 1

        distances = []
        weights = []

        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) < 5:
                continue

            Y_bin = Y[mask, :]
            nb = Y_bin.shape[0]
            nu = np.ones(nb) / nb

            # Compute cost matrix between conditional and reference distributions
            M = ot.dist(Y_bin, Y_ref, metric="euclidean") ** 2
            dist = ot.emd2(nu, mu, M)
            distances.append(np.sqrt(dist))
            weights.append(nb / N)

        indices[i] = np.sum(np.array(distances) * np.array(weights))

    return {"names": names, "wasserstein": indices / np.sum(indices)}
