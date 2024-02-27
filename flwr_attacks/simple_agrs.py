from flwr_attacks.helpers.debug_matrix import debug_matrix  # noqa: F401

from typing import List
from flwr.common import NDArrays
from flwr.server.strategy.aggregate import _compute_distances, _trim_mean 
import numpy as np

"""
The simple aggregations are based on the flwr framework.
They are changed mostly on their inputs and outputs in order to make them easier to implement in the attacks.
"""


def aggregate_krum_simple(
    params: List[NDArrays], num_malicious: int, to_keep: int
) -> List[int]:
    """
    Got it from the flwr framework and changed a bit to match my case.
    """

    # Compute distances between vectors
    distance_matrix = _compute_distances(params)

    # For each client, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(params) - num_malicious - 2)
    closest_indices = []
    for i, _ in enumerate(distance_matrix):
        closest_indices.append(
            np.argsort(distance_matrix[i])[1 : num_closest + 1].tolist()  # noqa: E203
        )

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]  # noqa: E203
        return best_indices

    # Return the index of the client which minimizes the score (Krum)
    return [np.argmin(scores)]


def aggregate_median_simple(params: List[NDArrays]) -> NDArrays:
    # Initialize a list to hold the median values
    median_w: NDArrays = [
        np.median(np.asarray(layer), axis=0) for layer in zip(*params)
    ]
    return median_w


def aggregate_trimmed_avg_simple(
    params, proportiontocut: float
) -> NDArrays:
    """Compute trimmed average."""
    # Create a list of weights and ignore the number of examples

    trimmed_w: NDArrays = [
        _trim_mean(np.asarray(layer), proportiontocut=proportiontocut)
        for layer in zip(*params)
    ]

    return trimmed_w

