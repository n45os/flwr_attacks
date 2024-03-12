import copy
import math
from typing import Callable, List

import numpy as np

from flwr.common import NDArray, NDArrays

from flwr_attacks.computations import compute_malicious, compute_l2_norm, average_params
from flwr_attacks.optimizers import _optimizer
from flwr_attacks.simple_agrs import aggregate_krum_simple, aggregate_median_simple, aggregate_trimmed_avg_simple


def minmax_attack(
		ndarray_list: List[NDArrays],
		perturbation: NDArrays,
		gamma_init: float,
		tau: float,
		step: float,
) -> NDArrays:
	"""
	Computing the sum of the average of the accessed gradients plus gamma times the perturbation vector
	"""
	def minmax_objective_fn(gamma):
		# Compute malicious gradient
		malicious_gradient = compute_malicious(ndarray_list, perturbation, gamma)
		# Include the malicious gradient in the distance computation
		weights_with_malicious = ndarray_list + [malicious_gradient]

		# Compute distance matrix
		distance_matrix = _compute_distances(weights_with_malicious)

		# Max distance from malicious
		max_distance_from_malicious = np.max(distance_matrix[-1, :-1])

		# Max benign distance
		benign_distances = np.triu(distance_matrix[:-1, :-1], k=1)
		max_benign_distance = np.max(benign_distances)

		return max_distance_from_malicious <= max_benign_distance
	# Compute malicious gradient
	
	gamma = _optimizer(minmax_objective_fn, gamma_init, tau, step)
	malicious_gradient = compute_malicious(ndarray_list, perturbation, gamma)
	return malicious_gradient, gamma


def minsum_attack(
		ndarray_list: List[NDArrays],
		perturbation: NDArrays,
		gamma_init: float,
		tau: float,
		step: float,
) -> NDArrays:
	"""
	Computing the sum of the average of the accessed gradients plus gamma times the perturbation vector
	"""

	def minsum_objective_fn(gamma_obj):
		# get a deep copy of the ndarray list (this is the list with the benign updates of the advs)
		ndarray_list_copy = copy.deepcopy(ndarray_list)

		# compute the malicious gradient
		new_malicious_gradient = compute_malicious(ndarray_list_copy, perturbation, gamma_obj)

		# append the malicious gradient to the list
		ndarray_list_copy.append(new_malicious_gradient)

		# compute the distance matrix
		distance_matrix = _compute_distances(ndarray_list_copy)

		# get the sum of the distances of each node
		list_of_sums = []
		for i, _ in enumerate(ndarray_list_copy):
			list_of_sums.append(np.sum(distance_matrix[i]))

		# get the malicious sums
		malicious_sums = list_of_sums[-1]

		# get the max of the benign sums
		max_benign_sums = np.max(list_of_sums[:-1])

		del ndarray_list_copy

		return malicious_sums <= max_benign_sums

	gamma = _optimizer(minsum_objective_fn, gamma_init, tau, step)
	malicious_gradient = compute_malicious(ndarray_list, perturbation, gamma)
	return malicious_gradient, gamma


def attack_multikrum(
		ndarray_list: List[NDArrays],
		perturbation: NDArrays,
		num_of_malicious: int,
		gamma_init: float,
		tau: float,
		step: float,
) -> NDArrays:
	def multikrum_objective_fn(_gamma):
		# Compute malicious gradient
		_malicious_gradient = compute_malicious(ndarray_list, perturbation, _gamma)

		# Perform krum aggregation to <num_malicious_clients> parameters together with the known
		# 	parameters. I will keep track of the indices of the malicious. The goal is the
		# 	krum aggregation to return a malicious index. While it doesn't happen, I'll
		# 	optimize the gamma.
		all_malicious: List[NDArrays] = [
			_malicious_gradient for _ in range(num_of_malicious)
		]
		list_to_check: List[NDArrays] = all_malicious + ndarray_list

		index_list = aggregate_krum_simple(
			list_to_check,
			# num_malicious=aggregation_specific["num_malicious_clients"],
			num_malicious=num_of_malicious,
			# to make sure that all the malicious are selected
			to_keep=num_of_malicious + 2,
		)


		# The list will be like [💀, 💀, 💀, 💀, 💀 ... , 😇, 😇, 😇, 😇, ...]
		# 	If multikrum krum selects all of the malicious clients, then gamma is optimized well.
		
		malicious_set = set(range(num_of_malicious))
		result = malicious_set.issubset(set(index_list))

		if _gamma < 0.001:
			return True
		return result

	gamma = _optimizer(multikrum_objective_fn, gamma_init, tau, step)
	malicious_gradient = compute_malicious(ndarray_list, perturbation, gamma)
	return malicious_gradient


def attack_krum(
		ndarray_list: List[NDArrays],
		perturbation: NDArrays,
		num_of_malicious: int,
		gamma_init: float,
		tau: float,
		step: float,
) -> NDArrays:
	def krum_objective_fn(_gamma):
		# Compute malicious gradient
		_malicious_gradient = compute_malicious(
			ndarray_list, perturbation, _gamma
		)

		# Perform krum aggregation to <num_malicious_clients> parameters together with the known
		# 	parameters. I will keep track of the indices of the malicious. The goal is the
		# 	krum aggregation to return a malicious index. While it doesn't happen, I'll
		# 	optimize the gamma.

		all_malicious: List[NDArrays] = [
			_malicious_gradient for _ in range(num_of_malicious)
		]
		list_to_check: List[NDArrays] = all_malicious + ndarray_list

		index = aggregate_krum_simple(
			list_to_check,
			num_malicious=0,
			to_keep=0,
		)[0]

		# The list will be like [💀, 💀, 💀, 💀, 💀 ... , 😇, 😇, 😇, 😇, ...]
		# 	If krum selects one of the malicious clients, then gamma is optimized well.

		if _gamma < 0.0001:
			return True

		return index < num_of_malicious

	gamma = _optimizer(krum_objective_fn, gamma_init, tau, step)
	malicious_gradient = compute_malicious(ndarray_list, perturbation, gamma)
	return malicious_gradient


def attack_fedmedian(
		ndarray_list: List[NDArrays],
		perturbation: NDArrays,
		num_of_malicious: int,
		gamma_init: float,
		tau: float,
		step: float,
) -> NDArrays:
	def fedmedian_objective_fn(_gamma, gamma_succ, i):

		# Compute malicious gradient
		old_malicious_gradient = compute_malicious(
			ndarray_list, perturbation, gamma_succ
		)
		_malicious_gradient = compute_malicious(
			ndarray_list, perturbation, _gamma
		)
		average_benign = average_params(ndarray_list)

		all_malicious: List[NDArrays] = [
			_malicious_gradient for _ in range(num_of_malicious)
		]
		list_to_check: List[NDArrays] = all_malicious + ndarray_list
		median: NDArrays = aggregate_median_simple(list_to_check)

		all_old_malicious: List[NDArrays] = [
			old_malicious_gradient for _ in range(num_of_malicious)
		]
		old_list: List[NDArrays] = all_old_malicious + ndarray_list
		old_median: NDArrays = aggregate_median_simple(old_list)

		l2norm = compute_l2_norm(average_benign, median)
		l2norm_old = compute_l2_norm(average_benign, old_median)

		if _gamma < 0.0001:
			return True

		return l2norm >= l2norm_old

	gamma = _optimizer(fedmedian_objective_fn, gamma_init, tau, step, use_previous=True)
	malicious_gradient = compute_malicious(ndarray_list, perturbation, gamma)
	return malicious_gradient


def attack_trimmedmean(
		ndarray_list: List[NDArrays],
		perturbation: NDArrays,
		num_of_malicious: int,
		beta: float,
		gamma_init: float,
		tau: float,
		step: float,
) -> NDArrays:
	def trimmedmean_objective_fn(_gamma, gamma_succ, i):
		# Compute malicious gradient
		old_malicious_gradient = compute_malicious(
			ndarray_list, perturbation, gamma_succ
		)
		_malicious_gradient = compute_malicious(
			ndarray_list, perturbation, _gamma
		)
		average_benign = average_params(ndarray_list)

		all_malicious: List[NDArrays] = [
			_malicious_gradient for _ in range(num_of_malicious)
		]
		list_to_check: List[NDArrays] = all_malicious + ndarray_list
		trimmed: NDArrays = aggregate_trimmed_avg_simple(
			list_to_check, beta
		)

		all_old_malicious: List[NDArrays] = [
			old_malicious_gradient for _ in range(num_of_malicious)
		]
		old_list: List[NDArrays] = all_old_malicious + ndarray_list
		old_trimmed: NDArrays = aggregate_trimmed_avg_simple(
			old_list, beta
		)

		l2norm = compute_l2_norm(average_benign, trimmed)
		l2norm_old = compute_l2_norm(average_benign, old_trimmed)

		if _gamma < 0.0001:
			return True

		return l2norm >= l2norm_old

	gamma = _optimizer(
		trimmedmean_objective_fn, gamma_init, tau, step, use_previous=True
	)
	malicious_gradient = compute_malicious(ndarray_list, perturbation, gamma)
	return malicious_gradient


def brute_attack(
		ndarray_list: List[NDArrays],
		perturbation: NDArrays,
		gamma: float = 1000,
) -> NDArrays:
	"""
	Applies a really big gamma to the given perturbation.
	Can be used in FedAvg and in the FANG attacks based on the Maniputaltion The Byzantine papaer.
	"""
	malicious_gradient = compute_malicious(ndarray_list, perturbation, gamma)
	return malicious_gradient


def lie_attack(
		clients_params_list: List[NDArrays],
		total_clients: int,
		total_malicious: int,
) -> NDArrays:
	s = (total_clients // 2 + 1) - total_malicious
	z_max = max(
		z
		for z in np.linspace(-3, 3, 1000)
		if 0.5 * (1 + math.erf(z / np.sqrt(2))) < (total_clients - s) / total_clients
	)
	malicious_parameters = []

	# Iterate over each layer
	for i, layer in enumerate(zip(*clients_params_list)):
		mu = np.mean(layer, axis=0)
		sigma = np.std(layer, axis=0)
		p_mal = mu + z_max * sigma
		malicious_parameters.append(p_mal)

	# Convert lists of lists to lists of NDArrays
	malicious_parameters = [np.array(layer) for layer in malicious_parameters]

	return malicious_parameters


def _compute_distances(weights: List[NDArrays]) -> NDArray:
	"""Compute distances between vectors.

	Input: weights - list of weights vectors
	Output: distances - matrix distance_matrix of squared distances between the vectors
	"""
	flat_w = np.array([np.concatenate(p, axis=None).ravel() for p in weights])
	distance_matrix = np.zeros((len(weights), len(weights)))
	for i, _ in enumerate(flat_w):
		for j, _ in enumerate(flat_w):
			delta = flat_w[i] - flat_w[j]
			norm = np.linalg.norm(delta)
			distance_matrix[i, j] = norm ** 2
	return distance_matrix
