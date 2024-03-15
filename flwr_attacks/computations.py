from functools import reduce
from typing import List
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common import NDArrays
import numpy as np

# from flwr_attacks.helpers.debug_matrix import debug_matrix

def average_params2(ndarrays_list: List[NDArrays]) -> NDArrays:
	"""Average a list of NDArrays."""
	# debug_matrix(ndarrays_list, "ndarrays_list")
	summed_parameters = [np.zeros_like(param) for param in ndarrays_list[0]]
	# debug_matrix(summed_parameters, "summed_parameters")
	for params in ndarrays_list:
		for i, param in enumerate(params):
			summed_parameters[i] += param
	num_networks = len(ndarrays_list)
	averaged_parameters = [param_sum / num_networks for param_sum in summed_parameters]
	# debug_matrix(averaged_parameters, "AVG"
	return averaged_parameters

def average_params(ndarrays_list: List[NDArrays]) -> NDArrays:
    """Average a list of NDArrays using a method similar to the first snippet."""
    num_networks = len(ndarrays_list)
    averaged_parameters = [
        reduce(np.add, layer_updates) / num_networks
        for layer_updates in zip(*ndarrays_list)
    ]
    return averaged_parameters

def generate_perturbations(ndarray_list, perturbation_type):

	avg_params = average_params(ndarray_list)

	if perturbation_type == "random" or perturbation_type not in ["puv", "pstd", "psgn"]:
		# Random perturbation. Choose randomly from ["puv", "pstd", "psgn"]
		perturbation_type = np.random.choice(["puv", "pstd", "psgn"])
	
	if perturbation_type == "puv":
		# Inverse unit vector
		perturbation = [-layer / np.linalg.norm(layer) for layer in avg_params]
		
	elif perturbation_type == "pstd":
		# Inverse standard deviation
		perturbation = []
		for layer_idx in range(len(avg_params)):
			# Extract the specific layer across all clients' gradients
			layer_samples = [client_gradient[layer_idx] for client_gradient in ndarray_list]
			# Calculate standard deviation for that layer
			layer_std = np.std(layer_samples, axis=0)
			perturbation.append(-layer_std)
		
	elif perturbation_type == "psgn":
		# Inverse sign of the average
		perturbation = scalar_multiply_ndarrays(-1.0, avg_params)

	return perturbation


def generate_all_perturbations(ndarray_list):
	# Returns a list of all perturbations

	all_pertrunations = []
	# Inverse unit vector
	all_pertrunations.append(("puv", generate_perturbations(ndarray_list, "puv")))
	# Inverse standard deviation
	all_pertrunations.append(("pstd", generate_perturbations(ndarray_list, "pstd")))
	# Inverse sign of the average
	all_pertrunations.append(("psgn", generate_perturbations(ndarray_list, "psgn")))

	return all_pertrunations


def compute_malicious(ndarray_list: List[NDArrays], perturbation: NDArrays, gamma: float) -> NDArrays:
	"""
	Computing the sum of the average of the accessed gradients plus gamma times the perturbation vector
	"""
	avg_gradient = average_params(ndarray_list)
	perturbed_sum = [avg + gamma * perturb for avg, perturb in zip(avg_gradient, perturbation)]
	return perturbed_sum

def compute_l2_norm(gr1: NDArrays, gr2: NDArrays) -> float:
	if len(gr1) != len(gr2):	
		raise ValueError("The two gradients lists must have the same length.")
	squared_diff_sum = 0.0
	for layer1, layer2 in zip(gr1, gr2):
		squared_diff_sum += np.sum(np.square(layer1 - layer2))
	return np.sqrt(squared_diff_sum)

# def compute_norm(gr: NDArrays) -> float:
# 	for arr in gr:
# 			total_norm += np.linalg.norm(arr)
# 			count += 1

def compute_l2_norm_squared(gr1: NDArrays, gr2: NDArrays) -> float:
	l2_norm = compute_l2_norm(gr1, gr2)
	return l2_norm * l2_norm

def scalar_multiply_ndarrays(scalar: float, ndarrays: NDArrays) -> NDArrays:
	"""Multiply a scalar with a list of NumPy ndarrays."""
	return [scalar * ndarray for ndarray in ndarrays]

def subtract_ndarrays(ndarrays1: NDArrays, ndarrays2: NDArrays) -> NDArrays:
	"""Subtract a list of NumPy ndarrays (ndarrays2) from another (ndarrays1)."""
	return [ndarray1 - ndarray2 for ndarray1, ndarray2 in zip(ndarrays1, ndarrays2)]

def calculate_norm_of_ndarrays(vectors: NDArrays) -> float:
	# Flatten each array and concatenate them into a single one-dimensional array
	flat_vectors = [vector.flatten() for vector in vectors]
	concatenated_vector = np.concatenate(flat_vectors)
	# Calculate the norm of the concatenated vector
	norm = np.linalg.norm(concatenated_vector)
	return norm