import numpy as np
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Parameters

def debug_matrix(matrix, print_string=None, depth=1):
	"""Inspect and print debugging info for given matrix."""
	print("+" * 10)
	if print_string:
		print(print_string)

	indent = "  " * depth

	# If it's Parameters dataclass
	if isinstance(matrix, Parameters):
		print(f"{indent}Type: Parameters")
		print(f"{indent}tensor_type: {matrix.tensor_type}")
		ndarrays = parameters_to_ndarrays(matrix)
		for idx, ndarray in enumerate(ndarrays):
			print(f"{indent}NDArray #{idx}: Shape: {ndarray.shape}")

	# If it's a numpy ndarray
	elif isinstance(matrix, np.ndarray):
		print(f"{indent}Type: numpy ndarray")
		print(f"{indent}Shape: {matrix.shape}")

	# If it's a list
	elif isinstance(matrix, list):
		print(f"{indent}Type: list")
		print(f"{indent}L Length: {len(matrix)}")
		for idx, item in enumerate(matrix):
			# Check if the list contains ndarrays
			if isinstance(item, np.ndarray):
				print(f"{indent}NDArray in list at index {idx}: Shape: {item.shape}")
				# debug_matrix(matrix, print_string=f"arr {idx} ", depth=2)
			else:
				print(f"{indent}Item in list at index {idx}: Type: {type(item)}")


	# If it's a dictionary
	elif isinstance(matrix, dict):
		print(f"{indent}Type: dictionary")
		print(f"{indent}D Length: {len(matrix)}")
		for key, value in matrix.items():
			print(f"{indent}Key: {key}, Type: {type(key)}")
			debug_matrix(value, depth + 1)

	# If it's a PyTorch Tensor
	elif "torch" in str(type(matrix)):
		print(f"{indent}Type: PyTorch Tensor")
		print(f"{indent}Shape: {matrix.shape}")

	# If it's another type
	else:
		print(f"{indent}Type: {type(matrix)}")
