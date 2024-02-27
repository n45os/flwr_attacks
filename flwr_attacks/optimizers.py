from typing import Callable


def _optimizer2(
		objective_fn: Callable[[float], bool], gamma_init: float, tau: float, step: float
) -> float:
	gamma = gamma_init
	gamma_succ = gamma
	while True:
		if objective_fn(gamma):
			gamma_succ = gamma
			gamma = gamma + step / 2
		else:
			gamma = gamma - step / 2

		step = step / 2

		if abs(gamma_succ - gamma) < tau:
			break
	return gamma_succ


def _optimizer(
		objective_fn: Callable[..., bool],
		gamma_init: float,
		tau: float,
		step: float,
		use_previous: bool = False,
		max_iter=100
) -> float:
	"""
	A generic optimizer function.

	Parameters:
	- objective_fn (callable): The objective function to be optimized. This function
	  should return a boolean.
	- gamma_init (float): Initial value of gamma.
	- tau (float): A threshold for stopping the optimization.
	- step (float): The step size for updating gamma.
	- use_previous (bool): If True, the objective function uses the previous gamma and
	  iteration count.

	Returns:
	- float: The optimized value of gamma.
	"""
	gamma = gamma_init
	gamma_succ = gamma
	iteration = 0

	while True:
		if use_previous:
			if objective_fn(gamma, gamma_succ, iteration):
				gamma_succ = gamma
				gamma = gamma + step / 2
			else:
				gamma = gamma - step / 2
			iteration += 1
		else:
			if objective_fn(gamma):
				gamma_succ = gamma
				gamma = gamma + step / 2
			else:
				gamma = gamma - step / 2
			iteration += 1

		step = step / 2

		if abs(gamma_succ - gamma) < tau or iteration > max_iter:
			break

	return gamma_succ
