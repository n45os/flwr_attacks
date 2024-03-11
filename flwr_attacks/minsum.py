from typing import Dict, List, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
	FitRes,
	Parameters,
	Scalar,
)

from flwr.common import NDArrays

from flwr.common.parameter import (
	ndarrays_to_parameters,
)

from flwr_attacks.algorithms import minsum_attack
from flwr_attacks.minmax import MinMax
from flwr_attacks.computations import generate_all_perturbations, generate_perturbations


class MinSum(MinMax):
	def attack_fit(
		self,
		server_round: int,
		results: List[Tuple[ClientProxy, FitRes]],
		failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
		parameters: Parameters,
	) -> Tuple[
		List[Tuple[ClientProxy, FitRes]],
		List[Union[Tuple[ClientProxy, FitRes], BaseException]],
		Dict[str, Scalar],
	]:
		metrics = {}

		adversial_parameters: List[Tuple[NDArrays, bool]] = self.extract_adversaries(
			results
		)
		if len(adversial_parameters) == 0:
			return results, failures, metrics

		if server_round > self.activation_round:
			if self.perturbation_type == "optimal":
				perturbations = generate_all_perturbations(
					[params for params, _ in adversial_parameters]
				)
				best_perturbation_type = None
				best_malicious_gradient = None
				best_gamma = 0
				for perturbation_type, perturbation in perturbations:
					malicious_gradient, gamma = minsum_attack(
						[params for params, _ in adversial_parameters],
						perturbation,
						self.gamma_init,
						self.tau,
						self.step,
					)
					if gamma > best_gamma:
						best_gamma = gamma
						best_perturbation_type = perturbation_type
						best_malicious_gradient = malicious_gradient

				perturbation_type = best_perturbation_type
				optimized_gamma = best_gamma
				malicious_gradient = best_malicious_gradient

			else:
				perturbation = generate_perturbations(
					[params for params, _ in adversial_parameters], self.perturbation_type
				)
				perturbation_type = self.perturbation_type
				malicious_gradient, optimized_gamma = minsum_attack(
					[params for params, _ in adversial_parameters],
					perturbation,
					self.gamma,
					self.tau,
					self.step,
				)

			# Add gamma to the metrics
			metrics["gamma"] = optimized_gamma
			metrics["perturbation_type"] = perturbation_type

			for client, fit_res in results:
				if client.cid in self.adversary_clients:
					fit_res.parameters = ndarrays_to_parameters(malicious_gradient)

		return results, failures, metrics
