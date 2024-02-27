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
from flwr_attacks.computations import generate_perturbations


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

		perturbation = generate_perturbations(
			[params for params, _ in adversial_parameters], self.perturbation_type
		)

		if server_round > self.activation_round:
			malicious_gradient = minsum_attack(
				[params for params, _ in adversial_parameters],
				perturbation,
				self.gamma,
				self.tau,
				self.step,
			)

			for client, fit_res in results:
				if client.cid in self.adversary_clients:
					fit_res.parameters = ndarrays_to_parameters(malicious_gradient)

		return results, failures, metrics
