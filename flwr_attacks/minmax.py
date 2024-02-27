from typing import Dict, List, Tuple, Union
from flwr_attacks.model_poisoning import ModelPoisoningAttack
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

from flwr_attacks.computations import generate_perturbations
from flwr_attacks.algorithms import minmax_attack


class MinMax(ModelPoisoningAttack):
	def __init__(
		self,
		*,
		adversary_fraction: int = 0.2,
		adversary_accessed_cids: List[str] = None,
		adversary_clients: List[str] = None,
		activation_round: int = 0,
        perturbation_type="psgn",  # puv: Inverse unit vector, pstd: Inverse standard deviation, psgn: Inverse sign of the average
		gamma_init: float = 10.0,  # Initial value, can be adjusted based on domain knowledge
		tau: float = 0.001,  # Threshold for stopping the optimization
		step: float = 0.1,  # step size
		**kwargs,
	) -> None:
		super().__init__(
			adversary_fraction=adversary_fraction,
			adversary_accessed_cids=adversary_accessed_cids,
			adversary_clients=adversary_clients,
			activation_round=activation_round,
			**kwargs,
		)
		self.perturbation_type = perturbation_type
		self.gamma_init = gamma_init
		self.tau = tau
		if step is None:
			self.step = gamma_init
		else:
			self.step = step
		self.gamma = gamma_init

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

		print(f"extracted {len(adversial_parameters)} adversarial parameters.")

		perturbation = generate_perturbations(
			[params for params, _ in adversial_parameters], self.perturbation_type
		)

		if server_round > self.activation_round:
			malicious_gradient = minmax_attack(
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
