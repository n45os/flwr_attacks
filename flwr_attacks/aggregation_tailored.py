from typing import Dict, List, Tuple, Union
from flwr_attacks.model_poisoning import ModelPoisoningAttack
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (
	Strategy,
	FedAvg,
	FedMedian,
	Bulyan,
	Krum,
	FedTrimmedAvg,
)
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
from flwr_attacks.algorithms import attack_multikrum, attack_fedmedian, attack_krum, attack_trimmedmean, brute_attack


class AggregationTailored(ModelPoisoningAttack):
	def __init__(
		self,
		*,
		adversary_fraction: int = 0.2,
		adversary_accessed_cids: List[str] = None,
		adversary_clients: List[str] = None,
		activation_round: int = 0,
		perturbation_type=None,
		gamma_init: float = 10.0,  # Initial value, can be adjusted based on domain knowledge
		tau: float = 0.001,  # Threshold for stopping the optimization
		step: float = None,  # step size
		beta: float = 0.1,  # Trimmed mean parameter
		strategy: Strategy = FedAvg(),
		** kwargs,
	) -> None:
		super().__init__(
			adversary_fraction=adversary_fraction,
			adversary_accessed_cids=adversary_accessed_cids,
			adversary_clients=adversary_clients,
			activation_round=activation_round,
			**kwargs,
		)
		self.strategy = strategy
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
		if server_round > self.activation_round:

			adversial_parameters: List[Tuple[NDArrays, bool]] = self.extract_adversaries(
				results
			)
			if len(adversial_parameters) == 0:
				return results, failures, metrics

			print("Adversarial parameters len : ", len(adversial_parameters))

			perturbation = generate_perturbations(
				[params for params, _ in adversial_parameters], self.perturbation_type
			)

			num_benign = len(adversial_parameters)
			num_malicious = int(
				self.adversary_fraction * num_benign / (1 - self.adversary_fraction)
			)

			print("Number of malicious clients: ", num_malicious)
			print("Number of benign clients: ", num_benign)
			
			if isinstance(self.strategy, FedMedian):
				print("FedMedian")
				malicious_gradient = attack_fedmedian(
					[params for params, _ in adversial_parameters],
					perturbation,
					num_malicious,
					self.gamma,
					self.tau,
					self.step,
				)
			elif isinstance(self.strategy, Krum):
				if self.strategy.num_clients_to_keep > 0:
					# multikrum
					print("MultiKrum")
					malicious_gradient = attack_multikrum(
						[params for params, _ in adversial_parameters],
						perturbation,
						num_malicious,
						self.gamma,
						self.tau,
						self.step,
					)
				else:
					# krum
					print("Krum")
					malicious_gradient = attack_krum(
						[params for params, _ in adversial_parameters],
						perturbation,
						num_malicious,
						self.gamma,
						self.tau,
						self.step,
					)
			elif isinstance(self.strategy, Bulyan):
				print("Bulyan")
				malicious_gradient = attack_multikrum(
					[params for params, _ in adversial_parameters],
					perturbation,
					num_malicious,
					self.gamma,
					self.tau,
					self.step,
				)
			elif isinstance(self.strategy, FedTrimmedAvg):
				print("FedTrimmedAvg")
				malicious_gradient = attack_trimmedmean(
					[params for params, _ in adversial_parameters],
					perturbation,
					num_malicious,
					self.strategy.beta,
					self.gamma,
					self.tau,
					self.step,
				)
				
			else:
				print("Unknown aggregation strategy. Applies a really big perturbation")
				malicious_gradient = brute_attack(
					[params for params, _ in adversial_parameters],
					perturbation,
					gamma = 100,				
				)

			for client, fit_res in results:
				if client.cid in self.adversary_clients:
					fit_res.parameters = ndarrays_to_parameters(malicious_gradient)

		return results, failures, metrics
