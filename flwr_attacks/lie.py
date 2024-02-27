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
from flwr_attacks.algorithms import lie_attack


class LieAttack(ModelPoisoningAttack):
	def __init__(
		self,
		*,
		adversary_fraction: int = 0.2,
		adversary_accessed_cids: List[str] = None,
		adversary_clients: List[str] = None,
		activation_round: int = 0,
		total_clients: int,

		**kwargs,
	) -> None:
		super().__init__(
			adversary_fraction=adversary_fraction,
			adversary_accessed_cids=adversary_accessed_cids,
			adversary_clients=adversary_clients,
			activation_round=activation_round,
			**kwargs,
		)
		self.total_clients = total_clients

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
			malicious_gradient = lie_attack(
				[params for params, _ in adversial_parameters],
				total_clients=self.total_clients,
				total_malicious=int(self.total_clients*self.adversary_fraction),
			)
			
			for client, fit_res in results:
				if client.cid in self.adversary_clients:
					fit_res.parameters = ndarrays_to_parameters(malicious_gradient)
		
		return results, failures, metrics
					
			
