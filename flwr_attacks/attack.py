"""Flower server attack simulation."""


from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import NDArrays


class Attack(ABC):
	"""Abstract base class for server attack simulaiton implementations."""

	
	@abstractmethod
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
		"""Aggregate training results.

		Parameters
		----------
		server_round : int
			The current round of federated learning.
		results : List[Tuple[ClientProxy, FitRes]]
			Successful updates from the previously selected and configured
			clients. Each pair of `(ClientProxy, FitRes)` constitutes a
			successful update from one of the previously selected clients. Not
			that not all previously selected clients are necessarily included in
			this list: a client might drop out and not submit a result. For each
			client that did not submit an update, there should be an `Exception`
			in `failures`.
		failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
			Exceptions that occurred while the server was waiting for client
			updates.
		parameters : Parameters
			The current (global) model parameters.

		Returns
		-------
		Tuple[List[Tuple[ClientProxy, FitRes]], List[Union[Tuple[ClientProxy, FitRes], BaseException]]]
			Training results with applied attack and failures.
		"""


	@abstractmethod
	def extract_adversaries(results: List[Tuple[ClientProxy, FitRes]]) -> List[Tuple[NDArrays, bool]]:
		"""Extract the list of malicious clients.

		Parameters
		----------
		results : List[Tuple[ClientProxy, FitRes]]
			Successful updates from the previously selected and configured
			clients. Each pair of `(ClientProxy, FitRes)` constitutes a
			successful update from one of the previously selected clients. Not
			that not all previously selected clients are necessarily included in
			this list: a client might drop out and not submit a result. For each
			client that did not submit an update, there should be an `Exception`
			in `failures`.

		Returns
		-------
		List[Tuple[NDArrays, bool]]
			The list of malicious clients and whether they are adversarial or not.
		"""
	
	@abstractmethod
	def compute_impact(self, attack_acc: float, benign_acc: float) -> float:
		"""Compute the impact of the attack.

		Parameters
		----------
		attack_acc : float
			The accuracy of the malicious clients.
		benign_acc : float
			The accuracy of the benign clients.

		Returns
		-------
		float
			The impact of the attack.
		"""
		