# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""

import copy
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr_attacks.attack import Attack

from flwr.common import (
	DisconnectRes,
	EvaluateRes,
	FitRes,
	Parameters,
	Scalar,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.server import Server, fit_clients

FitResultsAndFailures = Tuple[
	List[Tuple[ClientProxy, FitRes]],
	List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
	List[Tuple[ClientProxy, EvaluateRes]],
	List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
	List[Tuple[ClientProxy, DisconnectRes]],
	List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class AttackServer(Server):
	"""Flower server with the addition applying adversary attacks."""

	def __init__(
		self,
		*,
		client_manager: Optional[ClientManager] = SimpleClientManager(),
		strategy: Optional[Strategy] = None,
		attack: Optional[Attack] = None,
	) -> None:
		self._client_manager: ClientManager = client_manager
		self.parameters: Parameters = Parameters(
			tensors=[], tensor_type="numpy.ndarray"
		)
		self.strategy: Strategy = strategy if strategy is not None else FedAvg()
		self.attack: Attack = attack
		self.max_workers: Optional[int] = None

	# pylint: disable=too-many-locals
	def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
		"""Run federated averaging for a number of rounds."""
		history = History()

		# Initialize parameters
		log(INFO, "Initializing global parameters")
		self.parameters = self._get_initial_parameters(timeout=timeout)
		log(INFO, "Evaluating initial parameters")
		res = self.strategy.evaluate(0, parameters=self.parameters)
		if res is not None:
			log(
				INFO,
				"initial parameters (loss, other metrics): %s, %s",
				res[0],
				res[1],
			)
			history.add_loss_centralized(server_round=0, loss=res[0])
			history.add_metrics_centralized(server_round=0, metrics=res[1])

		# Run federated learning for num_rounds
		log(INFO, "FL starting")
		start_time = timeit.default_timer()

		for current_round in range(1, num_rounds + 1):
			# Train model and replace previous global model
			res_fit = self.fit_round(
				server_round=current_round,
				timeout=timeout,
			)
			attacked_parameters_prime = None
			if res_fit is not None:
				(
					parameters_prime,
					attacked_parameters_prime,
					fit_metrics,
					_,
				) = res_fit  # fit_metrics_aggregated
				if attacked_parameters_prime:
					# set the attacked parameters in the global model and save the benign parameters to use them in evaluation
					self.parameters = attacked_parameters_prime
					benign_parameters_prime = parameters_prime
				else:
					# If there are no attacked parameters, then we will use the normal parameters
					self.parameters = parameters_prime
					benign_parameters_prime = parameters_prime
				history.add_metrics_distributed_fit(
					server_round=current_round, metrics=fit_metrics
				)

			res_cen = None
			if attacked_parameters_prime is not None:
				# Evaluate attacked model using strategy implementation
				res_cen = self.strategy.evaluate(
					current_round, parameters=self.parameters
				)
				if res_cen is not None:
					loss_cen, metrics_cen = res_cen
					log(
						INFO,
						"[🔫] fit progress on attacked params: (%s, %s, %s, %s)",
						current_round,
						loss_cen,
						metrics_cen,
						timeit.default_timer() - start_time,
					)
					history.add_loss_centralized(
						server_round=current_round, loss=loss_cen
					)
					history.add_metrics_centralized(
						server_round=current_round, metrics=metrics_cen
					)

				# Evaluate attacked model on a sample of available clients
				res_fed = self.evaluate_round(
					server_round=current_round, timeout=timeout
				)
				if res_fed is not None:
					loss_fed, evaluate_metrics_fed, _ = res_fed
					if loss_fed is not None:
						history.add_loss_distributed(
							server_round=current_round, loss=loss_fed
						)
						history.add_metrics_distributed(
							server_round=current_round, metrics=evaluate_metrics_fed
						)

			# Evaluate benign model using strategy implementation
			ben_res_cen = self.strategy.evaluate(
				current_round, parameters=benign_parameters_prime
			)
			if ben_res_cen is not None:
				loss_cen, metrics_cen = ben_res_cen
				log(
					INFO,
					"[🔐] fit progress on benign params: (%s, %s, %s, %s)",
					current_round,
					loss_cen,
					metrics_cen,
					timeit.default_timer() - start_time,
				)
				history.add_loss_centralized(server_round=current_round, loss=loss_cen)

			# Evaluate model on a sample of available clients
			res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
			if res_fed is not None:
				loss_fed, evaluate_metrics_fed, _ = res_fed
				if loss_fed is not None:
					history.add_loss_distributed(
						server_round=current_round, loss=loss_fed
					)

			# compute attack impact for round
			if res_cen is not None and ben_res_cen is not None:
				round_impact = self.attack.compute_impact(
					attack_acc=res_cen[1]["accuracy"],
					benign_acc=ben_res_cen[1]["accuracy"],
				)
				history.add_metrics_centralized(
					server_round=current_round, metrics={"round_impact": round_impact}
				)
				log(INFO, "[🔥] attack round_impact: %s", round_impact)

		# Bookkeeping
		end_time = timeit.default_timer()
		elapsed = end_time - start_time
		log(INFO, "FL finished in %s", elapsed)
		return history

	def fit_round(
		self,
		server_round: int,
		timeout: Optional[float],
	) -> Optional[
		Tuple[
			Optional[Parameters],
			Optional[Parameters],
			Dict[str, Scalar],
			FitResultsAndFailures,
		]
	]:
		"""Perform a single round of federated averaging."""
		# Get clients and their respective instructions from strategy

		metrics = {}

		client_instructions = self.strategy.configure_fit(
			server_round=server_round,
			parameters=self.parameters,
			client_manager=self._client_manager,
		)

		if not client_instructions:
			log(INFO, "fit_round %s: no clients selected, cancel", server_round)
			return None
		log(
			DEBUG,
			"fit_round %s: strategy sampled %s clients (out of %s)",
			server_round,
			len(client_instructions),
			self._client_manager.num_available(),
		)

		# Collect `fit` results from all clients participating in this round
		results, failures = fit_clients(
			client_instructions=client_instructions,
			max_workers=self.max_workers,
			timeout=timeout,
		)
		log(
			DEBUG,
			"fit_round %s received %s results and %s failures",
			server_round,
			len(results),
			len(failures),
		)

		# Attack model and replace previous global model
		# If the user provides an attack, then we will use it
		att_parameters_aggregated = None
		if self.attack is not None and self.attack.adversary_fraction > 0 and self.attack.activation_round <= server_round:
			res_attack = self.attack.attack_fit(
				server_round=server_round,
				results=copy.deepcopy(results),
				failures=copy.deepcopy(failures),
				parameters=self.parameters,
			)
			if res_attack is not None:
				att_results, att_failures, attack_metrics = res_attack
				log(
					DEBUG,
					"fit_round %s applied the attack successfully, received %s results and %s failures",
					server_round,
					len(att_results),
					len(att_failures),
				)
				# Add the attack metrics to the metrics dictionary
				metrics.update(attack_metrics)

			# We will aggregate the both the attacked results and the normal results for comparison
			# Aggregate attacked training results
			att_aggregated_result: Tuple[
				Optional[Parameters],
				Dict[str, Scalar],
			] = self.strategy.aggregate_fit(server_round, att_results, att_failures)

			att_parameters_aggregated, att_metrics_aggregated = att_aggregated_result
			metrics["adversial_metrics"] = att_metrics_aggregated

		# Aggregate benign training results
		aggregated_result: Tuple[
			Optional[Parameters],
			Dict[str, Scalar],
		] = self.strategy.aggregate_fit(server_round, results, failures)

		parameters_aggregated, metrics_aggregated = aggregated_result
		metrics["benign_metrics"] = metrics_aggregated

		return (
			parameters_aggregated,
			att_parameters_aggregated,
			metrics,
			(results, failures),
		)
