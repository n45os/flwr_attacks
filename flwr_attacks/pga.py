import copy
import random
from typing import Callable, Dict, List, Tuple, Union

from flwr_attacks.model_poisoning import ModelPoisoningAttack
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (
    Strategy,
    Bulyan,
)
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)

from flwr.common.logger import log
from logging import ERROR

from flwr.client import Client

from flwr.common import NDArrays

from flwr.common.parameter import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from flwr_attacks.computations import (
    calculate_norm_of_ndarrays,
    scalar_multiply_ndarrays,
    subtract_ndarrays,
)


class PGAAttack(ModelPoisoningAttack):
    def __init__(
        self,
        *,
        adversary_fraction: int = 0.2,
        adversary_accessed_cids: List[str] = None,
        adversary_clients: List[str] = None,
        activation_round: int = 0,
        strategy: Strategy = None,
        client_fn: Callable[[str], ClientProxy] = None,
        reverse_train_fn: Callable[[int, float, NDArrays], NDArrays] = None,
        reverse_epochs: int = 1,
        reverse_lr: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(
            adversary_fraction=adversary_fraction,
            adversary_accessed_cids=adversary_accessed_cids,
            adversary_clients=adversary_clients,
            activation_round=activation_round,
            **kwargs,
        )

        if adversary_accessed_cids is not None:
            num_benign = len(adversary_accessed_cids)
        elif adversary_clients is not None:
            num_benign = len(adversary_clients)
        else:
            raise ValueError(
                "Adversary clients or adversary accessed cids not provided"
            )
        num_malicious = int(adversary_fraction * num_benign / (1 - adversary_fraction))

        # change the num_of_malicious clients to the strategies that need it based on the results that we have available
        # make a deep copy of the strategy
        self.strategy = copy.deepcopy(strategy)
        if hasattr(strategy, "num_malicious_clients"):
            if isinstance(strategy, Bulyan):
                while num_benign < 4 * num_malicious + 3:
                    num_malicious -= 1
            self.strategy.num_malicious_clients = num_malicious - 1

        if strategy is None:
            log(ERROR, "Strategy not provided")
            raise ValueError("Strategy not provided")

        self.client_fn = client_fn
        self.reverse_epochs = reverse_epochs
        self.reverse_lr = reverse_lr
        if reverse_train_fn is None:
            raise ValueError(
                "You haven't provided a reverse_train_fn method."
                "View instructions on how to implement it https://github.com/n45os/flwr_attacks/docs/pga-attack-implementation.md"
            )
        self.reverse_train_fn = reverse_train_fn

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
        params_copy = copy.deepcopy(parameters)

        useable_results: List[Tuple[ClientProxy, FitRes]] = [
            res
            for res in results
            if (
                res[0].cid in self.adversary_accessed_cids
                or res[0].cid in self.adversary_clients
            )
        ]

        # get the average num_examples of the results
        average_num_examples = int(
            sum([res[1].num_examples for res in results]) / len(useable_results)
        )

        used_updates: List[NDArrays] = [
            parameters_to_ndarrays(res[1].parameters) for res in useable_results
        ]

        tau = compute_τ(used_updates)

        # get a random client from the list of adversarial clients
        _client: Client = self.client_fn(
            cid=self.adversary_clients[
                random.randint(0, len(self.adversary_clients) - 1)
            ]
        )

        reversed_updates: NDArrays = self.reverse_train_fn(
            client=_client,
            epochs=self.reverse_epochs,
            lr=self.reverse_lr,
            updates=parameters_to_ndarrays(params_copy),
        )
        reversed_result = create_res_like(
            res=results[0],
            parameters=ndarrays_to_parameters(reversed_updates),
            num_examples=average_num_examples,
        )

        # project the reversed updates on a ball of radius tau
        if server_round > self.activation_round:
            gamma_succ = projection(
                tau=tau,
                attack_res=reversed_result,
                benign_results=useable_results,
                strategy=self.strategy,
                fraction_of_malicious=self.adversary_fraction,
                num_examples=average_num_examples,
            )

            malicious_gradient = scalar_multiply_ndarrays(
                scalar=gamma_succ, ndarrays=reversed_updates
            )

            for client, fit_res in results:
                if client.cid in self.adversary_clients:
                    fit_res.parameters = ndarrays_to_parameters(malicious_gradient)

        return results, failures, metrics


def create_res_like(
    res: Tuple[ClientProxy, FitRes], parameters: Parameters, num_examples: int
) -> Tuple[ClientProxy, FitRes]:
    return (
        res[0],
        FitRes(
            parameters=parameters,
            num_examples=num_examples,
            metrics=res[1].metrics,
            status=res[1].status,
        ),
    )


def projection(
    tau,
    attack_res: Tuple[ClientProxy, FitRes],
    benign_results: List[Tuple[ClientProxy, FitRes]],
    strategy: Strategy,
    fraction_of_malicious: float,
    num_examples: int,
) -> float:
    succ_d = 0
    succ_gamma = 1

    attack_update: NDArrays = parameters_to_ndarrays(attack_res[1].parameters)

    num_benign = len(benign_results)
    num_malicious = int(
        fraction_of_malicious * num_benign / (1 - fraction_of_malicious)
    )

    norm_attack_vector = calculate_norm_of_ndarrays(attack_update)

    scaled_attack_vector = scalar_multiply_ndarrays(
        scalar=(tau / norm_attack_vector),
        ndarrays=attack_update,
    )

    benign_aggregate, _ = strategy.aggregate_fit(
        server_round=0, results=benign_results, failures=[]
    )
    benign_aggregate_ndarrays: NDArrays = parameters_to_ndarrays(benign_aggregate)

    not_found_counter = 0
    found_counter = 0
    rrange = 5
    step = 0.01
    gamma = 0.00
    while gamma < rrange:
        new_attack_vector: NDArrays = scalar_multiply_ndarrays(
            gamma, scaled_attack_vector
        )

        attack_result: Tuple[ClientProxy, FitRes] = create_res_like(
            res=attack_res,
            parameters=ndarrays_to_parameters(new_attack_vector),
            num_examples=num_examples,
        )

        all_malicious: List[Tuple[ClientProxy, FitRes]] = [
            attack_result for _ in range(num_malicious)
        ]

        all_params_result: List[Tuple[ClientProxy, FitRes]] = (
            all_malicious + benign_results
        )
        attack_aggregate, _ = strategy.aggregate_fit(
            server_round=0, results=all_params_result, failures=[]
        )
        attack_aggregate_ndarrays: NDArrays = parameters_to_ndarrays(attack_aggregate)

        difference = subtract_ndarrays(
            attack_aggregate_ndarrays, benign_aggregate_ndarrays
        )

        d = calculate_norm_of_ndarrays(difference)

        if d > succ_d:
            succ_gamma = gamma
            succ_d = d
            found_counter += 1
            not_found_counter = 0
        else:
            not_found_counter += 1
            found_counter = 0
        if (
            not_found_counter > 120
        ):  # if we have not found a better gamma in 250 iterations, break
            break
        if (
            found_counter > 40
        ):  # if we have found a better gamma in 250 iterations, increase the gamma by 0.1
            step += 0.05

        gamma += step

    return succ_gamma


def compute_τ(benign_updates: List[NDArrays]) -> float:
    total_norm = 0.0
    num_updates = len(benign_updates)

    for update in benign_updates:
        total_norm += calculate_norm_of_ndarrays(update)

    tau = total_norm / num_updates
    return tau
