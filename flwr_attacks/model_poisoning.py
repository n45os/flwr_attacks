from logging import DEBUG, INFO, ERROR  # noqa: F401
from flwr.common.logger import log
from typing import Dict, List, Tuple, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)


from flwr.server.client_proxy import ClientProxy
from flwr.common.parameter import (
    parameters_to_ndarrays,
)


from flwr_attacks.attack import Attack
from flwr.common import NDArrays


class ModelPoisoningAttack(Attack):
    def __init__(
        self,
        *,
        adversary_fraction: int = 0.2,  # this is based on the knowledge that the adversary has on malicious
        adversary_accessed_cids: List[
            str
        ] = None,  # the cids that the adversary has access to. If None, the adversary has access only to the adversarial clients
        adversary_clients: List[
            str
        ] = None,  # the cids of the adversarial clients. These clients will be replaced with the attacked parameters
        activation_round: int = 0,  # the round that the attack will be activated
        **kwargs,  # the aggregation specific arguments. for examle, krum needs to have num_malicious_clients and num_clients_to_keep arguments
    ) -> None:
        if adversary_clients is not None:
            self.adversary_clients = adversary_clients
        else:
            raise ValueError(
                " Provide a list of the adversarial cids with the adversary_clients argument and optionally define the benign access with the adversary_accessed_cids argument."
            )

        if adversary_accessed_cids is None:
            self.adversary_accessed_cids = self.adversary_clients

        self.activation_round = activation_round
        self.adversary_fraction = adversary_fraction
        self.kwargs = kwargs  # for example in krum i want malicious_count and to_keep

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
        """
        this is the base class for model poisoning attacks
        """
        log(
            ERROR,
            "This is the base class for model poisoning attacks. Please use a specific attack class.",
        )
        pass

    def extract_adversaries(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[Tuple[NDArrays, bool]]:
        # get only the adversarial clients
        # the advers clients variable is a list of tuples.
        # it contains the parameters of the adversarial clients and a boolean that is true if the client is adversarial
        advers_results = []
        print(f"extracting adversarial parameters from {len(results)} results.")
        if self.adversary_clients is not None:
            # if the cid is in the list of adversarial clients, then it is adversarial
            advers_results = [
                (
                    parameters_to_ndarrays(fit_res.parameters),
                    cproxy.cid in self.adversary_clients,
                )
                for cproxy, fit_res in results
                if cproxy.cid in self.adversary_accessed_cids
            ]

        if advers_results == []:
            log(
                INFO,
                "No adversarial clients. The adversary client cids...",
            )

        print(f"extracted {len(advers_results)} adversarial parameters.")

        return advers_results

    def compute_impact(self, attack_acc: float, benign_acc: float) -> float:
        """
        Compute the impact of the attack
        """
        impact = benign_acc - attack_acc
        return impact
