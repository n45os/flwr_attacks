import numpy as np
from typing import List

def generate_cids(
    num_of_clients: int, adversary_fraction: float
) -> List[str]:
    """Generate the list of malicious clients.

    Parameters
    ----------
    num_of_clients : int
        The number of clients in the system.
    adversary_fraction : float
        The fraction of clients which are malicious.

    Returns
    -------
    Tuple[List[str], List[str]]
        The list of malicious client ids and the list of benign client ids. The malicious client ids have "advers" in them .The benign client ids have "benign" in them.
    """
    
    benign_count = num_of_clients - (int(num_of_clients * adversary_fraction))
    adversaries_count = int(num_of_clients * adversary_fraction)
    benign_cids = [x for x in range(int(benign_count))]
    adversaries_cid = [
        x
        for x in range(
            int(benign_count), int(benign_count) + int(adversaries_count)
        )
    ]

    return adversaries_cid, benign_cids

def edit_cids(adversary_fraction: float, all_cids: List[str]) -> List[str]:
    """Edit the list of malicious clients cid to add "advers" in it and make it easier to identify them.

    Parameters
    ----------
    adversary_fraction : float
        The fraction of clients which are malicious.
    all_cids : List[str]
        The list of all client ids.

    Returns
    -------
    List[str]
        The list of malicious client ids.
    """

    # add "advers_" on the start of the last int(len(all_cids) * adversary_fraction) cids
    # for example if we have 10 clients and 0.2 adversary fraction, we will have 2 adversarial clients
    # if the cids are a list ["zero", "1", "2", "3", "four", "5", "6", "7-something", "eight", "9"]
    # the edited cids will be a list ["zero", "1", "2", "3", "four", "5", "6", "7-something", "advers_eight", "advers_9"]

    # get the number of adversarial clients
    num_of_adversaries = int(len(all_cids) * adversary_fraction)
    # get the cids of the adversarial clients
    adversaries_cids = all_cids[-num_of_adversaries:]
    # add "advers_" on the start of the cids
    adversaries_cids = [f"advers_{cid}" for cid in adversaries_cids]
    # replace the cids of the adversarial clients
    all_cids[-num_of_adversaries:] = adversaries_cids
    return all_cids