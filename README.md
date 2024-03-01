<p align="center">
    <img src="https://github.com/n45os/flwr_attacks/blob/main/misc/flwr-att-lg.png?raw=true" width="140px" alt="modified flwr logo" />
</p>

# Flower Attacks  


![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/n45os/flwr_attacks?label=version)
[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)


This repository is an extension of the Flower Framework that makes possible creating, running, testing and simulating various adversary threats within the FLower Framework. The structure is made very similar to the strategy module to be easy and smooth for someone working with Flower to implement. 

## Features 
- **Federated Learning Attacks**: Implement and simulate various types of attacks on federated learning processes to assess their resilience and security
 
- **Integration with Flower**: Seamlessly integrates with the Flower framework, allowing for easy experimentation and extension. 

- **Extensible Design**: Designed to be easily extended with new types of attacks or modifications to existing ones.  

## Installation 
To install `flwr_attacks`, you can use pip: 

```bash 
pip install flwr_attacks
```

Usage
-----

After installation, you can use `flwr_attacks` as part of your federated learning experiments. Here is a basic example of how to integrate it with your Flower-based federated learning setup:



Configuration for the attack (assuming cfg is an existing configuration object)
```python
from flwr_attacks import MinMax, AttackServer, generate_cids

adversary_cids, benign_cids = generate_cids(NUM_CLIENTS, adversary_fraction=0.4)
all_cids = adversary_cids + benign_cids
```

 Initialize the MinMax attack with your configuration
```python
attack = MinMaxAttack(
    adversary_fraction=0.2,  # 20% of clients are adversaries
    activation_round=5,  # Activate attack at round 5
    adversary_clients=adversary_cids, # by default the attack will be able to access only the adversary clients. Use the argument adversary_accessed_cids to add specific access.
)

strategy = ...

# Create the AttackServer with the specified attack and strategy
attack_server = AttackServer(
    strategy=strategy,
    attack=attack,
)
```
Use the server as in a typical Flower server


Use simulation
```python 
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    clients_ids=all_cids,
    config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
    server=attack_server,
)
```
or start the server
```python 

fl.server.start_server(
    server=attack_server,
)
```

Contributing
------------

Contributions to `flwr_attacks` are welcome! If you have a new attack implementation, improvements or bug fixes, open an issue or a pull request.

License
-------

`flwr_attacks` is released under the Flower's Apache-2.0 License. See the LICENSE file for more details.

Contact
-------

For any questions or feedback, please contact Nassos Bountioukos Spinaris at nassosbountioukos@gmail.com.

Acknowledgments
---------------

Special thanks to the Flower framework team for providing a solid foundation for federated learning experiments.
