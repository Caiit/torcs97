# TORCS: Group 97
### Computational Intelligence (UvA) course of Artificial Intelligence MSc

To implement a car that can drive autonomously, there was made use of PyTorch to implement a Neural Network, as well as Evolutionary Programming and Swarm Intelligence. The Evolutionary Algorithm was used to optimise the weights of the Neural Network and Swarm Intelligence was based on a 'winner' and a 'helper' model, where the winner would aim to win the race, while the helper would obstruct other cars to prevent them from winning. 

This repo is a TORCS client that connects to a server: https://github.com/mpvharmelen/torcs-server. To run a driver, run `./start.sh`.

The training data can be found in the data folder, or created by setting TRAIN to True in `my_driver.py`.

The trained models can be found in the models folder. To train a model yourself, `torch_model.py` can be executed with the path to the folder of the training data and the name of the output model as arguments.

For evolving an existing model, run `evolve_nn.py`, but make sure to set the path to the model on line 153. Also, change the path to quickrace.xml on line 169.

To run the driver with only a neural network, set both TRAIN and SWARM to False. To run the driver with swarm, set SWARM to True.
