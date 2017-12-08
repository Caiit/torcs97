#!/usr/bin/python3
import copy
import math
import numpy as np
import operator
import os
import _thread
import threading
import torch
from torch.autograd import Variable

from torch_model import TwoLayerNet
from pytocl.main import main
from my_driver import MyDriver

# Check if cuda is available
CUDA = torch.cuda.is_available()

class Individual:
    '''
    Individual: TwoLayerNet with its own weights and noise.
    '''
    def __init__(self, model=None, noise=None):
        '''
        Initialise individual.
        model: the model of the individual.
        nosie: the noise of the individual.
        '''
        # Set amount of input, hidden and output nodes
        self.d_input = 22
        self.d_hidden = 15
        self.d_output = 2
        # Current score of the model
        self.score = 0
        # Number of weights including bias
        self.number_of_weights = self.d_input * self.d_hidden + self.d_hidden + self.d_hidden * self.d_output + self.d_output
        # If initialised with model, use that as model, otherwise create TwoLayerNet
        if (model):
            self.model = model
        else:
            self.model = TwoLayerNet(self.d_input, self.d_hidden, self.d_output, CUDA)
        # If initialised with noise, use that as noise, otherwise use random noise
        if (noise is None):
            self.noise = np.random.uniform(0.0, 0.2, self.number_of_weights)
        else:
            self.noise = noise
        # Taus used for updating
        self.tau = math.sqrt(2*math.sqrt(self.number_of_weights))**-1
        self.tau_ = math.sqrt(2*self.number_of_weights)**-1

        # Use cuda if available
        if (CUDA):
            self.variable_type = torch.cuda.FloatTensor
        else:
            self.variable_type = torch.FloatTensor

    def setScore(self, score):
        '''
        Set the score of the model.
        score: the score of the network.
        '''
        self.score = score

    def createOffspring(self):
        '''
        Create an offspring from this individual by copying the model
        and evolving the noise and weights.
        '''
        child = Individual(copy.deepcopy(self.model), self.noise.copy())
        child.evolveNoise()
        child.evolveWeights()
        return child

    def evolveNoise(self):
        '''
        Evolve the noise:
        ni'(j) = ni(j) exp(tau'N(0,1) + tauNj(0,1))
        '''
        self.noise = self.noise * np.exp(self.tau_ * np.random.normal(0, 1) + self.tau * np.random.normal(0, 1, self.number_of_weights))

    def evolveWeights(self):
        '''
        Evolve the weights:
        wi'(j) = wi(j) + ni'(j)Nj(0,1)
        '''
        x = self.d_hidden
        noise1_bias = self.noise[:x] * np.random.normal(0, 1, x)

        y = self.d_hidden + self.d_input * self.d_hidden
        noise1_weight = self.noise[x:y] * np.random.normal(0, 1, y - x)

        x = y
        y += self.d_output
        noise2_bias = self.noise[x:y] * np.random.normal(0, 1, y - x)

        x = y
        y += self.d_hidden * self.d_output

        # Transpose since weigth is 2x1
        noise2_weight = np.matrix(self.noise[x:y] * np.random.normal(0, 1, y - x)).T

        self.model.linear1.bias.data += torch.from_numpy(noise1_bias).type(self.variable_type)
        self.model.linear1.weight.data += torch.from_numpy(noise1_weight).type(self.variable_type)
        self.model.linear2.bias.data += torch.from_numpy(noise2_bias).type(self.variable_type)
        self.model.linear2.weight.data += torch.from_numpy(noise2_weight).type(self.variable_type)

    def saveModel(self):
        '''
        Save the model with its score.
        TODO: use variable path
        '''
        torch.save(self.model.state_dict(), "./models/test/model_without_blocking_score_" + str(self.score) + ".pt")

class EP:
    '''
    EP: evolutionary programming based on: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=784219&tag=1
        1. Generate initial population of mu individuals at random and set k=1
        2. Create offspring for each individual:
            for each weight j:
            for each weight j:
            for each weight j:
            for each weight j:
            ni'(j) = ni(j) exp(tau'N(0,1) + tauNj(0,1))
            wi'(j) = wi(j) + ni'(j)Nj(0,1)
            tau = sqrt(2*sqrt(n))^-1
            tau' = sqrt(2n)^-1
        3. Detemine the fitness of every individual
        4. Tournament selection (not really needed)
        5. Stop if halting criterion satisfied, else k++ -> 2
    '''
    def __init__(self, mu, random):
        '''
        Initialise EP
        mu: population size.
        random: True if initialising random model, False if loading model from a file.
        '''
        self.mu = mu
        self.initialisePopulation(random)
        # Use cuda if available
        if (CUDA):
            self.variable_type = torch.cuda.FloatTensor
        else:
            self.variable_type = torch.FloatTensor

    def initialisePopulation(self, random):
        '''
        Initialise the population with mu models.
        random: True if initialising random model, False if loading model from a file.
        TODO: use variable path
        '''
        if (not random):
            model = TwoLayerNet(22, 15, 2, False)
            model.load_state_dict(torch.load("./models/model_without_blocking_22.pt", map_location=lambda storage, loc: storage))
        else:
            model = None

        self.population = []
        for i in range(self.mu):
            self.population.append(Individual(model))

    def testIndividual(self, individual):
        '''
        Test the performance of a model by letting it drive in TORCS.
        individual: the individual to test.
        return: the score of the tested individual.
        TODO: use variable path
        '''
        # Kill all currently running torcs and start new one
        quickrace_path = "~/Documents/uni/master/jaar1/ci/torcs97/quickrace.xml"
        os.system("pkill torcs")
        os.system("torcs -r " + quickrace_path + " > /dev/null &")
        # Start driver with the model of the individual
        driver = MyDriver(individual.model)
        main(driver)
        # Obtain the score of the driver
        score = driver.score
        individual.setScore(score)
        return score

    def testPopulation(self):
        '''
        Test the performance of all individuals in the population.
        '''
        test_population = {}
        for individual in self.population:
            # TODO: hack, if already tested, don't test again to save time
            if (individual.score == 0):
                test_population[individual] = self.testIndividual(individual)
            else:
                test_population[individual] = individual.score
            print("Current score:", test_population[individual])
            individual.saveModel()
        # Sort population including parents and children based on their score
        sorted_population = [p[0] for p in sorted(test_population.items(), key=operator.itemgetter(1))]
        # Use best mu individuals as new population
        self.population = sorted_population[:self.mu]

    def run(self):
        '''
        Run one iteration of EP.
        '''
        self.children = []
        for i in range(self.mu):
            child = self.population[i].createOffspring()
            self.children.append(child)
        self.population += self.children
        self.testPopulation()
        # Save best model of this iteration
        self.population[0].saveModel()


if __name__ == "__main__":
    # Run EP for 100 iterations
    ep = EP(5, False)
    for i in range(100):
        print("i:", i)
        ep.run()
