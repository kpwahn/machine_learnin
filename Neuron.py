import random
import numpy as np
import math

class Neuron(object):

    def __init__(self, number_of_inputs):
        # Each neuron needs an input (and associated weight) from a bias
        self.bias_input = -1

        # Assign all the weights random values (between -1 and 1)     +1 for the bias weight (we need an extra weight for it)
        self.weights = np.array([random.uniform(-1, 1) for _ in range(number_of_inputs + 1)])

    def activation(self, features):

        sum_of_inputs_and_weights = 0.0

        # For each feature, multiply it by the weight
        for feature_index, feature in enumerate(features):
            sum_of_inputs_and_weights += feature * self.weights[feature_index]

        # Don't forget the bias                                     -1 for the bias weight
        sum_of_inputs_and_weights += self.bias_input * self.weights[len(self.weights) - 1]

        # Run that fancy sigmoid function
        activation = (1 / (1 + np.exp(-(sum_of_inputs_and_weights))) )

        return activation
