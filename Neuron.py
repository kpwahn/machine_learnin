import random
import numpy as np

class Neuron(object):

    def __init__(self, number_of_inputs):
        # Each neuron needs an input (and associated weight) from a bias
        self.bias_input = -1

        # What did the neuron activate with?
        self.activation = 0;

        # The neuron's error
        self.error = 0;

        # Learning rate. 0.1 by default
        self.learning_rate = 0.1

        # Assign all the weights random values (between -1 and 1)     +1 for the bias weight (we need an extra weight for it)
        self.weights = np.array([random.uniform(-1, 1) for _ in range(number_of_inputs + 1)])

    def activate(self, features):

        sum_of_inputs_and_weights = 0.0

        # For each feature, multiply it by the weight
        for feature_index, feature in enumerate(features):
            sum_of_inputs_and_weights += feature * self.weights[feature_index]

        # Don't forget the bias                                     -1 for the bias weight
        sum_of_inputs_and_weights += self.bias_input * self.weights[len(self.weights) - 1]

        # Run that fancy sigmoid function
        activation = (1 / (1 + np.exp(-(sum_of_inputs_and_weights))) )

        self.activation = activation

        return activation

    def set_error(self, new_error):
        self.error = new_error

    def update_weights(self, inputs):

        for index, weight in enumerate(self.weights):
            # Update the weight of the bias
            if index == 0:
                self.weights[index] = ( self.weights[index] - (self.learning_rate * self.error * -1 ) )
            else:
                # Update the rest of the weights
                self.weights[index] = (self.weights[index] - (self.learning_rate * self.error * inputs[index - 1]))