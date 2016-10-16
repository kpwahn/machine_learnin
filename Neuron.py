import random
import numpy as np

class Neuron(object):

    #DONT FORGET AN INPUT FOR THE BIAS. IT ALSO HAS IT'S OWN WEIGHT

    def __init__(self, number_of_inputs):
        self.bias_input = -1
        #assign all the weights randomly (between -1 and 1)            #+1 for the bias weight (we need an extra weight)
        self.weights = np.array([random.uniform(-1, 1) for num in range(number_of_inputs + 1)])

    def should_fire(self, row, index):
        sum_of_inputs_and_weights = 0.0

        for feature_index, feature in enumerate(row):
            sum_of_inputs_and_weights += feature * self.weights[feature_index]

        #Don't forget the bias                                           #-1 for the bias weight
        sum_of_inputs_and_weights += self.bias_input * self.weights[len(self.weights) - 1]

        #Should each neuron have a threshold? Each layer? Hardcoding for now
        print("Neuron", index, end="")
        if sum_of_inputs_and_weights <= 0:
            fired = True
            print(" DID NOT FIRE ", end="")
        else:
            fired = False
            print(" FIRED ", end="")
        print("on", row, "With a value of: ", sum_of_inputs_and_weights)
        #check against targets? Why bother, no learning is happening yet