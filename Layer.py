from Neuron import Neuron
import numpy as np

class Layer(object):

    def __init__(self):
        self.layer_of_neurons = np.array([])

    def create_layer_of_neurons(self, number_of_neurons, number_of_inputs):
        for i in range(number_of_neurons):
            self.layer_of_neurons = np.append(self.layer_of_neurons, Neuron(number_of_inputs))