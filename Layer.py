from Neuron import Neuron
import numpy as np

class Layer(object):

    def __init__(self):
        self.layer_of_neurons = np.array([])

    def create_layer_of_neurons(self, number_of_neurons, number_of_inputs):
        """ Creates a layer of neurons. The number of neurons is specified by parameter """

        for i in range(number_of_neurons):
            self.layer_of_neurons = np.append(self.layer_of_neurons, Neuron(number_of_inputs))

    def fire_neurons(self, data):

        # An array of what the neurons in the layer returned
        activation_of_layer = []

        # For each neuron in this layer, go run it's activation function
        for neuron in self.layer_of_neurons:
            activation_of_layer.append( neuron.activation(data) )

        return activation_of_layer
