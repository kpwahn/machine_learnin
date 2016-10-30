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
            activation_of_layer.append(neuron.activate(data))

        return activation_of_layer

    def calculate_error_output_neuron(self, neuron, target):
        # error = activation(1 - activation)(activation - target)
        return (neuron.activation * (1 - neuron.activation) * (neuron.activation - target) )

    def calculate_error_hidden_neuron(self, neuron, previous_layer):

        # The sum  of the weights of the neuron on the right times the error of that neruon
        sum_of_weights_times_errors = 0
        for neuron in previous_layer.layer_of_neurons:
            for weight in neuron.weights:
                sum_of_weights_times_errors += weight * neuron.error

        # activation ( 1 - activation) (sum from above)
        return ( neuron.activation * (1 - neuron.activation ) * sum_of_weights_times_errors )

    def update_error_output_layer(self, target):
        # For each neuron in the output layer, update it's error
        for neuron in self.layer_of_neurons:
            # Set the error       Update the error following the output neuron error function
            neuron.set_error(self.calculate_error_output_neuron(neuron, target))

    def update_error_hidden_layer(self, previous_layer):
        # For each neuron in the output layer, go calculate it's error
        for neuron in self.layer_of_neurons:
            neuron.set_error(self.calculate_error_hidden_neuron(neuron, previous_layer))

    def update_weights(self, inputs):
        # For each neuron in the layer, update it's weights
        for index, neuron in enumerate(self.layer_of_neurons):
            neuron.update_weights(inputs)