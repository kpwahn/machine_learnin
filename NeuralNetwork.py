from Layer import Layer
import numpy as np

class NeuralNetwork(object):

    def __init__(self, num_neurons_first_layer, num_of_inputs):
        """Create the Neural Network (initialize one layer, the first layer)"""

        # The neural network is just an array of layers
        self.neural_network = np.array([])

        # Creates a new layer of N neurons with N inputs(FEATURES)
        self.create_new_layer(num_neurons_first_layer, num_of_inputs)

    def connect_layer(self, layer):
        """ Adds the layer to the network by appending it to the neuralNetwork array
        :param layer: A Layer object to be added to the network
        """
        self.neural_network = np.append(self.neural_network, layer)

    def create_new_layer(self, number_of_neurons, number_of_inputs):
        layer = Layer()
        layer.create_layer_of_neurons(number_of_neurons, number_of_inputs)
        self.connect_layer(layer)

    def train(self, train_data, train_target):

        # For each row in my data (features)
        for index, row_of_features in enumerate(train_data):

            # Run the data through each layer of the network
            for layer in self.neural_network:

                # Fire all the neurons in that layer
                # First pass in in the inputs (train_data), then pass in the results from what fire_neurons returns
                row_of_features = layer.fire_neurons(row_of_features)

            # This is what the last row of the network output for that row_of_features
            predictions = row_of_features

            print(train_data[index], ": ", row_of_features)

            for index, prediction in enumerate(predictions):
                predictions[index] = round(prediction, 0)

            # print(predictions)
