from Layer import Layer
import numpy as np

class NeuralNetwork(object):

    def __init__(self, num_neurons_first_layer, num_of_inputs):
        """Create the Neural Network (initialize one layer, the first layer)"""

        # The neural network is just an array of layers
        self.neural_network = np.array([])

        # Creates a new layer of N neurons with N inputs(FEATURES)
        self.create_new_layer(num_neurons_first_layer, num_of_inputs)

        # How accurate is my network?
        self.accuracy = 0

    def connect_layer(self, layer):
        """ Adds the layer to the network by appending it to the neuralNetwork array
        :param layer: A Layer object to be added to the network
        """
        self.neural_network = np.append(self.neural_network, layer)

    def create_new_layer(self, number_of_neurons, number_of_inputs):
        layer = Layer()
        layer.create_layer_of_neurons(number_of_neurons, number_of_inputs)
        self.connect_layer(layer)

    def shuffle_data(self, train_data, train_targets):
        """Shuffle the training data and targets so we can continue to train"""

        indices = np.random.permutation(len(train_data))

        train_data = [train_data[i] for i in indices]
        train_targets = [train_targets[i] for i in indices]

        return train_data, train_targets

    def record_accuracy(self, target, output):

        # Was the max of the output row the neuron we wanted to fire?
        if output.index(max(output)) == target:
            self.accuracy += 1

    def update_errors(self, target):
        # For each layer in my network, from the back to the front
        for index, layer in reversed(list(enumerate(self.neural_network))):
            # Is this the first layer?
            if index == (len(self.neural_network) - 1):
                # Updates the neuron's errors in the output layer
                layer.update_error_output_layer(target)
            else:
                # Updates the neuron's error's in the hidden layers, passing in the previous layer in
                layer.update_error_hidden_layer(self.neural_network[index + 1])

    def update_weights(self, features):

        # For each layer in my network, from the back to the front
        for index, layer in reversed(list(enumerate(self.neural_network))):
            # If this isn't the first layer, we need the inputs (activations of neurons from hidden layers)
            if index != 0:
                activation_features = []
                # Pass in the activations of the last neurons (previous layer) as inputs
                for neuron in self.neural_network[index - 1].layer_of_neurons:
                    activation_features.append(neuron.activation)
                layer.update_weights(activation_features)
            else:
                layer.update_weights(features)

    def learn(self, features, target):
        # Update all the errors
        self.update_errors(target)
        # Update all the weights
        self.update_weights(features)

    def train(self, train_data, train_target):
        # How many times am I going to train this algorithm with my testing data?

        for i in range(100):
            # Reset my accuracy
            self.accuracy = 0

            # For each row in my data (features)
            for index, row_of_features in enumerate(train_data):

                # Run the data through each layer of the network
                for layer in self.neural_network:

                    # Fire all the neurons in that layer
                    # First pass in in the inputs (train_data), then pass in the results from what fire_neurons returns
                    row_of_features = layer.fire_neurons(row_of_features)

                # row_of_features is what the last row of the network output for that row_of_features
                #self.record_accuracy(train_target[index], row_of_features)

                self.learn(train_data[index], train_target[index])

            # Shuffle up my data to continue training
            train_data, train_target = self.shuffle_data(train_data, train_target)


    def predict(self, test_data, test_target):
        self.accuracy = 0

        # For each row in my data (features)
        for index, row_of_features in enumerate(test_data):

            # Run the data through each layer of the network
            for layer in self.neural_network:
                # Fire all the neurons in that layer
                # First pass in in the inputs (train_data), then pass in the results from what fire_neurons returns
                row_of_features = layer.fire_neurons(row_of_features)

            # row_of_features is what the last row of the network output for that row_of_features
            self.record_accuracy(test_target[index], row_of_features)

        print("Testing accuracy: ", self.accuracy / len(test_data))