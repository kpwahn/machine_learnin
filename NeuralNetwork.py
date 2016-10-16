from Layer import Layer
import numpy as np

class NeuralNetwork(object):

    def __init__(self):
        self.nerualNetwork = np.array([])

    def connect_layer(self, layer):
        self.nerualNetwork = np.append(self.nerualNetwork, layer)

    def create_new_layer(self, number_of_neurons, number_of_inputs):
        layer = Layer()
        layer.create_layer_of_neurons(number_of_neurons, number_of_inputs)
        self.connect_layer(layer)

    #take input and produce output (for a single layer)
    def train(self, train_data, train_target):
        self.see_who_fires(train_data)

    def see_who_fires(self, train_data):
        """Checks all the neurons and sees who fires"""
        for layer in self.nerualNetwork:
            for index, neuron in enumerate(layer.layer_of_neurons):
                #print("Neuron", index, end="")

                #each neuron takes the inputs, and decides to fire or not
                for row in train_data:
                    neuron.should_fire(row, index)
                    #have it return something? Maybe later