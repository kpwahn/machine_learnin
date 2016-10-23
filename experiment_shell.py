from NeuralNetwork import NeuralNetwork
from Loader import Loader
import numpy as np
import sys

def split_and_process_data(data, target, split_amount):
    """ Splits the data into training and testing data and targets"""

    # How many is __%(split_amount) of my dataset
    split_index = int( split_amount * len(data) )

    # Create a list of random indicies (as many indicies as data in my set)
    indices = np.random.permutation(len(data))

    # indices[:split_index] - A list of random indicies from 0 to the split_index - i.e. [127, 53, ..., 2]
    # dataset.data[ **list from above** ] - Creating a new list of just the indexes of dataset.data specified in the **list from above**
    #                                     - i.e. [ dataset.data[127], dataset.data[53], ..., dataset.data[2]
    train_data = data[ indices[:split_index] ]
    train_target = target[indices[:split_index]]

    # Same as above, but from the split_index to the end of the list
    test_data = data[ indices[split_index:] ]
    test_target = target[ indices[split_index:] ]

    return (train_data, train_target, test_data, test_target)

def neural_network(train_data, train_target, test_data, test_target):
    """This method implements the neural network for our machine learning algorithm"""

    # Number of features/inputs to our network
    num_features = len(train_data[0])
    # How many nodes in each layer?
    num_neurons_per_layer = [num_features, 3, 2]
    #  How many layers?
    num_layers = len(num_neurons_per_layer)


    if num_layers == len(num_neurons_per_layer):
        # create my network. Creates the first layer of N neurons with N inputs(FEATURES)
        neuralNetwork = NeuralNetwork(num_neurons_per_layer[0], num_features)

        # Create the rest of the layers (Start at one becuase we already made the first layer)
        for layer in range(1, num_layers):
                                    #(The num of neurons for that layer, the number of inputs is the number of neurons of the last layer)
            neuralNetwork.create_new_layer(num_neurons_per_layer[layer], len(neuralNetwork.neural_network[layer - 1].layer_of_neurons))

        neuralNetwork.train(train_data, train_target)
    else:
        print("You need to specify 'number of nodes per layer' for each layer")

def main(argv):

    # Load the data of the specified dataset
    data, target = Loader("Dummy").loadData()
    train_data, train_target, test_data, test_target = split_and_process_data(data, target,  0.7)

    # Which algorithm would you like?
    neural_network(train_data, train_target, test_data, test_target)

if __name__ == "__main__":
    main(sys.argv)