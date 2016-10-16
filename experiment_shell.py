from NeuralNetwork import NeuralNetwork
from Loader import Loader
import numpy as np
import sys


def split_data(data, target, split_amount):

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

def processData(data, target, split_amount):
    train_data, train_target, test_data, test_target = split_data(data, target, split_amount)

    #create my empty network
    neuralNetwork = NeuralNetwork()
    #create a new layer of N neurons with N inputs(FEATURES)
    neuralNetwork.create_new_layer(2, len(train_data[0]))
    #we only create one layer, but give us some outputs
    neuralNetwork.train(train_data, train_target)

def main(argv):

    data, target = Loader("Dummy").loadData()
    processData(data, target,  0.7)

if __name__ == "__main__":
    main(sys.argv)