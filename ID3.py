import numpy as np
from Node import Node
from sklearn.datasets.base import Bunch

class ID3(object):
    def __init__(self):
        self.tree = {}

    def reallyCalculateEntropy(self, data, weight):

        entropy = 0
        total = data["total"]
        #print(data)
        del data["total"]

        for i in data:
            value = - ( (data[i]/total) * np.log2(data[i]/total) if data[i] != 0 else 0 )
            entropy += value

        #print(entropy)
        #weight the entropy
        return((total/weight * (entropy)))

    def createTargetKeys(self, total):
        array = {}
        #create dictionary keys of possible targets
        for i in self.possible_target_values:
            array[i] = 0
        array["total"] = total

        return array

    def calculateEntropy(self, data):
        total = 0
        entropy = 0

        for i in data:
            total += len(data[i])

        #loops through the possible values of the features (i.e. "y", "n", ?"
        for i in data:
            array = self.createTargetKeys(len(data[i]))
            for j in data[i]:
                for k in self.possible_target_values:
                    if j[0] == k:
                        array[k] += 1
            #finds the entropy of each branch (the data that would go there)
            entropy += self.reallyCalculateEntropy(array, total)
        return entropy

    def createFeatureKeys(self, num_features_left, i):
        split_data = {}

        # create keys for dictionary of possible features
        for h in range(len(self.possible_column_values[i])):
            split_data[self.possible_column_values[i][h]] = []
        return split_data

    def findBestAttribute(self, train_data, train_target, num_features_left):

        entropies = []
        column_keys = []
        # loop through all features, and split
        for i in range(num_features_left):
            #creates a dictionary with the key words being what's possible for that given feature
            split_data = self.createFeatureKeys(num_features_left, i)
            #splits the data based on the value of the attribute
            for index, j in enumerate(train_data):
                 for h in range(len(self.possible_column_values[i])):
                    if j[i] == self.possible_column_values[i][h]:
                        #assigns it to the dictionary (it's target values)
                        split_data[self.possible_column_values[i][h]].append( train_target[index] )
            #now I know how many people chose each possible value (i.e. 14 people chose "y", 12 "n", 2 "?") and how
            #which target they were
            entropies.append(self.calculateEntropy(split_data))

            # then delete it arrays in split_data
            del split_data
            print(min(entropies))
        #this is the index of the column of the attribute we are splitting on
        return entropies.index(min(entropies)), self.possible_column_values[entropies.index(min(entropies))]

    def createTree(self, train_data, train_target, num_features_left):
            print(num_features_left)
            #if all examples have the same label
            if all(x == train_target[0] for x in train_target):
                node = Node(0, [train_target[0]])
            #else if no features left to test
            elif num_features_left <= 0:
                # return a leaf with the most common label
                node = Node(0, [max(set(train_target), key=train_target.count)])
            else:
                #consider each available feature
                #choose one that maximized info gain
                feature_index, key = self.findBestAttribute(train_data, train_target, num_features_left)
                #create a node for that feature
                node = Node(feature_index, key)

                #for each possible value of the feature
                for i in node.bunch:
                    #create a subset of the examples for this branch
                    subset_train_data = [[]]
                    subset_train_target = []
                    for index, j in enumerate(train_data):
                        if j[feature_index] == i:
                            subset_train_data.append(j)
                            subset_train_target.append(train_target[index])
                            #i need to pass this data recursively as the training data

                    #recursively call this function to create a new node for that branch
                    num_features_left -= 1
                    self.createTree(subset_train_data, subset_train_target, num_features_left)

    def buildTree(self, train_data, train_target, possible_column_values, possible_target_values):
        self.numAttributes = len(train_data[0])
        self.possible_column_values = possible_column_values
        self.possible_target_values = possible_target_values

        self.createTree(train_data, train_target, len(train_data[0]))