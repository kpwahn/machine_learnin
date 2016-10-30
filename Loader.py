from sklearn import datasets
from sklearn.datasets.base import Bunch
import pandas as pd
import numpy as np

class Loader(object):
    def __init__(self, filename):
        self.filename = filename

    def loadData(self):
        if self.filename == "Iris":
            dataset = datasets.load_iris()
            data, target = np.array(dataset.data), np.array(dataset.target)

        if self.filename == "Diabetes":
            dataset = pd.read_csv("C:/Users/Kendall/PycharmProjects/data_mining/diabetes.csv")

            data, target = np.array(dataset.values[:, 0:8]), np.array(dataset.values[:, 8:])

        if self.filename == "Dummy":
            data = np.array([ [0, 0], [1, 1], [2, 2], [3, 3], [88, 88], [100, 100]])
            target = np.array([0, 0, 0, 0, 1, 1])

        # normailize the data - Supposedly
        data = (data - data.min(0)) / data.ptp(0)

        return data, target