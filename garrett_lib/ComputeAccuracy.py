from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from AccuracyMeasure import AccuracyMeasure

class ComputeAccuracy(AccuracyMeasure):

    def __init__(self, data_true, data_test):
        if (isinstance(data_true, pd.Series) and isinstance(data_test, pd.Series)):
            self.data_true = data_true

            self.data_test = data_test

        else:
            raise TypeError("Data is of type Pandas DataFrame.")

    def getAccuracy(self, task, loss="reg"):
        """
        Returns the percent correct for classification task or
        the sum of squared error for regression tasif(self.data_true[i] == self.data_test[i]):
                    true+=1ks
        """
        self.data_true.index = range(0,len(self.data_true))
        self.data_test.index = range(0,len(self.data_test))
        if task is "classification":
            true = 0
            length = self.data_true.shape[0]
            for i in range(0,length):
                if(self.data_true[i] == self.data_test[i]):
                    true+=1
            return float(true)/length

        if task is "regression":
            true = 0
            length = self.data_true.shape[0]
            for i in range(0,length):
                true += np.square(self.data_true[i] - self.data_test[i])
            return np.sqrt(float(true)/length)

        else:
            return True
    "Given labels stored in data_true and data_test and the type of the model, get accuracy of the model"

    def getDataTrue(self):
        return self.data_true
    def getDataTest(self):
        return self.data_test
