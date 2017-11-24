from abc import ABCMeta, abstractmethod
import pandas as pd

class AccuracyMeasure(object):

    __metaclass__ = ABCMeta
   
    def __init__(self, data_true, data_test):
        if isinstance(data_true, pd.Series) and isinstance(data_test, pd.Series):
            self.data_true = data_true
            self.data_test = data_test
        else:
            raise TypeError("Data is of type Pandas DataFrame.")

    
    @abstractmethod
    def getAccuracy(self):pass
    "Given labels stored in data_true and data_test and the type of the model, get accuracy of the model"

    def getDataTrue(self):
        return self.data_true
    def getDataTest(self):
        return self.data_test
