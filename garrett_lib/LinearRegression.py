from abc import ABCMeta, abstractmethod
import pandas as pd
from Regression import Regression 
import numpy as np

class LinearRegression(Regression):
    
    def test(self, new_data):
    
        # Multiply weights self.w with new_data

        return pd.DataFrame(self.w.T.dot(new_data).T)
    "If trained, Given new data (pandas dataframe), return the output of the  model (Labels or Values) in a pandas series"

   


    def train(self, labels):
        
        # We have training data X and labels y...
        # w = (XtX)^-1(Xty)
        
        XtX = self.data.dot(self.data.T)

        Xty = self.data.dot(labels)

        piXtX = np.linalg.pinv(XtX)

        self.w = np.dot(piXtX, Xty)

        return True
    "Given training data, create the model"

    """
    def getData(self):
        return self.data
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data 
        else:
            raise TypeError("Model data is of type Pandas DataFrame.")
    """
