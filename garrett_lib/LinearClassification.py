from abc import ABCMeta, abstractmethod
import pandas as pd
from SupervisedModel import SupervisedModel

class LinearClassification(SupervisedModel):

    
    """
    @abstractmethod
    def test(self):pass
    "If trained, Given new data (pandas dataframe), return the output of the  model (Labels or Values) in a pandas series"

    __metaclass__ = ABCMeta
   
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data 
        else:
            raise TypeError("Model data is of type Pandas DataFrame.")
    
    
    @abstractmethod
    def train(self):pass
    "Given training data, create the model"

    def getData(self):
        return self.data
    """
