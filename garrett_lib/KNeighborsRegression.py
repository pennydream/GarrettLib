from abc import ABCMeta, abstractmethod
import pandas as pd
from Regression import Regression 
import numpy as np

class KNeighborsRegression(Regression):
    
    def test(self, new_data, k):
 
        test_x = new_data
        train_x = self.data
        train_y = self.labels

        ret_index = test_x.index
        train_x = train_x.as_matrix()
        train_y = train_y.as_matrix()
        test_x = test_x.as_matrix()
    
        # Find the distance between all test points and all train points.
        distance_matrix = np.array(map(lambda x: np.square(train_x - x).sum(axis=1), test_x))
        # Get the indexes of the top k points for each test point.
        index = distance_matrix.argsort()[:,:k]  
        # Use the top k test points to decide the predicted test point
        pred_y = []
        for line in index:
            pred_y.append(train_y[line].mean())
            #u, counts = np.unique(train_y[line],return_inverse=True)
            #indx = np.bincount(counts).argmax()
            #pred_y += u[indx]
        ret = pd.DataFrame(pred_y)
        ret.index = ret_index
        return ret 

    "If trained, Given new data (pandas dataframe), return the output of the  model (Labels or Values) in a pandas series"
 
    def train(self, labels):
        self.labels = labels
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
