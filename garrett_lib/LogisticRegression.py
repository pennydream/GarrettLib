from abc import ABCMeta, abstractmethod
import pandas as pd
from Classification import Classification
from ComputeAccuracy import ComputeAccuracy
import numpy as np

class LogisticRegression(Classification):


    def train(self, y, num_iters=150):
        train_x = self.data.as_matrix()
        train_y = y.as_matrix()[:,0]
        train_x = np.c_[train_x, np.ones(train_x.shape[0])]
        w = np.array(map(lambda x: 1 if x>0 else -1, np.random.randn(train_x.shape[1])))
        alpha = 0.01
        pocket = -1
    
        for i in range(0, num_iters):
            pred_num_y = map(lambda x: 1/(1+np.exp(-np.dot(w,x))), train_x)
            pred_y = np.array(map(lambda x: 1 if x>0 else -1, pred_num_y))        
            size = train_y.shape[0]
            correct = (train_y == pred_y).sum()
            acc = (correct.astype(np.float) / size) * 100
            w = w-np.dot(alpha*(pred_y - train_y), train_x)

            if(acc > pocket):
                pocket_w = w
                pocket = acc
        self.w = pocket_w
        return True

    def test(self, new_data):
        w = self.w
        test_x = new_data.as_matrix()
        test_x = np.c_[test_x, np.ones(test_x.shape[0])]
        temp = np.array(map(lambda x: 1/(1+np.exp(-np.dot(w,x))), test_x))
        return pd.Series(map(lambda x: 1 if x>0 else -1, temp))

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
