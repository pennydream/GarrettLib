import numpy as np
import pandas as pd
import random

class TrainTestSplit():
    
    def __init__(self):pass
    
    def split(self, X, y):

        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            pass
        else:
            raise TypeError("Model data is of type Pandas DataFrame.")
        
        n = X.shape[0]
        num_train = int(n*0.666)
        num_test = n - num_train

        #index = np.random.randint(0,n,num_train)
        train_index = random.sample(X.index, num_train)

        print train_index
        test_index = X.index[map(lambda x: False if x in train_index else True, X.index.astype(np.ndarray))]
        
        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]
        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]
        return train_x, train_y, test_x, test_y
