import numpy as np
import pandas as pd

class TrainTestSplit():
    
    def __init__(self):pass
    
    def split(self, X, y):
        n = len(X) 
        num_train = int(n*0.666)
        num_test = n - num_train

        index = np.random.randint(0,n,num_train)
        test_index = X.index[map(lambda x: True if x in index else False, X.index.astype(np.ndarray))]
        
        train_x = X.iloc[index]
        train_y = y.iloc[index]
        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]
        return train_x, train_y, test_x, test_y
