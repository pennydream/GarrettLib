from abc import ABCMeta, abstractmethod
import pandas as pd
from SupervisedModel import SupervisedModel
from ComputeAccuracy import ComputeAccuracy
import numpy as np

class CrossValidation(SupervisedModel):

    def getAccuracy(self, y_1, y_2, task="regression"): 
        
        cm = ComputeAccuracy(y_1, y_2)
        return cm.getAccuracy(task)
        
    "Given labels stored in data_true and data_test and the type of the model, get accuracy of the model"

    def run(self, Model, labels, task="regression", cv=3):

        if cv<2 :
            print "cv must be at least 2."
            return False
        index = np.array(self.data.index.copy())
        np.random.shuffle(index)
        accuracies = []
        for test_index in np.split(index,cv):
            
            train_index = self.data.index[map(lambda x: False if x in test_index else True, self.data.index.astype(np.ndarray))]

            train_x = pd.DataFrame(self.data.iloc[train_index])
            train_y = pd.DataFrame(labels.iloc[train_index])
            test_x = pd.DataFrame(self.data.iloc[test_index])
            test_y = pd.DataFrame(labels.iloc[test_index])
            
            model = Model(train_x)
            model.train(train_y)
            accuracies.append(self.getAccuracy(model.test(test_x)[0], test_y[0]))
            
        return np.array(accuracies)
    
    def train(self): return True 
    def test(self): return True 
