from abc import ABCMeta, abstractmethod
import pandas as pd
from Classification import Classification
from ComputeAccuracy import ComputeAccuracy
import numpy as np

class LogisticRegression(Classification):


    def train(self, labels, num_iters = 100):
        self.labels = labels

        train_x = self.data
        train_y = self.labels

        train_x = train_x.as_matrix()
        train_y = train_y.as_matrix()

        # Add a constant to the training data
        train_x = np.c_[train_x, np.ones(train_x.shape[0])]
        index = np.arange(train_x.shape[0])

        # Initialize w with random values
        w = np.array(map(lambda x: 1 if x>0 else -1, np.random.randn(train_x.shape[1])))

        # Get the predictions for the initialized w
        pred_y_inner = np.array(map(lambda x: 1 if np.dot(w,x)>0 else -1, train_x))
        pocket = 0

        for i in range(0, num_iters):
            # Get accuracy for each iteration
            ca = ComputeAccuracy(pd.DataFrame(train_y)[0], pd.DataFrame(pred_y_inner)[0])
            acc = ca.getAccuracy("classification")
            print acc
            #acc = compute_accuracy(train_y, pred_y_inner)

            # Check if accuracy is better than previous one. If so, save it
            if acc > pocket:
                pocket = acc
                pocket_w = w

            # Check if accuracy on training data is meant. If so, exit.
            if acc >= 1.0:
                self.w = w
                return True;

            # Choose a random misclassified point.
            random_sample_index = np.random.choice(index[pred_y_inner != train_y[:,0]])

            # Update the weights using w <- w + xy
            w = w + 0.01*(train_x[random_sample_index]*train_y[random_sample_index])
            # Update the prediction for this iteration
            pred_y_inner = np.array(map(lambda x: 1 if np.dot(w,x)>0 else -1, train_x))

        # Return the saved weights with the highest accuracy
        self.w = pocket_w

        return True

    def test(self, test_x):
        test_index = test_x.index
        test_x = test_x.as_matrix()
        test_x = np.c_[test_x, np.ones(test_x.shape[0])]
        ret = pd.Series(np.array(map(lambda x: 1 if np.dot(self.w,x)>0 else -1, test_x)))
        ret.index = test_index
        return ret

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
