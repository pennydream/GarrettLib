import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from TrainTestSplit import TrainTestSplit
# Test getAccuracy method of ComputeAccurucy class for classification tasks... returns a float
def test_TrainTestSplit_init():
    """
    Given a pandas dataframe, test the creation of a model class.
    """
    X = pd.DataFrame([
         [1,2],
         [2,4],
         [3,4],
         [2,3]
         ])

    y = pd.DataFrame([ 
         0, 1, 1, 1
         ])

    train_x, train_y, test_x, test_y = TrainTestSplit().split(X,y)
    
    assert X.shape[0] == (train_x.shape[0] + test_x.shape[0])

def test_TrainTestSplit_dtype():
    """
    Test that the initialization of a TrainTestSplit class throws a type error for 
    things that are not pandas dataframes
    """
    X = "Junkie data type" 

    y = pd.Series([ 
         0, 1, 1, 1
         ])

    some_true = "A wrong data type of type string" 
    some_test = 5 #another wrong data type
    with pytest.raises(TypeError):
        TrainTestSplit().split(X,y)
