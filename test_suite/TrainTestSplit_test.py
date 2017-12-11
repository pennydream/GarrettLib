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
    some_true = pd.Series([
         0, 1, 1, 0 
         ])

    some_test = pd.Series([ 
         0, 1, 1, 1
         ])

    m = TrainTestSplit(some_true, some_test)

    assert m.getAccuracy("classification") == 0.75 

def test_TrainTestSplit_dtype():
    """
    Test that the initialization of a TrainTestSplit class throws a type error for 
    things that are not pandas dataframes
    """
    some_true = "A wrong data type of type string" 
    some_test = 5 #another wrong data type
    with pytest.raises(TypeError):
        TrainTestSplit().split(X,y)
