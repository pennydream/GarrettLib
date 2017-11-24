import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from Model import Model
from Classification import Classification

# Create a realization of the Classification class that can be used for testing
class Helper_Classification(Classification):
    def train(self):
        return True
    def test(self):
        return True

# Get init and data stored in Classification class
def test_Classification_init():
    """
    Given a pandas dataframe, test the creation of a classification class.
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Classification(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_Classification_dtype():
    """
    Test that the initialization of a Classification class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        Classification(some)   

def test_Classification_train():
    """
    Test that Classification has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Classification(some)
    assert m.train()


def test_Classification_test():
    """
    Test that Classification has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Classification(some)
    assert m.test()


