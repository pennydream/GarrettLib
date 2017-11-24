import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from Model import Model
from Regression import Regression

# Create a realization of the Regression class that can be used for testing
class Helper_Regression(Regression):
    def train(self):
        return True
    def test(self):
        return True

# Get init and data stored in Regression class
def test_Regression_init():
    """
    Given a pandas dataframe, test the creation of a regression class.
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Regression(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_Regression_dtype():
    """
    Test that the initialization of a regression class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        Regression(some)   

def test_Regression_train():
    """
    Test that regression has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Regression(some)
    assert m.train()


def test_Regression_test():
    """
    Test that regression has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Regression(some)
    assert m.test()


