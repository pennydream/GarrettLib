import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from Model import Model
from SupervisedModel import SupervisedModel

# Create a realization of the SupervisedModel class that can be used for testing
class Helper_SupervisedModel(SupervisedModel):
    def train(self):
        return True
    def test(self):
        return True

# Get init and data stored in SupervisedModel class
def test_SupervisedModel_init():
    """
    Given a pandas dataframe, test the creation of a model class.
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_SupervisedModel(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_SupervisedModel_dtype():
    """
    Test that the initialization of a Model class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        SupervisedModel(some)   

def test_SupervisedModel_train():
    """
    Test that SupervisedModel has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_SupervisedModel(some)
    assert m.train()


def test_SupervisedModel_test():
    """
    Test that SupervisedModel has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_SupervisedModel(some)
    assert m.test()


