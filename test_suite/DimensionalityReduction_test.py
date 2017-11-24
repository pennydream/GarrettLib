import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from Model import Model
from DimensionalityReduction import DimensionalityReduction

# Create a realization of the DimensionalityReduction class that can be used for testing
class Helper_DimensionalityReduction(DimensionalityReduction):
    def train(self):
        return True
    def convert(self):
        return True

# Get init and data stored in DimensionalityReduction class
def test_DimensionalityReduction_init():
    """
    Given a pandas dataframe, test the creation of a DimensionalityReduction class.
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_DimensionalityReduction(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_DimensionalityReduction_dtype():
    """
    Test that the initialization of a DimensionalityReduction class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        DimensionalityReduction(some)   

def train_DimensionalityReduction_train():
    """
    Test that DimensionalityReduction has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_DimensionalityReduction(some)
    assert m.train()


def test_DimensionalityReduction_convert():
    """
    Test that DimensionalityReduction has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_DimensionalityReduction(some)
    assert m.convert()


