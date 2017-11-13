import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from Model import Model

# Create a realization of the Model class that can be used for testing
class Helper_Model(Model):
    def train(self):
        return True

# Get init and data stored in Model class 
def test_Model_init():
    """
    Given a pandas dataframe, test the creation of a model class.
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Model(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_Model_dtype():
    """
    Test that the initialization of a Model class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type" 
    with pytest.raises(TypeError):
        Model(some)    
