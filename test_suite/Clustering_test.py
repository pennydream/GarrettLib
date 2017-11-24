import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from Model import Model
from Clustering import Clustering

# Create a realization of the Clustering class that can be used for testing
class Helper_Clustering(Clustering):
    def train(self):
        return True
    def convert(self):
        return True

# Get init and data stored in Clustering class
def test_Clustering_init():
    """
    Given a pandas dataframe, test the creation of a Clustering class.
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Clustering(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_Clustering_dtype():
    """
    Test that the initialization of a Clustering class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        Clustering(some)   

def train_Clustering_train():
    """
    Test that Clustering has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Clustering(some)
    assert m.train()


def test_Clustering_convert():
    """
    Test that Clustering has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = Helper_Clustering(some)
    assert m.convert()


