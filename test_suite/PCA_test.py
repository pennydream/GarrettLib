import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from Model import Model
from PCA import PCA

# Get init and data stored in PCA class
def test_PCA_init():
    """
    Given a pandas dataframe, test the creation of a PCA class.
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = PCA(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_PCA_dtype():
    """
    Test that the initialization of a PCA class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        PCA(some)   

def train_PCA_train():
    """
    Test that PCA has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = PCA(some)
    assert m.train()


def test_PCA_convert():
    """
    Test that PCA has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = PCA(some)
    m.train(2)
    results = m.convert(some)
    assert results.shape[0] == 4
    assert results.shape[1] == 2
    


