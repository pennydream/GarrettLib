import pytest
import numpy as np
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from Model import Model
from KMeans import KMeans

# Get init and data stored in KMeans class
def test_KMeans_init():
    """
    Given a pandas dataframe, test the creation of a KMeans class.
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = KMeans(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_KMeans_dtype():
    """
    Test that the initialization of a KMeans class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        KMeans(some)   

def train_KMeans_train():
    """
    Test that KMeans has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = KMeans(some)
    assert m.train()


def test_KMeans_convert():
    """
    Test that KMeans has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = KMeans(some)
    assert m.convert(3)

def test_KMeans_distance():
    """
    test that finding the sum of squared distance is correct
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = KMeans(some)
    
    x = pd.Series([1,2])
    y = pd.Series([1,4])

    assert np.sqrt(4/2) == m.distance(x,y)
